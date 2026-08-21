"""親のfollow-in labeling（F1-3） — 太郎の注視に随伴させて、物の名前を言う。

【設計】F/docs/仕様_F1-3_親のfollow-in labelingと耳の配線.md（前半：人間側の設計意図、
後半：技術付録）。このクラスは付録の「部品1：環境側」が指す `ParentLabeling` の本体。

【役割分担】プラグインではない（`run/plugins/base.py` の「読むだけ」規約の対象外）。
E/scripts/e_toy_env.py（ToySupineEnv）が保持し、既存の親介入（`_parent_intervene`）と
同じ「環境側の住人」として振る舞う＝おもちゃの位置を動かし、視線を読み、情報(info)を返す。
1600行のenv本体をこれ以上太らせないため独立ファイルに分離した（ユーザー指示）。

【状態機械】選ぶ（ランダム）→振る→注視確認（gaze_deg以内がgaze_hold_sec継続、かつ
gaze_margin_deg以上の差で「近い方」と判定できること）→発話→不応期→次へ。
attract_timeout_secまでに注視が来なければ、発話せず黙って次の試行へ（見ていないのに
名前だけ聞く、という汚れたデータを作らないため＝前半の設計意図どおり）。

【左右入れ替え】発話がshuffle_after回に達するたびに、e_toy_env._set_anchor()の
toy1/toy2角度符号を反転させて呼び直す（場所の丸暗記を防ぐ仕掛け。前半参照）。

【gaze_margin_degの根拠】[Tier3・幾何からの導出] toy1/toy2は視線正面から左右に
toy_angle_deg度ずつ振り分けて配置される（既定12度・間隔24度）。球面三角不等式
|θ1-θ2| <= 間隔 <= θ1+θ2 より、近い方への角度θ1がgaze_deg(10度)以内なら、
もう一方への角度θ2は「間隔-θ1 = 24-10 = 14度」以上が保証される（margin>=4度）。
既定値3.0度はこの保証(4度)より安全側に低く取った値。gaze_deg・toy_angle_degを
大きく変える実験では、この保証が成り立たなくなる可能性がある点に注意。
"""
import csv
import os

import numpy as np


class ParentLabeling:
    """親の状態機械。ToySupineEnv.step() から毎物理stepぶん update(env) を呼ばれる。

    Args（実験ファイルの `world.parent_labeling` から kwargs直渡し。技術付録の値が既定）:
        enabled              既定False。Falseなら update() は常に None を返す（無効化）。
        shake_amp_m          おもちゃを振る振幅[m]（F1-3aの決定的検証と同条件）
        shake_hz             振る周波数[Hz]（同上）
        gaze_deg             「見ている」とみなす角度[度]（実測：吸い付き時の中央値8.1度）
        gaze_margin_deg      近い方/遠い方の角度差の下限[度]（曖昧域対策。上のdocstring参照）
        gaze_hold_sec        gaze_deg以内が何秒続いたら「見た」とみなすか
        attract_timeout_sec  振ってから諦めるまでの秒数
        utterances           {"toy1": 語, "toy2": 語}
        respond_prob         注視が確認できたときに実際に発話する確率
        refractory_sec       発話後、黙る秒数（連呼防止）
        shuffle_refractory_sec  左右入れ替え直後、黙る秒数（親がおもちゃを動かしている間）
        shuffle_after        発話が何回に達するごとに左右を入れ替えるか
        solo_presentation    既定False。Trueなら、振って名指しする間、もう片方の
                             おもちゃを視界外（FAR_AWAY）へ退避させる＝1個ずつ見せる提示。
                             【F1-3c・2026-08-19】2個並べたままだと、発話瞬間の視覚に
                             両方のおもちゃが同時に写り、どちらの語のときもほぼ同じ絵に
                             なって連合が分かれない（目視で確認）。実際の親も1個だけ
                             目の前に持ってきて名づけるので、人間模倣としても自然。
                             不応期には両方が戻る（親が持ち替える動作に相当）。
    """

    _PICK = "pick"
    _SHAKE = "shake"
    _REFRACTORY = "refractory"

    def __init__(self, enabled=False, shake_amp_m=0.02, shake_hz=1.5,
                 gaze_deg=10.0, gaze_margin_deg=3.0, gaze_hold_sec=0.3,
                 attract_timeout_sec=3.0, utterances=None, respond_prob=1.0,
                 refractory_sec=1.0, shuffle_refractory_sec=3.0, shuffle_after=5,
                 solo_presentation=False):
        self.enabled = bool(enabled)
        self.shake_amp_m = float(shake_amp_m)
        self.shake_hz = float(shake_hz)
        self.gaze_deg = float(gaze_deg)
        self.gaze_margin_deg = float(gaze_margin_deg)
        self.gaze_hold_sec = float(gaze_hold_sec)
        self.attract_timeout_sec = float(attract_timeout_sec)
        self.utterances = dict(utterances or {"toy1": "ぶーぶー", "toy2": "わんわん"})
        self.respond_prob = float(respond_prob)
        self.refractory_sec = float(refractory_sec)
        self.shuffle_refractory_sec = float(shuffle_refractory_sec)
        self.shuffle_after = int(shuffle_after)
        self.solo_presentation = bool(solo_presentation)
        self.reset()

    def reset(self):
        """エピソード境界で呼ぶ（前エピソードの状態を持ち越さない。e_orienting.reset()と同じ位置づけ）。"""
        self._state = self._PICK
        self._target = None
        self._shake_t = 0.0
        self._hold_t = 0.0
        self._silent_t = 0.0
        self._silent_until = 0.0
        self.n_utterances = 0     # 累計発話回数（testのshuffle判定にも使う）
        self.n_swaps = 0          # 累計入れ替え回数

    # ------------------------------------------------------------------
    def update(self, env):
        """1物理stepぶん進める。発話が起きたstepだけ {"text","target"} を返す（他はNone）。

        env: ToySupineEnv自身（self）。位置の読み書き・_gaze_angle_to・_set_anchor
             ・np_random（乱数はenvのnp_randomを使う＝依頼書の指示）を使う。
        """
        if not self.enabled:
            return None
        if not (getattr(env, "_toy", False) and getattr(env, "_toy2", False)):
            return None      # toy1・toy2の両方が無いと「follow-in labeling」が成立しない
        dt = float(env.dt)

        if self._state == self._REFRACTORY:
            self._silent_t += dt
            if self._silent_t < self._silent_until:
                return None
            self._state = self._PICK

        if self._state == self._PICK:
            self._target = "toy1" if env.np_random.random() < 0.5 else "toy2"
            self._state = self._SHAKE
            self._shake_t = 0.0
            self._hold_t = 0.0

        # ここに来るのは常に SHAKE 状態（振っている最中）
        self._apply_shake(env)
        if self.solo_presentation:
            self._hide_other(env)
        if self._gaze_on_target(env):
            self._hold_t += dt
        else:
            self._hold_t = 0.0     # 連続でないとカウントしない（「続いたら」の判定）
        self._shake_t += dt

        if self._hold_t >= self.gaze_hold_sec:
            return self._speak(env)
        if self._shake_t >= self.attract_timeout_sec:
            self._state = self._PICK   # 諦めて黙って次の試行へ（前半の設計意図どおり）
        return None

    # ------------------------------------------------------------------ 内部
    def _gaze_on_target(self, env):
        """今の視線が self._target を「見ている」と判定できるか。

        判定：近い方のおもちゃであり、かつ他方との角度差が gaze_margin_deg 以上
        （2おもちゃの間隔24度に対する曖昧域対策。docstring冒頭の導出参照）。
        """
        body = "test_object1" if self._target == "toy1" else "test_object2"
        other = "test_object2" if self._target == "toy1" else "test_object1"
        a_t = env._gaze_angle_to(body)
        a_o = env._gaze_angle_to(other)
        if a_t is None or a_o is None:
            return False
        return a_t <= self.gaze_deg and (a_o - a_t) >= self.gaze_margin_deg

    def _apply_shake(self, env):
        """今振っているおもちゃの位置に、正弦波の揺れを加える（F1-3aの決定的検証と同条件）。

        揺れの軸は片目カメラのローカルx軸（「右」方向）＝_set_anchorの左右振り分けと同じ軸。
        中心（base）は env._rest_pos / _rest_pos2（変更しない）＝揺れは表示上の上乗せだけで、
        毎stepの _apply_tether / _hold_toy2 が置いた位置を、このメソッドが上書きする形になる
        （step()内の呼び出し順：_hold_toy2の直後にupdate()を呼ぶ設計）。
        """
        cid = int(env.model.camera("eye_left").id)
        right = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 0]
        wob = right * self.shake_amp_m * np.sin(2 * np.pi * self.shake_hz * self._shake_t)
        if self._target == "toy1":
            base = getattr(env, "_rest_pos", None)
            if base is None:
                return
            env._place(env._toy_qadr, np.array(base, dtype=float) + wob)
            env.data.qvel[env._toy_dadr:env._toy_dadr + 6] = 0.0
        else:
            base = getattr(env, "_rest_pos2", None)
            if base is None:
                return
            env._place(env._obj2_qadr, np.array(base, dtype=float) + wob)
            env.data.qvel[env._obj2_dadr:env._obj2_dadr + 6] = 0.0

    def _hide_other(self, env):
        """名指ししていない方のおもちゃを視界外へ退避させる（solo_presentation時のみ）。

        _apply_shake と同じく「毎stepの _apply_tether / _hold_toy2 が置いた位置を
        上書きする」流儀。SHAKE状態の間だけ呼ばれるので、不応期に入れば
        次のstepから環境側の通常配置（両方見える）へ自動的に戻る。
        退避先は e_toy_env.FAR_AWAY（未使用物体の既存の退避先）＋オフセット
        （toy2無効時の退避位置 FAR_AWAY+[0.5,0,0] と重ねないため）。
        """
        from e_toy_env import FAR_AWAY   # 循環import回避のため関数内で読む
        #   （e_toy_env側の `from parent_labeling import ...` と同じ素の名前で読む
        #    ＝E/scripts がsys.path上にある前提。読み込み済みの同一モジュールを指す）
        if self._target == "toy1":
            qadr, dadr = env._obj2_qadr, env._obj2_dadr
        else:
            qadr, dadr = env._toy_qadr, env._toy_dadr
        env._place(qadr, FAR_AWAY + np.array([1.0, 0.0, 0.0]))
        env.data.qvel[dadr:dadr + 6] = 0.0

    def _speak(self, env):
        """注視が確認できた瞬間。respond_probで実際に言うかを決め、言えば不応期へ入る。

        5回に1回（shuffle_after）は、言い終えた直後に左右を入れ替える
        （場所の丸暗記を防ぐ仕掛け。前半の設計意図参照）。
        """
        if env.np_random.random() >= self.respond_prob:
            # 応えないと決めた回（既定respond_prob=1.0では起きない）＝黙って次の試行へ。
            self._state = self._PICK
            return None
        text = self.utterances.get(self._target)
        result = {"text": text, "target": self._target}
        self.n_utterances += 1
        if self.n_utterances % self.shuffle_after == 0:
            env._toy_swap_sign = -float(getattr(env, "_toy_swap_sign", 1.0))
            env._set_anchor()          # toy1/toy2の角度符号を反転させて再配置
            self.n_swaps += 1
            self._silent_until = self.shuffle_refractory_sec
        else:
            self._silent_until = self.refractory_sec
        self._state = self._REFRACTORY
        self._silent_t = 0.0
        return result


class WordSchedule:
    """語の再生装置（F1-4h） — 親の状態機械を経由せず、指定した時刻に指定した語を
    太郎の耳へ直接注入する（乳児語彙テストの音声再生に相当）。

    【設計】F/docs/設計_F1-4h_語彙テストの測定装置.md 技術付録「3. 語の再生装置」
    「4. 記録と集計」。ParentLabelingが発話を耳へ入れているのと**同じ経路だけ**を
    流用する（ToySupineEnv.step()が self._parent_utterance を通じて
    info["parent_utterance"]へ載せ、run/trainer.py._hear_parent_utteranceが
    耳→連合器へ渡す。トップのdocstring・parent_labeling.py参照）。
    状態機械・振り・視線確認は一切持たない＝ParentLabelingとは独立した最小限のクラス。

    Args（`world.word_test` から kwargs直渡し。実験ファイル側の書式は
        技術付録どおり `{"schedule": [...], "csv": "パス"}`）:
        schedule  [{"t": 秒, "word": 語}, ...]（時刻昇順を想定。エピソード開始
                  ＝env.data.time=0からの経過秒）。既定None＝空リスト＝
                  update()は毎stepNoneを返すだけ（CSV書き出しもcsv未指定なら無効）
                  ＝1ビットも既存の挙動を変えない。
        csv       毎stepの視線角度・語イベントを書き出す先のパス。既定None＝
                  書き出さない（技術付録4節「CSVの列」の実体。既存のCSVフック
                  （run/plugins/common/trace.py 等）はa1/a2/直近語を持たないため、
                  ここに自前で持たせた＝依頼書「無ければ自前のCSV書き出しを
                  持たせる」の判断）。
    """

    def __init__(self, schedule=None, csv=None):
        self.schedule = list(schedule or [])
        self.csv_path = csv
        self._csv_fp = None
        self._csv_writer = None
        self.reset()

    def reset(self):
        """エピソード境界で呼ぶ（ParentLabeling.resetと同じ位置づけ）。

        注意：CSVファイルはエピソードをまたいで開いたままにする（試行=語イベントは
        1エピソード内で複数回・複数試行の実験JSONを想定しており、エピソードごとに
        ファイルを作り直すと集計側が読みにくくなるため）。
        """
        self._next_idx = 0

    def _ensure_csv(self):
        if self.csv_path is None or self._csv_writer is not None:
            return
        d = os.path.dirname(self.csv_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._csv_fp = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fp)
        self._csv_writer.writerow(["t", "a1", "a2", "word"])

    def update(self, env):
        """1物理stepぶん進める。

        csv指定があれば毎stepの視線角度（a1=toy1への角度・a2=toy2への角度）と
        直近に注入された語（無ければ空）を1行書く（技術付録4節）。
        scheduleの時刻を跨いだstepだけ {"text": word, "target": None} を返す
        （他はNone）。target は ParentLabeling.update() の戻り値形式に合わせた
        だけで、WordScheduleでは使わない（toy1/toy2の対応づけは無いテスト場面）。

        env: ToySupineEnv自身（self）。env.data.time（エピソード開始からの秒数）・
             env._gaze_angle_to・env.taro（run/taro_setup.pyがビルド後に
             env.unwrapped.taroへ設定する。耳へ到達する唯一の経路）を使う。
        """
        word = None
        if self._next_idx < len(self.schedule):
            item = self.schedule[self._next_idx]
            if float(env.data.time) >= float(item.get("t", 0.0)):
                # 【なぜ、2026-08-21】taro.hearingが無効（既定False）のまま語を
                #   注入しても、run/trainer.py._hear_parent_utteranceの早期return
                #   （hearing/lexiconがNoneなら何もしない）で黙って捨てられる。
                #   語彙テストでこれが起きると「試行は回ったが一語も学習器に
                #   届いていない」という気づきにくい空振りになる（設計
                #   技術付録3節「注意：hearing有効が前提」の指定）。
                #   ビルド時点（e_scene.build()）ではtaro（run/taro_setup.py側）が
                #   まだ存在せずhearing設定を見られないため、taroが配線済みに
                #   なる最初の使用時（＝ここ）で止める。
                taro = getattr(env, "taro", None)
                assert getattr(taro, "hearing", None) is not None, (
                    "WordSchedule: taro.hearing が無効です（実験JSONの "
                    "taro.hearing を true にしてください）。語を注入しても "
                    "run/trainer.py._hear_parent_utterance の早期returnで "
                    "黙って捨てられ、テストとして成立しません。")
                word = item.get("word")
                self._next_idx += 1
        if self.csv_path is not None:
            self._ensure_csv()
            a1 = env._gaze_angle_to("test_object1")
            a2 = env._gaze_angle_to("test_object2")
            self._csv_writer.writerow([
                f"{float(env.data.time):.4f}",
                "" if a1 is None else f"{a1:.3f}",
                "" if a2 is None else f"{a2:.3f}",
                word or ""])
            self._csv_fp.flush()
        if word is not None:
            return {"text": word, "target": None}
        return None
