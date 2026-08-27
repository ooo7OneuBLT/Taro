"""学習ループ本体。太郎を動かし、学習させ、プラグインに測らせる。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
1175〜1304行（学習ループ）＋1069〜1098行（checkpoint）＋1128〜1148行（睡眠リプレイ）
＋1150〜1183行（体を育てる）を**そのまま写した**もの。

元は930行の1つの関数の中にあり、測定（自己モデル・接触・手が視野に入る割合）も
同じ場所に書かれていたため、「測り方を変える」と「学習を変える」が区別できなかった。
→ 測るのは**プラグイン**に渡す（`run/plugins/`）。ここは学習だけを持つ。

【役割の分担】
    run/config.py       設定（実験ファイルから）
    run/taro_setup.py   太郎の中身（脳・学習器・神経調節・小脳）
    run/trainer.py      ここ。太郎と環境を噛み合わせて回す
    run/plugins/        外から測る道具（太郎を変えない）

注意：同じ設定で同じ結果が出るかを必ず確かめる（`run/tools/check_divergence.py`）。
  乱数を消費する順序が1つ違うだけで同じシードでも別の学習になる（落とし穴 項3）。
  注意：測定は体を進めるので、測定の前後で状態を控えて戻している（項79・項80）。
    その仕組みは `_snapshot` / `_restore`。ここを崩すと再現性が失われる。
"""
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from run.taro_setup import Taro, rescale_action, mse, to_tensor  # noqa: E402
from run.context import Ctx                                     # noqa: E402

torch.set_num_threads(1)
DT = 0.01                    # MuJoCo の1物理ステップの秒数


def _cosine_sim(a, b):
    """コサイン類似度。F1-4b（語から注意への読み出し回路）の一致度計算に使う。

    a: np.ndarray、b: tuple/list（Lexicon.assoc()の戻り値）。どちらかがゼロベクトル
    なら0.0（run/plugins/common/word_learning.py の_cos()と同じ規約だが、Noneでなく
    0.0を返す＝呼び出し側set_recognition()がそのままmax(0,・)に渡せる形にする）。
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _mj_arrays(d):
    """MuJoCo の `data` から**書き換えられる配列を全部**控える。

    【なぜ機械的に取るか、2026-07-30】測定は体を進めるので、測定の前後で状態を
    戻す必要がある（落とし穴 項79・項80）。ところが `qpos` と `qvel` だけ戻しても
    足りず、実測では「環境を進めた結果（obs_out）」だけが食い違い続けた。
    加速度・ソルバの前回解・センサの値も次の計算に影響するため。
    注意：手で並べると必ず取りこぼす。`mujoco.mj_copyData` は Python に露出していない
      （mujoco 3.3.0 で確認）ので、属性を走査して配列を集める。
    注意：読み取り専用の配列（サイズ情報など）は書き戻せないので飛ばす。
    """
    out = {}
    for nm in dir(d):
        if nm.startswith("_"):
            continue
        try:
            v = getattr(d, nm)
        except Exception:
            continue          # 一部の属性は参照するだけで例外を出す（触らない）
        if isinstance(v, np.ndarray) and v.size and v.flags.writeable:
            out[nm] = v.copy()
    return out


def _release_renderers(env):
    """眼球の描画用メモリ（OpenGL）を明示的に返す。体を作り直す前に呼ぶ。

    【なぜ要るか、2026-07-30】体を育てる実験（500回ごとに体を作り直す）で、
    GPU のメモリが尽きて学習が落ちた（OpenGL error 0x505 → exit 139）。
    `e_toy_env` は眼球カメラの `mujoco.Renderer` を辞書に持ち、
    **モデルが変わったときにしか閉じない**。作り直しのたびに古いものが残る。
    注意：`env.close()` はこの辞書を触らないので、ここで自分で閉じる。
    """
    u = getattr(env, "unwrapped", env)
    cache = getattr(u, "_eye_renderers", None)
    if not cache:
        return 0
    n = 0
    for r in list(cache.values()):
        try:
            r.close()
            n += 1
        except Exception as e:      # noqa: BLE001
            # 注意：握りつぶすが黙らない（閉じ損ねはメモリが残るだけ）
            print(f"注意[renderer] 閉じるのに失敗: {type(e).__name__}: {e}", flush=True)
    cache.clear()
    u._eye_renderers = {}
    u._eye_renderers_model = None
    # 使い回しのキャッシュも捨てる（次の体で作り直させる）
    for nm in ("_vision_cache", "_vision_t"):
        if hasattr(u, nm):
            setattr(u, nm, None)
    return n


class Trainer:
    """太郎に「生きて学ぶ」をさせる。

    注意：このクラスは**測らない**。測るのはプラグイン。
      ここが測り始めると、また「学習ループの中に測定が埋まる」状態に戻る。
    """

    def __init__(self, cfg, *, plugins=(), verbose=True, log_row=None):
        self.cfg = cfg
        self.plugins = list(plugins)
        self.verbose = verbose
        self._log_row = log_row or (lambda row: None)
        self.env = None
        self.taro = None
        # 測定の前に体を控えるための入れ物（重いので1つ作って使い回す）
        self._snap_data = None
        self._snap_model = None
        self._last_parent_events = []      # 【2026-08-18新設・F1-3】直近step_kの発話イベント
        # 【2026-08-19新設・F1-4b】語から注意への読み出し。「思い浮かべている」語の
        #   記憶＝(vec, 思い浮かべ始めた時刻)。cfg.word_attentionがNoneのままなら
        #   一度も書き込まれず参照もされない＝既定挙動不変（設計後半「部品2」）。
        self._active_assoc = None
        # 【2026-08-24新設・作業C速度改善】cfg.profile=False（既定）のときは
        #   self._prof_enabled=False固定で、_prof_t0/_prof_addは何もしない
        #   （time.perf_counter()すら呼ばない）。学習の乱数消費・結果は1ビットも
        #   変わらない（時刻計測とpsutilはどちらもMuJoCo/torch/random/numpyの
        #   状態を読み書きしない）。実測：run/tools/check_profile_overhead.py。
        self._prof_enabled = bool(getattr(cfg, "profile", False))
        self._prof = {}          # {区分: 直近チェックポイント以降の累積秒数}
        self._prof_proc = None
        if self._prof_enabled:
            import psutil          # 遅延import（既定OFF時は依存を増やさない）
            self._prof_proc = psutil.Process(os.getpid())

    # ------------------------------------------------------------ 組み立て
    def build(self):
        """環境と太郎を作る。乱数の順序を元の実装に合わせる。"""
        cfg = self.cfg
        # 太郎の乱数は**4系統**ある。1つでも撒き忘れると同じシードで再現できない。
        #   【2026-07-30 に判明】4つ目（Python標準の random）を誰も撒いていなかった。
        #     `taro_core/src/body/internal_state.py` が「泣き始めるか」「寝入るか」
        #     「うとうとするか」を `random.random()` で決めている（同 168/121/146行）。
        #     泣く・眠るは**内受容感覚として脳の入力に入る**ので、1回でもタイミングが
        #     ずれると以後の学習が丸ごと分岐する。
        #   注意：元の経路（E/scripts/e_growth_train.py）にも同じ撒き忘れがある
        #     ＝過去の実験は同一シードでも再現していなかった（落とし穴 項79）。
        torch.manual_seed(cfg.seed)     # ①脳の初期値・探索のゆらぎ・睡眠リプレイの抜き取り
        np.random.seed(cfg.seed)        # ②numpy を使う処理（agency の並べ替えなど）
        random.seed(cfg.seed)           # ③内臓（泣く・寝る・うとうと）
        # ④環境の初期姿勢（gym の np_random）は env.reset(seed=) で撒く（taro_setup.py）

        from run.plugins.common import scene as scene_mod
        # 月齢は「学習ループ側の月齢」で上書きする（体を育てると学習中に変わるので、
        #   シーンの固定値では合わない）。＝シーンは月齢以外の環境を担う。
        self._cur_age = cfg.age_at(0)
        taro_spec = dict(cfg._taro)
        if self._cur_age is not None:
            taro_spec["age_months"] = float(self._cur_age)
        self._taro_spec = taro_spec
        self.env, self.scene, self.hands = scene_mod.build(
            cfg.scene, taro=taro_spec, seed=cfg.seed, verbose=self.verbose, hybrid=True)
        if self._cur_age is None:
            # シーンが決めた月齢を控える（体を育てない実験ではこのまま）
            self._cur_age = float(self.scene["body"]["age_months"])

        self.taro = Taro(cfg, self.env, seed=cfg.seed, verbose=self.verbose)
        self.state = self.taro.init_state(self.taro.first_obs)
        # 目標指向の探索が使う（過去に経験した感覚）
        # 注意：【逸脱/工学近似・2026-07-30 判明】これは Self-Prior ではない。
        #   先行研究（Kim, Kanazawa, Yoshida, Kuniyoshi 2025, arXiv:2504.11075）の
        #   self-prior は「経験した観測の**頻度分布**」を明示的に学習する
        #   （離散版＝カウント→Categorical／連続版＝正規化フロー NSF を最尤推定）。
        #   太郎は生ベクトルをFIFOに貯めて**一様ランダムに1つ選ぶだけ**で、
        #   頻度の重みが完全に消えている。⇒ doc/人間模倣からの逸脱リスト.md
        self.goal_buf = []
        # 手先位置の目標表現（案C・Goal Babbling段階1）。既存の goal_buf とは
        #   **別の**リストにする（encode_target/goal_buf は一切変更しない、という
        #   仕様の要求を厳密に満たすため。設計の統合判断「決定4」）。
        #   既定（goal_space!="reach_self"）ではこのリストは一度も使われない。
        self.reach_goal_buf = []
        self.reach_traj = None
        if cfg.goal_babbling and cfg.goal_space == "reach_self":
            from goal_babbling.trajectory import GoalTrajectory
            self.reach_traj = GoalTrajectory(L=cfg.goal_traj_len)
        self._t0 = time.time()
        self._build_probe_ctx()
        self._build_ctx()
        for p in self.plugins:
            p.setup(self.ctx)
        return self

    # -------------------------------------------------------- 環境の操作
    def step_k(self, a):
        """1回の判断で K 物理ステップ進める（＝同じ命令を K tick 保持する）。

        注意：【逸脱・2026-07-30】**感覚運動遅延（神経伝導の遅れ）が無い**。
          人間は運動指令が筋に届くまで、感覚が脳に届くまでに 0.1〜0.2秒かかる。
          太郎は命令がその tick で力になり、感覚もその tick で返る。
          ⇒ K=10（0.1秒）は「人間のフィードバック遅延に合わせた**間隔**」であって、
            遅れそのものは入っていない。
          MIMo grows!（López et al. 2025）は FIFO バッファとして実装している。
          → 人間模倣からの逸脱リスト「2026-07-30 感覚運動遅延が無い」

        【2026-08-11・新しい駆動モジュール】spinal_drive_mode="reflex_common"のとき、
          K tick中は**毎tick**（既存は a を固定してK回env.stepを呼ぶだけ）、
          伸張反射＋共通駆動を計算し直し、その tick のクリップ済み行動を
          env.step に渡す（案①4-3節「探索側の命令に加算」・設計8節「trainer.py」）。
          未指定（既定"cpg"）のときは、現状と1バイト差ない経路（この分岐を
          一度も通らない・従来通りK回同じaでenv.step）をそのまま使う。
        """
        o, term = self.state["obs"], False
        t = self.taro
        self._last_parent_events = []      # 【2026-08-18新設・F1-3】この判断(K tick)分の発話イベント
        if getattr(t, "reflex_common_active", False):
            am = self.env.unwrapped.actuation_model
            a_np = np.asarray(a, dtype=np.float64)
            for _ in range(self.cfg.K):
                r = t.brain.step_reflex_common(DT, am.muscle_lengths, am.muscle_velocities)
                a_tick = np.clip(a_np + r, 0.0, 1.0).astype(np.float32)
                o, rew, te, tr, info = self.env.step(a_tick)
                self._hear_parent_utterance(o, info)
                if te or tr:
                    term = True
                    break
            # 【2026-08-13・触覚の順応】K回のenv.stepループを終えた最後のoにだけ
            #   適用する（ループ途中の各tickには適用しない。仕様4節）。
            o = t.apply_touch_adaptation(o, is_reset=False)
            return o, term
        for _ in range(self.cfg.K):
            o, r, te, tr, info = self.env.step(a)
            self._hear_parent_utterance(o, info)
            if te or tr:
                term = True
                break
        o = t.apply_touch_adaptation(o, is_reset=False)
        return o, term

    def _vision_backend_encode(self, o):
        """`t.vision_backend.encode()` に渡す画像を選ぶ（F1-7新設）。

        中心窩カメラ（`eye_left_fovea`/`eye_right_fovea`）が観測に含まれていれば
        そちらを、無ければ従来どおり `eye_left`/`eye_right` を渡す。
        `body.fovea_camera` が既定False（未指定シーン含む）のときは、観測に
        該当キーが無いため常に従来経路＝**1ビットも挙動が変わらない**。

        中心窩画像は撮影時点で既に視野15度に切り出し済み（MuJoCo側のカメラfovy）
        なので、下流バックエンド（例：dinov2_vits14）が持つ `fovea_crop` を
        二重にかけると情報を失う。バックエンドが `fovea_px` 属性を持つ場合のみ、
        encode()呼び出しの間だけ「crop が実質no-opになる値」へ一時的に差し替え、
        呼び出し後に元へ戻す（vision_backends.py は触ってよいファイルの外なので、
        ここでの一時上書きで対応する）。customバックエンドはfovea_crop自体を
        呼ばないので、この上書きは何もしない。
        """
        t = self.taro
        has_fovea = "eye_left_fovea" in o and "eye_right_fovea" in o
        if has_fovea:
            img_l, img_r = o["eye_left_fovea"], o["eye_right_fovea"]
        else:
            img_l, img_r = o["eye_left"], o["eye_right"]
        backend = t.vision_backend
        old_fovea_px = None
        has_fovea_px_attr = has_fovea and hasattr(backend, "fovea_px")
        if has_fovea_px_attr:
            old_fovea_px = backend.fovea_px
            backend.fovea_px = 10 ** 9   # fovea_crop側のmin(fovea_px,h)でno-op化
        # 【作業C・2026-08-24】この時間は呼び出し元（_hear_parent_utterance＝
        #   step_k内＝"env_step"に含まれる／_apply_word_attention・
        #   _apply_word_production＝"produce"に含まれる）の**内訳**として重ねて足す
        #   （他区分との合計が二重計上になる。CSV列名 t_vision_backend_sec は
        #   「他区分の一部」であって独立区分ではないと作業記録に明記）。
        _prof_t = self._prof_t0()
        try:
            return backend.encode(img_l, img_r)
        finally:
            if has_fovea_px_attr:
                backend.fovea_px = old_fovea_px
            self._prof_add("vision_backend", _prof_t)

    def _hear_parent_utterance(self, o, info):
        """親の発話（info["parent_utterance"]）を耳→連合器へ渡す（2026-08-18新設・F1-3）。

        taro.hearing/taro.lexicon は既定None（cfg.hearing=False）＝この関数は
        呼ばれても何もしない＝既存実験の挙動は1ビットも変わらない
        （E/scripts/parent_labeling.py が無効なら info にキー自体が無いので、
        いずれにせよ何も起きない）。

        仕様書（F/docs/仕様_F1-3_....md 技術付録「部品2」）の指定どおり：
          tokens = taro.hearing.hear(text)
          state  = taro.fusion.vision(左目画像, 右目画像)  ※視覚64次元のみ
          taro.lexicon.observe(tokens, state=state)
        confidences は F1-2 仕様書の単純化「発話全体を1チャンクとして扱ってよい」に従い、
        全要素同値（谷が生まれない＝発話全体が1つの単位として切り出される）を渡す。
        """
        t = self.taro
        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return
        pu = info.get("parent_utterance") if isinstance(info, dict) else None
        if not pu:
            return
        text = pu.get("text")
        tokens = t.hearing.hear(text)
        confidences = [1.0] * len(tokens)
        # detach＝この統計（共起の平均）は学習の逆伝播に使わない値であるため
        #   （Lexiconは純Pythonの累積平均。計算グラフを持ち越さない）。
        # 【2026-08-19・F1-3b】視覚表現の作り方はt.vision_backend経由（差し替え可能）。
        #   既定null時はcustomバックエンドがtaro.fusion.visionをそのまま呼ぶだけなので、
        #   出力は従来コード（t.fusion.vision(...).detach().cpu().tolist()）と同一
        #   （custom backendのencode()内でも同じdetach().cpu()を行う）。
        state = self._vision_backend_encode(o).tolist()
        # 【F2-8・2026-08-25】「見慣れた景色」の平均を育てる（reverse_lookupの
        #   両側引き算に使う）。語彙にも想像にも触らない純粋な足し算で、
        #   すでに計算済みのstateを渡すだけ＝追加の計算コストはゼロ。
        t.lexicon.observe_view(state)
        chunk = t.lexicon.observe(tokens, confidences, state=state)
        # 【なぜstateもここに置くか】word_learningプラグイン（読むだけ）が「正解物の
        #   特徴EMA」を作るのに使う（プラグインがtaro.fusionを呼び直すと二重計算に
        #   なるうえ、この瞬間のobs（o）はstep_kのK tick内の値でctx.last["obs_out"]
        #   より新しい＝取り直すと精度が落ちる）。
        self._last_parent_events.append(
            {"text": text, "target": pu.get("target"), "state": state})
        # 【2026-08-19新設・F1-4b】語から注意への読み出し回路（設計：
        #   F/docs/設計_F1-4b_語から注意への読み出し回路.md 後半「部品2」）。
        #   cfg.word_attentionがNoneのままなら、この行自体を実行しない
        #   （lexicon.assoc()の呼び出しコストすら払わない＝既定挙動・コスト不変）。
        if self.cfg.word_attention is not None and chunk is not None:
            vec = t.lexicon.assoc(chunk)
            if vec is not None:
                self._active_assoc = (vec, self.ctx.sim_sec)

    def _apply_word_attention(self, o):
        """語から注意への読み出し回路（設計：F/docs/設計_F1-4b_....md 後半「部品2」）。

        「思い浮かべている」語のメモ（_active_assoc）と、いま実際に見ているものの
        DINOv2表現の一致度を、視線誘導反射（OrientingReflexV2）へ渡す。渡された
        反射側は、一致度が高い間は保持（hold）の解除を遅らせる
        （E/scripts/e_orienting_v2.py の set_recognition() / REC_THRESHOLD）。

        既定（cfg.word_attention is None）では最初のifで即returnし、DINOv2の
        encode等の追加計算は一切走らない＝既存実験の挙動・コストは1ビットも
        変わらない（仕様の要求どおり）。
        """
        wa = self.cfg.word_attention
        if wa is None or not wa.get("enabled", True):
            return
        t = self.taro
        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return          # 耳無効なら思い浮かべる元(_active_assoc)自体が育たない
        orienting = getattr(self.env.unwrapped, "_orienting", None)
        if orienting is None:
            return          # 視線誘導反射(orienting_reflex)が無効なら渡す先が無い
        active_sec = float(wa.get("active_sec", 3.0))
        active = self._active_assoc
        now = self.ctx.sim_sec
        if active is not None and (now - active[1]) < active_sec:
            vec = self._vision_backend_encode(o)
            sim = _cosine_sim(vec, active[0])
            orienting.set_recognition(max(0.0, sim))
        else:
            # 期限切れ（ACTIVE_SEC超過）。信号を明示的に0へ戻す
            #   （戻さないと反射側に前回のsimが居座り続けて保持が解除されなくなる）。
            orienting.set_recognition(0.0)

    def _apply_word_production(self, o):
        """見た物の名前を言う（F2「初語」、設計：F/docs/設計_F2_初語（見た物の名前を
        言う）.md 第2部）。

        既定（cfg.produce is None）では最初のifで即returnし、逆引き
        （lexicon.reverse_lookup）・generate()・報酬計算等の追加計算は一切走らない
        （既存実験の挙動もコストも1ビットも変わらない）。

        トリガー＝「じっと見ていて（視線誘導反射が保持中＝サッケード中でなく
        _should_hold()が真）、かつそれが何か分かっている（逆引きの確信度が
        閾値以上）」（設計「いつ言うか」節）。既存の信号を読むだけで、
        新しい仕組みは作らない（E/scripts/e_orienting_v2.py:1058 _should_hold()・
        :1132 holdingと同じ定義）。直前に喋っていなければ（クールダウン）発声する。
        """
        pd = self.cfg.produce
        self.ctx.last_produce = None
        self.ctx.last_babble = None
        if pd is None or not pd.get("enabled", True):
            return
        t = self.taro
        if getattr(t, "produce_learner", None) is None:
            return          # 産出の配線が無ければ何もしない

        # 【F2-1・実装作業④・2026-08-23】喃語モード（mode="babble"）。
        #   設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「④」。
        #   mode未指定（既定"word"）ならこのifを一歩も通らず、以降の従来どおりの
        #   word_mode処理（視覚トリガー・逆引き・報酬・学習）に進む＝1ビットも
        #   変わらない。babbleモードは視覚トリガーを使わず、クールダウンだけで
        #   自発的に発声し、逆引き・報酬・学習は一切通らない（声を出して
        #   小脳の帳面に書くだけ）。
        if pd.get("mode", "word") == "babble":
            self._apply_babble(pd)
            return

        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return          # 耳/連合器の配線が無ければ何もしない
        orienting = getattr(self.env.unwrapped, "_orienting", None)
        if orienting is None:
            return          # 視線誘導反射が無効なら「注視している」を判定できない

        # ① 逆引き：いま見ている視覚ベクトル → 一番似ているchunk（言いたい語）。
        vec = self._vision_backend_encode(o)
        # 【F2-8・2026-08-25】逆引きの前に「見慣れた景色」の平均へ今の見えを足す。
        #   ここは毎ステップ通るので、産出中は太郎が見たものすべてが平均に入る。
        t.lexicon.observe_view(vec)
        chunk, sim = t.lexicon.reverse_lookup(vec)
        if chunk is None:
            return
        # 【Tier3・2026-08-22】閾値はorienting_reflexの認識信号REC_THRESHOLD
        #   （e_orienting_v2.py:461＝0.80、F1-3cの保存済み画像で較正）に揃えた。
        #   「見た物と語の一致度」を扱う信号を2種類持たせない判断（設計「残る
        #   未決」節）。
        threshold = float(pd.get("threshold", 0.80))
        if sim < threshold:
            return
        # ② 注視が続いている（サッケード実行中でなく、いままさに同じ対象へ
        #   留まっている）。e_orienting_v2.py _update_habituation() が保持判定に
        #   使っているのと同じ2条件をそのまま読む（新しい仕組みは作らない）。
        holding = orienting._sacc_remaining <= 0.0 and orienting._should_hold()
        if not holding:
            return
        # ③ クールダウン（直前に喋っていない）。
        #   【Tier3・2026-08-22・文献根拠なし】同じ語を連呼し続けないための
        #   最小限の歯止め。値は仮（人間の平均発話間隔より長めに取っただけ）。
        cooldown_sec = float(pd.get("cooldown_sec", 2.0))
        now = self.ctx.sim_sec
        if t._last_produce_sec is not None and (now - t._last_produce_sec) < cooldown_sec:
            return

        # ---- ここからトリガー成立：② generate() で実際に発声する -----------------
        t._last_produce_sec = now
        target_word = t.hearing.vocab.decode(list(chunk))
        target_tokens = t.produce_vocab.encode(target_word)
        max_length = int(pd.get("max_length", 8))

        # 【F2-2・実装作業③・2026-08-23】ブローカ野で発話計画を立てる。
        #   帳面（produce_cerebellum）に無いモーラは motor=None のまま計画に入り、
        #   generate() 側で大脳皮質が自力で選ぶ（探索的な動き）。use_hear_fallback
        #   は既定Falseのまま（設計第2部決定3：正解表 vocal_tract.hear() は使わない）。
        #   設計：F/docs/設計_F2-2_見た物の名前を言う.md 第4部「③」。
        t.produce_broca.plan(
            target_word, t.produce_cerebellum, t.produce_vocal_tract,
            use_hear_fallback=False)
        plan_length = t.produce_broca.get_plan_length()
        known_moras = sum(
            1 for item in t.produce_broca.motor_buffer if item["motor"] is not None)

        # 【F2-2・実装作業③】speech_plan を渡すと generate() 内部で
        #   max_chars = min(計画長, stamina)（stamina未指定なのでmax_lengthが
        #   代わりに使われる。実装作業⑦の実測：taro_core/F にはまだ肺(stamina)の
        #   機構が移植されておらず taro に stamina属性が無いため、ここではstamina
        #   引数を渡さない＝計画が切られることはない。詳細は作業記録参照）。
        # 【F2-2・2026-08-23】発話時の探索ノイズ（NE）をどうするか。
        #   NEノイズは青斑核（locus_coeruleus.py）の仕組みで、本来は
        #   **学習のための探索**（報酬が得られていないほど色々試す）。
        #   目標Bは発音を報酬で学習していたので探索が要ったが、
        #   **F2-2は学習しない（帳面から引いて実行するだけ）ので探索は不要**。
        #
        #   さらに、ここで使われる `t.ne.get_ne_level()` は
        #   **手足の運動学習の報酬**で上下する値（trainer.py の運動探索と共用）。
        #   発話とは無関係な信号で口が震えることになる。
        #
        #   実測（2026-08-23）：ノイズを消して計画をそのまま実行すれば
        #   「わんわん」は完璧に出るのに、実走行では27回中0回しか一致せず
        #   「わおざあ」「らあぱん」等になっていた。
        #   （ノイズの"隣にずらす"実装自体にも問題がある。
        #     taro_core/src/brain/taro_brain.py の `_apply_ne_noise` の⚠を参照）
        #
        #   `produce.exploration_noise`（既定False＝ノイズ無し）で切り替える。
        #   学習を有効にする（produce.learn=true）ときは探索が要るので、
        #   そのときだけ True にすること。
        _ne = t.ne.get_ne_level() if bool(pd.get("exploration_noise", False)) else 0.0
        generated, log_probs, hidden = t.brain.generate(
            hidden=None, max_length=max_length, eos_idx=2,
            vocal_tract=t.produce_vocal_tract, ne_level=_ne,
            cerebellum=t.produce_cerebellum, speech_plan=t.produce_broca)
        generated_word = t.produce_vocab.decode(generated)

        # 【F2-2・実装作業⑤・2026-08-23】自分の声を聞く（hidden state更新）。
        #   _apply_babble に入れたものと同じ（B原本 B/src/environment/core_b.py:
        #   610-615 self_babble() の移植）。学習はしない。次に発声するときの
        #   初期hiddenには使わない（wordモードは従来どおり毎回 hidden=None、
        #   taro_setup.py _setup_produce のコメント参照）。ここでは t._produce_hidden
        #   を更新するだけに留める（babbleモードとの一貫性のため）。
        if generated:
            full_tokens = [1] + generated + [2]
            listen_input = torch.tensor([full_tokens], device=t.brain._device())
            with torch.no_grad():
                _, h = t.brain.forward_hidden(listen_input)
            t._produce_hidden = h

        # 【F2-2・実装作業④・2026-08-23】報酬・学習は既定OFF（設計第2部決定1：
        #   「学習ではなく実行」。帳面が正解を持っているので学ぶものが無い）。
        #   produce.learn（既定False）で有効化できる形で**コードは消さずに残す**
        #   （将来、聴覚経路(逸脱その29)や automatization(逸脱その33)に手をつける
        #   ときにそのまま使える）。
        reward = None
        if bool(pd.get("learn", False)):
            # ③ 出た音 vs 言いたかった語 の一致度を報酬にする（B原本の移植、
            #   taro_core/src/brain/imitation_reward.py:出典コメント参照）。
            from imitation_reward import compute_imitation_reward, compute_alignment_credit
            reward = compute_imitation_reward(
                target_tokens, generated, vocab=t.produce_vocab,
                vocal_tract=t.produce_vocal_tract)

            # 【12ヶ月産出実験の準備①・2026-08-23】発話まるごとに単一のδだけで
            #   学習すると「4文字のどれが良くてどれが悪かったか」が太郎に分からない
            #   （探索範囲 588^4 通り）。文字ごとの一致度（compute_alignment_credit、
            #   B2-2・imitation_reward.py:出典コメント参照）を使い、良かった文字は
            #   より強く強化・目標語にない余分な文字は抑制する（探索範囲 588×4通り）。
            credits = compute_alignment_credit(
                target_tokens, generated, vocab=t.produce_vocab,
                vocal_tract=t.produce_vocal_tract)

            # ④ 学習：**運動学習(taro.learner)とは別の学習器**（taro.produce_learner）で
            #   REINFORCE（設計「⚠学習器の共有」案a。taro_setup.py _setup_produce
            #   のコメント参照＝勾配の対象が運動系と重ならない）。
            rpe = t.produce_dop.compute_rpe(reward)
            pl = t.produce_learner.learn_action(log_probs, rpe, credits=credits)
            t.produce_learner.update(0.0, pl)

        self.ctx.last_produce = {
            "target_word": target_word, "sim": float(sim),
            "generated_word": generated_word,
            "reward": float(reward) if reward is not None else None,
            "plan_length": int(plan_length), "known_moras": int(known_moras)}

    def _apply_babble(self, pd):
        """喃語モード（F2-1）：視覚トリガーなし・クールダウンだけで自発的に声を出す。

        設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「④」。
        逆引き・報酬・学習は通らない。声を出して小脳の帳面（forward_map/
        inverse_map）に書くだけ（決定：喃語フェーズは帳面を溜めるだけ）。
        """
        t = self.taro
        cooldown_sec = float(pd.get("cooldown_sec", 2.0))
        now = self.ctx.sim_sec
        if t._last_produce_sec is not None and (now - t._last_produce_sec) < cooldown_sec:
            return
        t._last_produce_sec = now

        # 【F2-1・実装作業⑤】顎のサイクル数を1〜2から一様ランダムに引く
        # （設計第2部決定1）。実験ファイルの produce.jaw_cycles で候補を
        # 差し替え可能（既定[1, 2]）。
        jaw_choices = pd.get("jaw_cycles", [1, 2])
        jaw_cycles = random.choice(jaw_choices)

        generated, log_probs, hidden = t.brain.generate(
            hidden=t._produce_hidden, max_length=int(pd.get("max_length", 8)),
            eos_idx=2, vocal_tract=t.produce_vocal_tract, ne_level=t.ne.get_ne_level(),
            cerebellum=t.produce_cerebellum, jaw_cycles=int(jaw_cycles))
        generated_word = t.produce_vocab.decode(generated)

        if not generated:
            return

        # 【F2-1・実装作業⑦】自分の声を聞く（hidden state更新）。学習はしない。
        #   B原本 B/src/environment/core_b.py:610-615 self_babble() の移植。
        #   BOS(1)/EOS(2)は Vocabulary の固定インデックス（taro_core/src/senses/
        #   hearing.py）。次の喃語発声はこのhiddenを引き継ぐ（word モードは
        #   従来どおり毎回 hidden=None のまま・不変）。
        full_tokens = [1] + generated + [2]
        listen_input = torch.tensor([full_tokens], device=t.brain._device())
        with torch.no_grad():
            _, h = t.brain.forward_hidden(listen_input)
        t._produce_hidden = h

        self.ctx.last_babble = {
            "generated_word": generated_word, "jaw_cycles": int(jaw_cycles),
            "length": len(generated)}

    # ---------------------------------------------------- 作業C：処理時間の計測
    def _prof_t0(self):
        """区間の開始時刻。既定OFF（cfg.profile=False）なら常に0.0を返すだけ
        （time.perf_counter()を一切呼ばない＝コストも副作用も無い）。"""
        return time.perf_counter() if self._prof_enabled else 0.0

    def _prof_add(self, key, t0):
        """区間の終了。既定OFFなら即return（呼び出し元でif分岐しなくてよいようにする）。"""
        if not self._prof_enabled:
            return
        self._prof[key] = self._prof.get(key, 0.0) + (time.perf_counter() - t0)

    def reset_state(self):
        self.state["obs"], _ = self.env.reset()
        # 【2026-08-13・触覚の順応】新しい物理観測が生まれる瞬間（env.reset()直後）
        #   にだけ適用する（仕様4節。fusion.py・encode_targetの中では絶対に呼ばない）。
        self.state["obs"] = self.taro.apply_touch_adaptation(self.state["obs"], is_reset=True)
        self.state["hidden"] = self.taro.brain.init_motor_hidden()
        self.state["prev_a"] = torch.zeros(self.taro.n_act)
        # 【2026-08-12追記】立ち上がり検出（前tickの状態）を持つreward_contributor
        #   （例：_MouthTouchBonusContributor）を、エピソード境界でリセットする。
        #   hasattr で判定する緩い規約＝reset() を持たない既存の
        #   _DoubleTouchBonusContributor はスキップされる（既存挙動は不変）。
        for c in self.taro.reward_contributors:
            if hasattr(c, "reset"):
                c.reset()

    # -------------------------------------------------- 測定器へ渡す入れ物
    def _build_probe_ctx(self):
        """`E/scripts/e_probes.py` に渡す入れ物。判定の実体は e_probes 側にある。"""
        import e_probes
        t = self.taro
        _probe_log_dir = os.path.join(_ROOT, os.path.dirname(self.cfg.log or "")
                                      or os.path.join("E", "logs", "run"))
        # 【なぜ、2026-08-06】inverse_probe等（追加依頼1）はこのディレクトリへ
        #   直接ファイルを書く（evaluate/agency_probeは書かないので、これまで
        #   runシステム経由では未作成でも表面化しなかった）。無いと
        #   FileNotFoundErrorで落ちるので、ここで作っておく。
        os.makedirs(_probe_log_dir, exist_ok=True)
        self.probe_ctx = e_probes.ProbeContext(
            brain=t.brain, fusion=t.fusion, target_fusion=t.target_fusion,
            nat_head=t.nat_head, env=self.env, rescale_action=rescale_action,
            zc=t.infer_latent, act_mean=t.act_mean, step_k=self.step_k,
            ln_prop=t.encode_target, reset_state=self.reset_state,
            infer_goal_action=t.infer_goal_action,
            state=self.state, n_act=t.n_act, n_eval=self.cfg.n_eval,
            K=self.cfg.K, DT=DT, seed=self.cfg.seed,
            # 注意：リポジトリのルート基準にする。cwd 相対だと、どこから実行したかで
            #   逆モデルの診断ログ（inv_probe_*.txt 等）の出力先が変わる。
            log_dir=_probe_log_dir,
            # 【なぜ、2026-08-06】学習側(cfg.efference_copy)と測定側(effcopy)が
            #   食い違うと「学習側は無効化したのに測定側だけこっそり有効」という
            #   事故になる（監査指摘）。必ず同じ設定値を渡す。
            effcopy=self.cfg.efference_copy)

    def _build_ctx(self):
        """プラグインに渡す入れ物。プラグインは読むだけ。"""
        u = self.env.unwrapped
        dt = float(u.model.opt.timestep) * int(u.frame_skip) * self.cfg.K
        # 注意：name を入れ忘れると、ダッシュボードの見出しがフォルダ名になる
        #   （2026-07-31 に実際にそうなっていた）。プラグインは spec しか見られない。
        self.ctx = Ctx(env=self.env, spec={"name": self.cfg.name,
                                           "scene": self.cfg.scene, "taro": self.cfg._taro,
                                           "run": self.cfg._run},
                       scene=self.scene, n_steps=self.cfg.steps, dt=dt,
                       brain=self.taro.brain, log=self._log_row)
        # 自己モデルのプラグインが測るのに使う（e_probes への入れ物）
        self.ctx.probe_ctx = self.probe_ctx
        self.ctx.taro = self.taro
        # closed_loop_probe が目標姿勢のサンプリングに使う経験バッファ（参照を渡すだけ、
        #   中身は学習ループが書き足す＝2026-08-06、追加依頼1）
        self.ctx.goal_buf = self.goal_buf

    # ---------------------------------------------------------- 体を育てる
    def _regrow(self, new_age):
        """保存を挟まずに env だけ作り直す（脳・経験・オプティマイザは残る）。

        【なぜ、2026-07-29】以前は「0ヶ月で学ぶ → 保存 → 別プロセスで4ヶ月として
        読み込み」で体を切り替えていた。ところが保存されるのは**脳の重みだけ**で、
        オプティマイザの状態・経験バッファ・馴化は消える（落とし穴 項75）。
        段階的成長を9回の保存で作ると、その条件だけ内部状態が8回消える＝交絡。
        """
        cfg = self.cfg
        old = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        old_touch = int(self.env.observation_space["touch"].shape[0]) if cfg.touch else 0
        # 視覚のレンダラ（OpenGLの描画用メモリ）を**明示的に閉じる**。
        #   【なぜ、2026-07-30】体を育てる実験で学習が落ちた：
        #     WARNING: OpenGL error 0x505 in or before mjr_makeContext
        #     ⇒ 0x505 は GPU のメモリ不足。体を作り直すたびに眼球用の
        #       `mujoco.Renderer` が新しく作られ、**古いものが解放されずに積み上がる**。
        #     条件Cは500回ごとに作り直すので36回ぶん溜まり、実測で
        #     3000回目（6回目の作り直し）で落ちた（exit=139＝メモリアクセス違反）。
        #   注意：`env.close()` では解放されない（レンダラは環境が持つ Python の辞書）。
        _release_renderers(self.env)
        self.env.close()
        from run.plugins.common import scene as scene_mod
        spec = dict(self._taro_spec)
        spec["age_months"] = float(new_age)
        self.env, self.scene, self.hands = scene_mod.build(
            cfg.scene, taro=spec, seed=cfg.seed, verbose=False, hybrid=True)
        new = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        if old != new:
            # 注意：ここで止めるとき、新しく作った env を閉じてから投げる
            #   （閉じないと描画コンテキストが残る）
            close_env(self.env)
            raise AssertionError(
                f"体を作り直したら観測/行動の次元が変わった {old} -> {new}。"
                "（落とし穴 項75）")
        # ---- 触覚の地図を新しい体のものに差し替える -------------------------
        # 【なぜ、2026-07-31】触覚は `observation` とは別のキーなので、上の
        #   次元チェックを**素通りする**。実測では observation は801のまま、
        #   touch だけ 4,824 → 9,804 に変わっていた。
        #   SomatosensoryCortex は「点→部位」の対応表を持っているので、
        #   入れ替えないと**静かに別の部位を読む**（落とし穴 項86）。
        if cfg.touch:
            new_touch = int(self.env.observation_space["touch"].shape[0])
            if cfg.somatosensory:
                self.taro.on_body_change(self.env)
                if old_touch != new_touch:
                    print(f"[body-growth] 触覚の地図を差し替え {old_touch} → {new_touch}次元"
                          "（学習した部位の重みは保持）", flush=True)
            elif old_touch != new_touch:
                close_env(self.env)
                raise AssertionError(
                    f"体を作り直したら触覚の次元が変わった {old_touch} -> {new_touch}。\n"
                    "  somatosensory=false の触覚エンコーダは入力次元が固定なので使えない。\n"
                    "  somatosensory=true にすると部位ごとの要約（有無/強さ/重心）になり、"
                    "点数が変わっても層の形が変わらない。")
        elif self.taro.double_touch is not None:
            # 【2026-08-05追記：全身一般化】cfg.touch=False（taro自身の脳が触覚を
            #   使っていない）シーンでは、上の `if cfg.touch:` ブロックに一度も
            #   入らないため、taro.on_body_change 自体がこれまで一度も呼ばれて
            #   いなかった。double_touch（taro自身の fusion.touch/target_fusion.touch
            #   とは別の自前の SomatosensoryCortex インスタンス）は cfg.touch に
            #   関わらず存在しうるので、cfg.touch=False でも呼ぶ必要がある。
            #   （taro_setup.py の on_body_change 内部で fusion.touch が None の
            #   ときは即座に return するだけなので、これを追加しても cfg.touch=False
            #   の既存の挙動には一切影響しない。設計1-7節・実装ノウハウ
            #   2026-08-05項が指摘した早期returnの問題と同根＝「成長のたびに
            #   呼ばれるべき処理が、既存のtouch専用ガードの内側に置かれていて
            #   touch=falseでは一度も実行されない」という同型のバグを、
            #   trainer.py側の呼び出し元でも見つけて直した。実装時に検証スクリプト
            #   （run/tools/check_double_touch_generalized.py [6]）で実際に
            #   再現・確認済み：この修正が無いと、成長後にdetect()が
            #   AssertionError（触覚の次元が地図と合わない）で必ず落ちる）。
            self.taro.on_body_change(self.env)
        self.probe_ctx.env = self.env       # 測定器も新しい体を見る
        self.ctx.env = self.env
        u = self.env.unwrapped
        self.ctx.model, self.ctx.data = u.model, u.data
        # プラグインに知らせる（geom の id を引き直させる）。
        #   注意：元の実装はここが無く、さらに `env.unwrapped` を作り直し前のまま
        #     参照していた＝体を育てる実験では接触を古い体で見ていた。
        for p in self.plugins:
            p.on_body_change(self.ctx)
        # 【なぜ、2026-08-03・Tier3】体が変わると固有感覚などの分布が変わり、次に来る
        #   生の予測誤差が急変しうる。自己接触と同じ機序（速い平均だけが跳ねてprogressが
        #   マイナスに振れる）が成長イベント直後にも起きる恐れがある（調査報告
        #   2026-08-03 1-3節）。pe_fast/pe_slowを同じ値に揃えてprogressを0から
        #   再スタートさせる（ゼロリセットはしない。理由はresync()のdocstring参照）。
        self.taro.lp.resync()
        self.reset_state()                  # 新しい体で立て直す。脳と経験は保持されたまま
        # 手先位置の目標表現（案C）：ホーム姿勢を作り直した体で再計算する
        #   （設計の決定7。可動域自体が成長で変わりうるため、初期姿勢を1回だけ
        #   固定すると成長後に古い基準のまま目標を作ることになる）。
        if cfg.goal_babbling and cfg.goal_space == "reach_self":
            self.taro.home_reach_goal = self.taro.encode_reach_goal(self.state["obs"]).detach()

    # ------------------------------------------------------------ 睡眠
    def consolidate(self, n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチで再生し、予測経路を復習で固める。"""
        t = self.taro
        eps = t.hippo.replay()
        N = len(eps)
        if N < bs:
            return
        SV = torch.stack([e[0] for e in eps]); PA = torch.stack([e[1] for e in eps])
        AA = torch.stack([e[2] for e in eps]); CF = torch.stack([e[3] for e in eps])
        CLP = torch.stack([e[4] for e in eps]); NLP = torch.stack([e[5] for e in eps])
        H = torch.cat([e[6] for e in eps], dim=1)          # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = t.emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)
            out, _ = t.brain.motor_gru(emb, hb)
            z, kl, rc = t.brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + t.nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = t.block_pe(pred, NLP[idx]) + kl + rc       # 学習ループと同じ基準
            t.learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(t.learner.brain.parameters(),
                                           t.learner.grad_clip)
            t.learner.optimizer.step()

    # ------------------------------------------- 測定を「無かったこと」にする
    #
    # 【なぜ要るか、2026-07-30】自己モデルの測定（`e_probes.evaluate`）は
    #   **見るだけではなく実際に動かす**（n_eval=80 判断＝8秒ぶん）。
    #   そのため測定を1回するたびに：
    #     ・太郎の体が 8秒（agency も測るなら 14秒）ぶん進む
    #     ・学習が「測定後の体」から再開する
    #     ・チェックポイント間隔を変えると学習そのものが変わる（項80）
    #     ・同じシードでも結果がばらつく入り口になる（項79）
    #   実測：測定を外すと4回すべて完全一致。入れると step 1 で既に食い違う。
    #        食い違うのは体の状態（関節・前庭・触覚・視覚）で、**脳の重みは一致**。
    #
    # ⇒ 測定の前にすべてを控え、測定が終わったら**元に戻す**。
    #    学習から見て測定は「無かったこと」になる。
    #
    # 注意：これで過去の実験とは学習の軌道が変わる（＝数値が比較できなくなる）。
    #   それでも直すのは、過去の数値がそもそも再現しないため（項79）。
    # 注意：測定**値**のばらつきは残りうる（測定の中の非決定性は消えない）。
    #   ただしそれは「測るときの誤差」で、学習が分岐して増幅するのとは別物。

    def _snapshot(self):
        """体・内臓・乱数の状態をまるごと控える。"""
        import copy
        import random
        u = self.env.unwrapped
        d = u.data
        # --- 体（MuJoCo）を丸ごと控える ----------------------------------
        #   qpos/qvel だけでは足りない。加速度（qacc）・ソルバの前回解
        #   （qacc_warmstart）・センサの値も次の計算に影響する。
        #   注意：手で並べると必ず取りこぼす（実測で obs_out だけ食い違い続けた）。
        #   注意：`mujoco.mj_copyData` は Python に露出していない（mujoco 3.3.0 で確認）。
        #     なので**書き換えられる配列を機械的に全部**控える。
        snap = {
            "mjarr": _mj_arrays(d),
            "time": float(d.time),
            # --- 学習ループが持つ「いま」-----------------------------------
            "state": {"obs": self.state["obs"],
                      "hidden": self.state["hidden"].clone(),
                      "prev_a": self.state["prev_a"].clone()},
            # --- 乱数（3系統＋Python標準）測定が引いた分を巻き戻す ----------
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        }
        # --- 環境の乱数（gym が別に管理している。reset の姿勢の揺らぎに使う）---
        npr = getattr(u, "np_random", None)
        if npr is not None:
            snap["env_rng"] = npr.bit_generator.state
        # --- 内臓（HybridEnv）。空腹・眠気・不快・覚醒・泣いている残り時間 ----
        organs = {}
        for nm in ("internal_state", "stomach", "blood_vessel", "adenosine",
                   "homeostasis", "_step_in_second", "_feeding"):
            if hasattr(self.env, nm):
                organs[nm] = copy.deepcopy(getattr(self.env, nm))
        snap["organs"] = organs
        # --- 筋の状態（MuJoCo の外・Python 側にある）------------------------
        #   MuscleModel は活性化ダイナミクス（力がじわっと立ち上がる一次遅れ）を
        #   `self.activity` という**Python の配列**で持っている。d.act ではない。
        #   注意：ここを戻さないと「測定中に力んだ状態」から学習が再開する。
        #     2026-07-30 の実測では、これが obs_out（環境を進めた結果）が
        #     食い違う原因だった。元の学習ループにも同じ穴がある。
        snap["actu"] = self._numeric_attrs(getattr(u, "actuation_model", None))
        # --- 環境が持つ数値の状態（おもちゃのタイマー・視覚のキャッシュなど）---
        #   列挙漏れを防ぐため**数値と配列は機械的に全部**控える。
        #   （`_t_since_reset` `_toy_pending` `_glow_until` `_vision_cache` など、
        #     手で並べると必ず取りこぼす）
        snap["envnum"] = self._numeric_attrs(u)
        # --- 反射（前庭動眼反射・視線誘導反射）は内部状態を持つオブジェクト ----
        refl = {}
        for nm in ("_vor", "_orienting"):
            v = getattr(u, nm, None)
            if v is not None:
                refl[nm] = copy.deepcopy(v)
        snap["reflex"] = refl
        # --- 視覚のキャッシュ（辞書なので上の数値の走査では拾えない）-----------
        #   `e_toy_env` は「前回の描画から VISION_MIN_DT 経っていなければ使い回す」
        #   ため、描いた画像（辞書）と時刻を持っている。
        #   注意：戻さないと**測定中に描いた画像**が学習に混ざりうる。
        snap["vis"] = {nm: copy.deepcopy(getattr(u, nm, None))
                       for nm in ("_vision_cache", "_vision_t")
                       if hasattr(u, nm)}
        if getattr(u, "vision", None) is not None:
            snap["vis_out"] = copy.deepcopy(getattr(u.vision, "sensor_outputs", None))
        return snap

    @staticmethod
    def _numeric_attrs(obj):
        """obj が持つ「数値・配列」の属性を控える。オブジェクトは触らない。

        注意：`model` `data` は MuJoCo の本体なので除く（ここで触ると壊れる）。
        注意：配列は必ず copy する（`data` のビューだと戻す意味がなくなる）。
        """
        if obj is None:
            return {}
        out = {}
        for nm, v in vars(obj).items():
            if nm in ("model", "data", "_model", "_data"):
                continue
            if isinstance(v, np.ndarray):
                out[nm] = v.copy()
            elif isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                out[nm] = v
            elif isinstance(v, bool):
                out[nm] = v
        return out

    def _restore(self, snap):
        """控えた状態に戻す。測定が体を進めた分を巻き戻す。"""
        import mujoco
        import random
        u = self.env.unwrapped
        d = u.data
        # 体を戻す。順序が大事：
        #   ①全部の配列を戻す ②派生量を作り直す（mj_forward）
        #   ③ソルバの前回解（qacc_warmstart）と加速度を**もう一度**戻す
        #     — mj_forward がこれらを上書きするので、後から入れ直さないと
        #       物理が「測定の続き」から解かれて静かに軌道が変わる。
        arr = snap["mjarr"]
        for nm, v in arr.items():
            cur = getattr(d, nm, None)
            if isinstance(cur, np.ndarray) and cur.shape == v.shape:
                cur[:] = v
        d.time = snap["time"]
        mujoco.mj_forward(u.model, d)
        for nm in ("qacc_warmstart", "qacc", "act_dot", "sensordata"):
            v = arr.get(nm)
            cur = getattr(d, nm, None)
            if v is not None and isinstance(cur, np.ndarray) and cur.shape == v.shape:
                cur[:] = v
        self.state["obs"] = snap["state"]["obs"]
        self.state["hidden"] = snap["state"]["hidden"]
        self.state["prev_a"] = snap["state"]["prev_a"]
        torch.set_rng_state(snap["torch_rng"])
        np.random.set_state(snap["np_rng"])
        random.setstate(snap["py_rng"])
        if "env_rng" in snap and getattr(u, "np_random", None) is not None:
            u.np_random.bit_generator.state = snap["env_rng"]
        for nm, v in snap["organs"].items():
            setattr(self.env, nm, v)
        # 筋の状態（活性化ダイナミクス）を戻す。ここが抜けると力んだまま再開する
        am = getattr(u, "actuation_model", None)
        if am is not None:
            for nm, v in snap.get("actu", {}).items():
                cur = getattr(am, nm, None)
                if isinstance(cur, np.ndarray) and isinstance(v, np.ndarray) \
                        and cur.shape == v.shape:
                    cur[:] = v          # 中身を書き戻す（参照を差し替えない）
                else:
                    setattr(am, nm, v)
        for nm, v in snap.get("envnum", {}).items():
            cur = getattr(u, nm, None)
            if isinstance(cur, np.ndarray) and isinstance(v, np.ndarray) \
                    and cur.shape == v.shape:
                cur[:] = v
            else:
                setattr(u, nm, v)
        for nm, v in snap.get("reflex", {}).items():
            setattr(u, nm, v)
        for nm, v in snap.get("vis", {}).items():
            setattr(u, nm, v)          # 視覚のキャッシュと描画時刻
        if "vis_out" in snap and getattr(u, "vision", None) is not None:
            u.vision.sensor_outputs = snap["vis_out"]

    # ------------------------------------------------------------ 記録
    def _record(self, step):
        """全プラグインの値＋太郎の状態を**1行**にして残す。

        【なぜ1行にまとめるか、2026-07-30】以前は各プラグインが自分で `ctx.log(行)` を
        呼んでいたので、道具を2つ以上使うと**CSVの行が道具ごとに分裂**していた。
        1行に揃えると、グラフを描く側は**列を決め打ちしなくてよい**
        （`run/tools/dashboard.py` が列を自動で見つけて全部描く）。

        注意：ここに入れるのは「記録」だけ。学習の数値は変えない。
        """
        t = self.taro
        row = {"step": step,
               # 太郎が生きた時間（判断×K×DT）。実時間ではない
               "life_min": round(step * self.cfg.K * DT / 60.0, 3),
               "age_months": round(float(self._cur_age), 4),
               "real_min": round((time.time() - self._t0) / 60.0, 2),
               # 探索の強さ（ノルアドレナリン由来）。0.05 + ne*0.45
               "noise": round(0.05 + t.ne.get_ne_level() * 0.45, 5),
               "ne_maturation": round(float(t.ne.maturation), 5)}
        if self._act_accum:      # 力の出し具合（0付近＝フリーズの警告）
            row["act_abs"] = round(float(np.mean(self._act_accum[-200:])), 5)
        if self._caps_accum:     # 行動の変化量（下がりすぎ＝固まった）
            row["d_action2"] = round(float(np.mean(self._caps_accum[-200:])), 6)
        if self._eff_accum:      # 努力（代謝コスト）
            row["effort"] = round(float(np.mean(self._eff_accum[-200:])), 5)
        if self.cfg.cerebellum:  # 小脳の馴染み度（自動化がどれだけ効くか）
            row["cereb_err"] = round(float(t.cereb.err_ema.item()), 5)
        # 【2026-08-24新設・作業C速度改善】cfg.profile=True のときだけ列が増える。
        #   既定（False）ではこのifブロックに一歩も入らない＝run.csvの列は
        #   1個も増えない（既存の全実験のCSVと1バイトも変わらない）。
        if self._prof_enabled:
            row["rss_mb"] = round(self._prof_proc.memory_info().rss / (1024.0 * 1024.0), 1)
            for k, v in self._prof.items():
                row[f"t_{k}_sec"] = round(v, 4)
            self._prof = {}      # このチェックポイント区間ぶんを記録したのでリセット
        for p in self.plugins:
            m = p.metrics(self.ctx)
            if m:
                row.update(m)
        self._log_row(row)

    # ------------------------------------------------------- チェックポイント
    def _checkpoint(self, step):
        """測るのはプラグイン。ここは呼ぶだけ。

        注意：測定は体を進めるので、前後で状態を控えて戻す（上の注記を参照）。
          `finally` で戻すのは、測定が例外で落ちても学習を汚さないため。
        """
        self.ctx.step = step
        snap = self._snapshot()
        try:
            for p in self.plugins:
                p.on_checkpoint(self.ctx)
        finally:
            self._restore(snap)
        self._record(step)
        life_min = step * self.cfg.K * DT / 60.0
        real_min = (time.time() - self._t0) / 60.0
        t = self.taro
        noise = 0.05 + t.ne.get_ne_level() * 0.45
        tags = [p.line(self.ctx) for p in self.plugins]
        tags = [x for x in tags if x]
        cereb_tag = (f"cereb=on(err={t.cereb.err_ema.item():.2f})"
                     if self.cfg.cerebellum else "cereb=off")
        # 【taro-C5】|行動|＝力の出し具合（0付近＝フリーズ警告）
        extra = ""
        if self._act_accum:
            extra += f" |act|={np.mean(self._act_accum[-200:]):.3f}"
            if self.cfg.effort_cost and self._eff_accum:
                extra += f" effort={np.mean(self._eff_accum[-200:]):.3f}(lam={self.cfg.effort_cost})"
        # 注意：タグはASCIIのみ（Windowsのcp932で出せない文字を混ぜると print が例外を投げ
        #   学習が途中で落ちる。上付き2・絵文字で実際に2回踏んだ＝落とし穴 項35）
        if self._caps_accum:
            extra += f" da2={np.mean(self._caps_accum[-200:]):.4f}(lam={self.cfg.caps})"
        print(f"[seed{self.cfg.seed} rew={self.cfg.reward} age={self._cur_age:.2f}mo] "
              f"life={life_min:.0f}min | " + "  ".join(tags) +
              f" | noise={noise:.3f}(mat={t.ne.maturation:.2f}) {cereb_tag}{extra}"
              f" real={real_min:.0f}min", flush=True)

    # ------------------------------------------------------------ 本体
    def run(self):
        """学習ループ。e_growth_train.py 1175〜1304行の写し。"""
        cfg, t = self.cfg, self.taro
        env = self.env
        state = self.state
        n_train = cfg.steps
        ckpt = cfg.checkpoint
        self._act_accum, self._eff_accum, self._caps_accum = [], [], []
        from learning_progress import smoothness_cost, LearningProgress
        # 自己接触の興味度ボーナス（reach_self専用、2026-08-03）用の、progress本体
        #   （t.lp）とは完全に独立な新しいトラッカー。初回tickで実測値から初期化する
        #   （固定init=1.0は使わない。落とし穴チェックリスト項96と同じ初期化
        #   アーティファクトを避けるため）。設計：作業記録（非公開）
        #   2026-08-03_reach_self専用新奇性報酬の設計.md（3.3節）。
        self._self_touch_lp = None
        pe_fast, pe_slow = t.lp.pe_fast, t.lp.pe_slow
        reach_goal, reach_prev_dist = None, 0.0

        if cfg.grows:
            print(f"[body-growth] 月齢 {float(cfg.age_months or 0.0)} → {float(cfg.age_to)}／"
                  f"開始{cfg.age_start}回目・{cfg.age_ramp or '一瞬'}回かけて・"
                  f"{cfg.age_every}回ごとに見直し", flush=True)

        self._checkpoint(0)
        for i in range(n_train):
            # ---- 体を育てる（age_to が無ければ何もしない）----------------------
            if cfg.grows and cfg.age_every > 0 and i > 0 and i % cfg.age_every == 0:
                na = cfg.age_at(i)
                if abs(na - self._cur_age) > 1e-9:
                    self._regrow(na)
                    print(f"[body-growth] {i}回目：月齢 {self._cur_age:.3f} → {na:.3f} ヶ月",
                          flush=True)
                    self._cur_age = na
                    env = self.env            # 作り直したので持ち替える
            # ---- 感覚を受け取り、内部表現を作る -------------------------------
            _prof_t = self._prof_t0()   # 【作業C】cfg.profile=False（既定）なら常に0.0
            obs_in = state["obs"]       # 環境を進める**前**の観測（原因追跡用に控える）
            sv = t.fusion.encode(state["obs"])
            cf = t.target_fusion.encode(state["obs"]).detach()
            clp = t.encode_target(state["obs"])
            # 手先位置の目標表現（案C）。既存の clp/encode_target とは別枠で並行に
            #   計算する（設計の統合判断「決定3・決定4」）。既定では None のまま
            #   ＝下の分岐は一度も実行されない。
            gclp = None
            reach_space = bool(cfg.goal_babbling and cfg.goal_space == "reach_self")
            if reach_space:
                gclp = t.encode_reach_goal(state["obs"])
            z, kl, rc, hn = t.infer_latent(sv, state["prev_a"], cf, state["hidden"].detach())
            # ---- 行動を作る（運動野＋小脳のブレンド）--------------------------
            mean, std, _w_c, e_c = t.motor_drive(z)
            # ---- 目標指向の探索（Goal Babbling）------------------------------
            # 切替：fixed=i%2固定／ne=ノルアドレナリン／pe=予測誤差+NE（既定）
            # 注意：どちらの実装（legacy=goal_buf／reach_self=reach_goal_buf）を
            #   使っているかで、判定に使うバッファを変える（決定4）。
            #   goal_space!="reach_self" のときは _gbuf=self.goal_buf で従来と完全に同一。
            goal_step = False
            _gbuf = self.reach_goal_buf if reach_space else self.goal_buf
            if cfg.goal_babbling and len(_gbuf) >= 64:
                if cfg.goal_switch == "fixed":
                    goal_step = (i % 2 == 0)
                elif cfg.goal_switch == "ne":
                    goal_step = torch.rand(1).item() < (1.0 - t.ne.get_ne_level())
                else:      # "pe"：驚きを主役＋NEを下駄
                    # 【逸脱/工学近似 注意】向き（驚き大→探索）はEFE/LC-NE/予測符号化に基づく
                    # 人間模倣だが、"足し算・等重み・この正規化"という式には根拠なし＝恣意的。
                    rel = min(pe_fast / (pe_slow + 1e-6), 2.0) / 2.0
                    explore_drive = min(t.ne.get_ne_level() + rel, 1.0)
                    goal_step = torch.rand(1).item() < (1.0 - explore_drive)
            if goal_step:
                if reach_space:
                    # ---- 手先位置の目標表現（案C・Goal Babbling段階1）--------
                    if cfg.goal_trajectory:
                        if not self.reach_traj.active():
                            use_home = torch.rand(1).item() < cfg.goal_home_prob
                            if cfg.goal_negative_control:
                                target_g = t.dummy_reach_goal(gclp)
                            elif use_home:
                                target_g = t.home_reach_goal
                            else:
                                target_g = self.reach_goal_buf[
                                    torch.randint(len(self.reach_goal_buf), (1,)).item()]
                            self.reach_traj.begin(target_g)
                        g_step = self.reach_traj.waypoint(gclp)
                    else:
                        # 診断用：軌道を使わず、いきなり目標へ向かう（goal_trajectory=False）
                        use_home = torch.rand(1).item() < cfg.goal_home_prob
                        if cfg.goal_negative_control:
                            g_step = t.dummy_reach_goal(gclp)
                        elif use_home:
                            g_step = t.home_reach_goal
                        else:
                            g_step = self.reach_goal_buf[
                                torch.randint(len(self.reach_goal_buf), (1,)).item()]
                    mean = t.infer_reach_goal_action(z.detach(), gclp, mean, g_step)
                elif cfg.closed_loop_reach:
                    # 1つの目標を「届く/停滞」まで保持してにじり寄る（held goal、legacy）
                    if reach_goal is None:
                        reach_goal = self.goal_buf[
                            torch.randint(len(self.goal_buf), (1,)).item()].clone()
                        reach_prev_dist = mse(clp, reach_goal).item()
                    g = reach_goal
                    mean = t.infer_goal_action(z.detach(), clp, mean, g)
                else:
                    g = self.goal_buf[torch.randint(len(self.goal_buf), (1,)).item()]
                    mean = t.infer_goal_action(z.detach(), clp, mean, g)
            else:
                reach_goal = None            # 探索に切替 → リーチ終了（legacy）
                if self.reach_traj is not None:
                    self.reach_traj.reset()  # 探索に切替 → 軌道は打ち切り（reach_self）
            a, lp = t.brain.explore(mean, std)
            self._prof_add("brain_fwd", _prof_t)   # 【作業C】fusion.encode〜explore()
            self.goal_buf.append(clp.detach())
            if len(self.goal_buf) > 2000:
                # 注意：【逸脱・2026-07-30】先行研究は経験のカウントを**一度も捨てない**
                #   （離散版は 30,000ステップ通して累積し、それで「馴化」が起きる）。
                #   太郎は古い方から捨てるので、昔の経験の頻度情報が消える。
                self.goal_buf.pop(0)
            reach_pred = None
            if reach_space:
                self.reach_goal_buf.append(gclp.detach())
                if len(self.reach_goal_buf) > 2000:
                    self.reach_goal_buf.pop(0)
                reach_pred = gclp + t.reach_head(z.detach(), a.detach())
            pred = clp + t.nat_head(torch.cat([z, a.detach()], dim=-1))
            # 拮抗筋モード：a(n_joint) → to_env_action で筋活性化へ写像。OFFなら a_env==a
            a_env = t.brain.to_env_action(a)
            _prof_t = self._prof_t0()   # 【作業C】物理+描画+触覚（MIMo内部・env.step一式）
            state["obs"], term = self.step_k(rescale_action(a_env, env.action_space))
            self._prof_add("env_step", _prof_t)
            _prof_t = self._prof_t0()   # 【作業C】env.step後の脳forward（encode_target）
            nlp = t.encode_target(state["obs"])
            self._prof_add("brain_fwd", _prof_t)
            n_gclp = t.encode_reach_goal(state["obs"]) if reach_space else None
            if cfg.closed_loop_reach and reach_goal is not None:
                nd = mse(nlp, reach_goal).item()
                if nd >= reach_prev_dist:
                    reach_goal = None        # 近づかなくなった＝停止条件
                else:
                    reach_prev_dist = nd
            if cfg.replay:
                t.hippo.record(sv.detach(), state["prev_a"].detach(), a.detach(),
                               cf.detach(), clp.detach(), nlp.detach(),
                               state["hidden"].detach())
            # ---- 測る（プラグイン）------------------------------------------
            # 注意：判断ごとに1回（K tick ぶんに1回）呼ぶ。元の実装と同じ頻度。
            #   接触のような一瞬の事象は tick 単位で見た方が正確だが、
            #   まず元と同じ数値が出ることを確かめるため頻度も合わせる。
            self.ctx.step = i + 1
            # 【2026-08-19新設・F1-4b】語から注意への読み出し回路（部品2の後半）。
            #   cfg.word_attentionがNoneなら_apply_word_attention内で即returnし、
            #   DINOv2 encode等の追加計算は一切走らない（既定挙動・コスト不変）。
            _prof_t = self._prof_t0()   # 【作業C】語彙(DINOv2)読み出し＋発話生成
            self._apply_word_attention(state["obs"])
            # 【2026-08-22新設・F2】見た物の名前を言う（初語）。cfg.produceがNoneなら
            #   _apply_word_production内で即returnし、逆引き・generate()等の追加計算は
            #   一切走らない（既定挙動・コスト不変。上のword_attentionと同じ流儀）。
            self._apply_word_production(state["obs"])
            self._prof_add("produce", _prof_t)
            # このステップの内部の値を「置いておく」だけ（プラグインは読むだけ）。
            #   同じシードで結果がばらつく原因を追うのに使う（落とし穴 項79）。
            #   注意：参照を入れるだけなので計算はしない＝学習の数値は変わらない。
            self.ctx.last = {"obs_in": obs_in, "obs_out": state["obs"], "sv": sv,
                             "cf": cf, "clp": clp, "z": z, "mean": mean, "std": std,
                             "a": a, "pred": pred, "nlp": nlp}
            # 【2026-08-18新設・F1-3】この判断(K tick)ぶんの親の発話イベント（無ければ空）。
            #   ctx.last_double_touchと同じ「置いておくだけ」の流儀（プラグインは読むだけ）。
            self.ctx.last_parent_utterance = list(self._last_parent_events)
            _prof_t = self._prof_t0()   # 【作業C】測る道具（プラグイン）
            for p in self.plugins:
                p.on_step(self.ctx)
            self._prof_add("plugin", _prof_t)
            # ---- 学習 --------------------------------------------------------
            _prof_t = self._prof_t0()   # 【作業C】報酬・RPE・逆伝播（学習）
            pe = t.block_pe(pred, nlp)
            progress = t.lp.update(pe.item())
            pe_fast, pe_slow = t.lp.pe_fast, t.lp.pe_slow
            # 内発的動機。progress＝学習進度（誤差が減っていれば正）。
            #   none＝内発的報酬を一切与えない対照条件（2026-08-13、内発的報酬完全OFFの
            #   実装）。rew_task=0.0固定。progress自体（上のt.lp.update呼び出し）は
            #   cfg.rewardの値に関わらず毎tick計算し続ける（ctx.last_rewardのログに
            #   progress/pe_fast/pe_slowを出すため。仕様「やること」節の判断根拠）。
            if cfg.reward == "progress":
                rew_task = progress
            elif cfg.reward == "none":
                rew_task = 0.0
            elif cfg.reward == "posture_height":
                # 【2026-08-15・座位保持の学習】頭の高さ報酬（MIMo公式standup.py流用、
                #   人間模倣からの逸脱として登録済み）。double_touchガード（後述の
                #   t.double_touch is not None ブロック内のforループ）は経由しない
                #   （double_touch無効の実験では一度も呼ばれず報酬が不発火になる、
                #   という設計段階で判明した重大な制約。設計1-2節(c)参照）。
                rew_task = self.env.unwrapped.posture_height_reward()
            else:
                rew_task = t.brain.sensorimotor_reward(pe.item())
            rew = rew_task
            if cfg.effort_cost:
                # 【taro-C5】努力コスト：活性化²の筋力重み付き平均を報酬から引く
                # （Selinger 2015 等の代謝最小化。注意二乗・λ・重みは近似＝感度確認対象）
                effort = float((a.detach() ** 2 * t.eff_w).sum())
                rew = rew_task - cfg.effort_cost * effort
                self._eff_accum.append(effort)
            # 【CAPS, Mysore et al. 2021】行動の急変にペナルティ。
            # λ=0 でも**必ず記録する**：OFF側の値が無いと「λが弱くて効かなかった」のか
            #   「元からこの値なのか」を切り分けられない（2026-07-25 に実際に困った）。
            smooth = smoothness_cost(a.detach(), state["prev_a"].detach())
            self._caps_accum.append(smooth)
            if cfg.caps:
                rew = rew - cfg.caps * smooth
            self._act_accum.append(float(a.detach().abs().mean().item()))
            # 頭へのダブルタッチを報酬に直結する（2026-08-03、2026-08-05に全身一般化）。
            #   t.double_touch が None（既定）のときはこのブロックは一度も実行されない
            #   ＝既存実験の挙動は1ビットも変わらない。
            #   【足す順序】効果コスト・CAPSペナルティを引いた"後"の rew に足す
            #   （＝「疲れても・急変しても、頭に触れたことそのものは変わらず報われる」。
            #   努力コストで割り引かれると、ダブルタッチのボーナスが埋もれて
            #   探索を駆動しにくくなるため。CAPSも同様＝接触の瞬間の速い動きを
            #   罰でかき消したくない）。
            #   【2026-08-05：全身一般化・設計1-4節/1-6節】
            #   ・ガードから reach_space を外し、t.double_touch is not None のみにした
            #     （reach_self・touch=true専用という制約を断つ）。
            #   ・detect() は自前構築のtouch_map経由（引数から t.target_fusion.touch を
            #     渡す旧経路は廃止）で、touched_names=cfg.double_touch_touched_groups を渡す。
            #   ・「hitならボーナスを足す」1行は t.reward_contributors のforループへ
            #     切り出した（compute(ctx)->floatを持つオブジェクトのリスト）。
            if t.double_touch is not None:
                hit, toucher_p, touched_p, hit_names = t.double_touch.detect(
                    to_tensor(state["obs"]["touch"]),
                    toucher_name=f"{cfg.reach_arm_side}_palm",
                    touched_names=cfg.double_touch_touched_groups)
                # ログ用に生のpresenceを置いておく（プラグインが on_step 後に読める
                #   ようにするため。self.ctx.last と同じ「置いておくだけ」の流儀）。
                #   contributor.compute(ctx) はこの値（'hit'）を読むので、
                #   forループより"前"に必ず置く。
                self.ctx.last_double_touch = {
                    "hit": hit, "toucher_presence": toucher_p,
                    "touched_presence": touched_p, "hit_names": hit_names,
                    "self_interest_touch_pe": None,
                    "self_interest_progress": None,
                    "self_interest_bonus": 0.0}
                for contributor in t.reward_contributors:
                    rew = rew + contributor.compute(self.ctx)
                # 自己接触の興味度ボーナス（reach_self専用、2026-08-03）。
                #   設計：作業記録（非公開）
                #   2026-08-03_reach_self専用新奇性報酬の設計.md（3.4節）＋
                #   レビュー（同フォルダ\2026-08-03_reach_self新奇性報酬_レビュー.md、
                #   命名をnovelty→interestへ修正）。
                #   【2026-08-05】以前は外側の if reach_space and t.double_touch is
                #   not None: に暗黙に含まれていたため reach_space ガードが効いて
                #   いた。外側のガードから reach_space を外したので、この
                #   reach_self専用ブロックだけ明示的に reach_space で囲み、挙動を
                #   変えない（設計1-4節：このボーナスは t.reward_contributors に
                #   含めない）。
                #   既定 self_touch_interest_bonus=0.0＝OFF。このifの中身は一度も
                #   実行されない＝既存実験の挙動は1ビットも変わらない
                #   （double_touch_bonusのように「0.0を足す」のではなく、
                #   そもそも計算自体をスキップする、より強い保証）。
                interest_touch_pe, interest_progress, interest_bonus = None, None, 0.0
                if reach_space and cfg.self_touch_interest_bonus:
                    # 【実装上の重大な注意】t.block_pe(pred, nlp, blocks=touch_blocks) は
                    #   使わない。taro_setup.py の block_pe は「ブロック数が1個以下なら
                    #   固有感覚のみの旧実装と完全に同一にする」特例
                    #   （if not use_blocks or len(use_blocks) <= 1: return mse(pred, target)）
                    #   を持ち、touch_embed は通常1個しかないため、この特例に落ちて
                    #   「触覚だけの誤差」のつもりが全身まるごとの誤差を返してしまう
                    #   （値が返るので例外が出ず気づきにくい、設計3.5節）。
                    #   ここでは対象ブロックだけを自分でスライスしてMSEを取る。
                    touch_blocks = [(s, e, nm) for (s, e, nm) in (t.blocks or ())
                                    if nm in ("touch_embed", "touch")]
                    if not touch_blocks:
                        raise ValueError(
                            "self_touch_interest_bonus には touch_embed ブロックが要る"
                            "（cfg.touch_mode=\"target\"。config._checkで弾いているはず"
                            "だが、ここに来た場合は t.blocks の構成が想定と違う）。")
                    tot, wsum = 0.0, 0.0
                    for (s, e, nm) in touch_blocks:
                        tot = tot + mse(pred[..., s:e], nlp[..., s:e])
                        wsum += 1.0
                    interest_touch_pe = float((tot / wsum).item())
                    if self._self_touch_lp is None:
                        self._self_touch_lp = LearningProgress(
                            tau_fast=t.lp.tau_fast, tau_slow=t.lp.tau_slow,
                            init=interest_touch_pe)
                    else:
                        interest_progress = self._self_touch_lp.update(interest_touch_pe)
                    # SAGG-RIAC式：符号を捨てて絶対値を興味度にする（Baranes & Oudeyer
                    #   2013。「増加も減少も両方に興味を持つ」）。hitのtickだけに足す
                    #   （毎tick追跡はするが、報酬に足すのは自己接触の瞬間のみ）。
                    if hit and interest_progress is not None:
                        interest_bonus = cfg.self_touch_interest_bonus * abs(interest_progress)
                        rew = rew + interest_bonus
                if os.environ.get("TARO_DEBUG_SELF_TOUCH_INTEREST"):
                    # progress（progress本体、t.lp由来）も同じ行に出す。8.2(c)の
                    #   「既存progressが同じtickでマイナスなのに、新しいボーナスは
                    #   0以上」を直接見比べるため（progressはこの少し上で確定済み）。
                    print(f"[DEBUG_SELF_TOUCH_INTEREST] step={i+1} hit={hit} "
                          f"touch_pe={interest_touch_pe} local_progress={interest_progress} "
                          f"bonus={interest_bonus} body_progress={progress}", flush=True)
                # interest系の値を確定させてからログ辞書へ反映する（reach_self以外
                #   では None/0.0 のまま＝上で先に置いた初期値と同じ）。
                self.ctx.last_double_touch["self_interest_touch_pe"] = interest_touch_pe
                self.ctx.last_double_touch["self_interest_progress"] = interest_progress
                self.ctx.last_double_touch["self_interest_bonus"] = interest_bonus
            # 方策の学習（ドーパミン）は努力コスト込みの報酬を見る＝「疲れは損」を学ぶ
            rpe = t.dop.compute_rpe(rew)
            pl = t.learner.learn_action([lp], rpe)
            hl = t.homeo.homeostatic_loss(sv); t.homeo.observe(sv)
            t.learner.update(pe + hl + kl + rc, pl)
            self._prof_add("learn", _prof_t)
            # ---- 測る（プラグイン、rew・rpe確定後）--------------------------
            #   on_step（694行目付近）はrew・rpe確定"前"に呼ばれるため、これらを
            #   読みたいプラグインはここで新設した on_step_late を実装する
            #   （2026-08-03、設計「接触時の報酬直接測定の設計」1節）。
            #   pe_fast/pe_slow/progress_core/surprise_trace/progress は
            #   2026-08-04追加（設計「progress報酬の内訳露出と検証実験の設計」1節）。
            #   既存キー（rew, rpe）は変更しない。t.lp が既に持っている値を読むだけで、
            #   新規計算・学習への副作用は一切無い。
            self.ctx.last_reward = {
                "rew": rew, "rpe": rpe,
                "pe_fast": pe_fast, "pe_slow": pe_slow,
                "progress_core": pe_slow - pe_fast,
                "surprise_trace": t.lp._surprise_trace,
                "progress": progress,
            }
            _prof_t = self._prof_t0()   # 【作業C】測る道具（プラグイン、rew確定後）
            for p in self.plugins:
                p.on_step_late(self.ctx)
            self._prof_add("plugin", _prof_t)
            if reach_space:
                # 手先位置の目標表現（案C）：reach_head の学習は**独立optimizer**
                #   （設計の統合判断「決定2」）。既存の自己モデル学習（pe+hl+kl+rc）
                #   には混ぜない＝2026-07-30までの全実験の学習曲線の基準を汚さない。
                #   reach_pred/n_gclp は z.detach()/a.detach() 経由なので、この
                #   backward は reach_head のパラメータにしか勾配を流さない
                #   （小脳(cere_opt)と同じ「別のAdamで独立に更新する」パターン）。
                reach_pe = t.block_pe(reach_pred, n_gclp, blocks=t.reach_blocks)
                t.reach_opt.zero_grad(); reach_pe.backward(); t.reach_opt.step()
            if cfg.cerebellum:
                # 小脳は方策とは別に、実際に行った運動を教師なしで真似て自動化する
                closs = t.cereb.imitation_loss(z.detach(), a.detach())
                t.cere_opt.zero_grad(); closs.backward(); t.cere_opt.step()
                t.cereb.observe(e_c)          # 馴染み度の基準を更新（自己正規化）
            t.dev_clock.tick()                # ③発達年齢を進める（覚醒中の学習1回）
            # 【NE decoupling】NE（探索）にはタスク報酬だけを見せる（努力コスト抜き）。
            # 努力コストで下がった報酬をNEに見せると「失敗した→探索せよ」と誤読し、
            # 大振幅ノイズが努力コストを打ち消す自滅ループになる（実測 noise 0.095→0.5）。
            t.ne.observe_reward(rew_task); t.ne.release_ne()
            if term:
                reach_goal = None
                if self.reach_traj is not None:
                    self.reach_traj.reset()
            if cfg.mature:
                # 成熟は sim秒でなく発達年齢（学習回数）で駆動する
                t.ne.mature(t.dev_clock.progress(n_train))
            state["hidden"] = hn.detach()
            if cfg.efference_copy:
                # 【なぜ、2026-08-06】遠心性コピー切替(cfg.efference_copy)。既定True＝
                #   これまでの無条件更新と完全に同じ挙動。Falseにするとprev_aが
                #   常にゼロのまま固定される（C側のC_EFFCOPYと同じ形）。
                state["prev_a"] = a.detach()
            if term:
                self.reset_state()
            if cfg.replay and (i + 1) % ckpt == 0:
                self.consolidate()            # 睡眠：この間の経験を再生して定着
            if (i + 1) % ckpt == 0:
                self._checkpoint(i + 1)

        # ---- 後片付け --------------------------------------------------------
        out = {}
        for p in self.plugins:
            rep = p.report(self.ctx)
            if rep:
                out[p.name] = rep
        if cfg.save:
            self.taro.save(os.path.join(_ROOT, cfg.save) if not os.path.isabs(cfg.save)
                           else cfg.save,
                           extra={"age_final": self._cur_age})
        # 注意：env を閉じるのは呼び出し側（train / view）の finally に任せる。
        #   ここで閉じると、途中で例外が出たときに閉じられないまま残る。
        return out


def close_env(env):
    """環境を閉じる。閉じる処理そのものの失敗で本来の例外を隠さない。"""
    if env is None:
        return
    try:
        _release_renderers(env)     # 描画用メモリを先に返す
        env.close()
    except Exception as e:      # noqa: BLE001
        # 注意：握りつぶすが黙らない。閉じ損ねはメモリが残るだけで結果は汚さない。
        print(f"注意[close] 環境を閉じるときに失敗: {type(e).__name__}: {e}", flush=True)


def train(cfg, *, plugins=(), verbose=True, log_row=None):
    """設定から学習を1本回す。これが新しい経路の本体。

    注意：途中で例外が出ても環境を必ず閉じる（`finally`）。閉じ損ねると
      MuJoCo の描画コンテキストが残り、次の実行が不安定になる。
    """
    tr = Trainer(cfg, plugins=plugins, verbose=verbose, log_row=log_row)
    try:
        tr.build()
        return tr.run()
    finally:
        close_env(getattr(tr, "env", None))
