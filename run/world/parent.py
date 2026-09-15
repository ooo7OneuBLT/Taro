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
        vanish_silent_targets  既定空集合。ここに載った的（例："toy5"）は消失時に
                             「○○ないね」を言わない（親は黙る。仕様_短期B）。
                             発話イベント.csvには記録が残らない（採点は注意.csv
                             のvanished立ち上がりで行う。仕様書「後半」参照）。
    """

    _PICK = "pick"
    _SHAKE = "shake"
    _REFRACTORY = "refractory"
    # 【消失発話・2026-09-05・仕様_M2】机が空のまま「○○ないね」を1回言う状態。
    #   _PICKで次の的を選ぶ前に、旧的があればここを経由する（vanish_utterance時のみ）。
    _VANISH = "vanish"

    def __init__(self, enabled=False, shake_amp_m=0.02, shake_hz=1.5,
                 respond_to_voice=False, voice_respond_prob=0.4,
                 voice_delay_range=(0.135, 1.408),
                 # 【親の言い直し・2026-09-03】4つともNoneが既定＝従来どおり
                 #   voice_respond_prob・_pick_textを使う（1ビットも変えない）。
                 voice_respond_prob_correct=None, voice_respond_prob_error=None,
                 voice_text_correct=None, voice_text_error=None,
                 gaze_deg=10.0, gaze_margin_deg=3.0, gaze_hold_sec=3.0,
                 attract_timeout_sec=12.0, utterances=None, respond_prob=1.0,
                 refractory_sec=1.0, shuffle_refractory_sec=3.0, shuffle_after=5,
                 repeat_labels=0, repeat_gap_sec=2.0, repeat_timeout_sec=4.0,
                 spin_period_sec=8.0, fixed_yaw_deg=0.0,
                 present_angles=None,
                 cycle_pick=False,
                 solo_presentation=False, shake_pause_sec=2.0,
                 follow_gaze=False, follow_dist=0.086, follow_speed=0.5,
                 vanish_utterance=False, vanish_template="{word}ないね",
                 vanish_gap_before_sec=1.0, vanish_gap_after_sec=1.5,
                 vanish_wait_notice=False, vanish_notice_delay_sec=0.5,
                 vanish_notice_max_sec=3.0, vanish_silent_targets=(),
                 park_after_sec=None, park_offset_deg=20.0):
        self.enabled = bool(enabled)
        self.shake_amp_m = float(shake_amp_m)
        self.shake_hz = float(shake_hz)
        self.gaze_deg = float(gaze_deg)
        self.gaze_margin_deg = float(gaze_margin_deg)
        # 【F2-9・2026-08-26】0.3→3.0。学習が成立するとき、子の視界では命名の
        #   約3秒前から対象が優位（Yu & Smith 2012 [Tier2・N=6]）。効かなければ
        #   最初に疑ってよい値（設計_F2-9 技術付録A・落とし穴の節）。
        self.gaze_hold_sec = float(gaze_hold_sec)
        # 【F2-9】3.0→12.0。gaze_hold_sec(3.0) < timeout が必須の従属変更。
        self.attract_timeout_sec = float(attract_timeout_sec)
        # gaze_hold >= timeout だと注視カウントが完走する前に必ず諦める
        #   ＝一度も発話しない。実験ファイルの上書きミスをここで止める
        #   （F2-8の閾値スケール事故と同型の予防。設計_F2-9 落とし穴の節）。
        if self.gaze_hold_sec >= self.attract_timeout_sec:
            raise ValueError(
                f"gaze_hold_sec({self.gaze_hold_sec}) >= attract_timeout_sec"
                f"({self.attract_timeout_sec})。この設定では親が一度も発話できない")
        self.utterances = dict(utterances or {"toy1": "ぶーぶー", "toy2": "わんわん"})
        self.respond_prob = float(respond_prob)
        # 【発話の動機・2026-08-31・設計_発話の動機.md 決定B'】太郎の声への反応。
        #   既定OFF（respond_to_voice=False）＝既存シーンは1ビットも変わらない。
        #   反応率0.4＝親は乳児の声の3〜5割に反応（Gros-Louis et al. 2006系・Tier2）。
        #   遅れ0.135〜1.408秒＝母親の応答潜時の中央値の範囲（Hilbrink, Gattis &
        #   Levinson 2015・一次確認）。分布の形が取れないため範囲から一様に引く。
        #   親は正誤を判定しない：声が聞こえたら「いま見せている物の名前」を言う
        #   だけ（正解なら承認、間違いなら言い直しが、判定ゼロで結果的に生まれる）。
        self.respond_to_voice = bool(respond_to_voice)
        self.voice_respond_prob = float(voice_respond_prob)
        self.voice_delay_range = tuple(float(x) for x in voice_delay_range)
        # 【親の言い直し・2026-09-03】設計_親の言い直し.md 第2部。Noneなら
        #   従来どおり voice_respond_prob 1本・_pick_text を使う。
        self.voice_respond_prob_correct = (
            None if voice_respond_prob_correct is None else float(voice_respond_prob_correct))
        self.voice_respond_prob_error = (
            None if voice_respond_prob_error is None else float(voice_respond_prob_error))
        self.voice_text_correct = voice_text_correct
        self.voice_text_error = voice_text_error
        self._voice_timer = None       # None=予約なし／秒=満了で返事
        self._voice_correct = None     # 予約時点の正誤判定（満了時まで固定）
        self.n_voice_responses = 0
        self.refractory_sec = float(refractory_sec)
        self.shuffle_refractory_sec = float(shuffle_refractory_sec)
        # 【連呼・2026-09-02・F2-40】最初の名づけの後、太郎が注視を続けている間に
        #   repeat_gap_sec おきに最大 repeat_labels 回、同じ的を言い直す。
        #   人間の親は1回の提示で同じ語を繰り返す（育児語の反復【Tier2】・
        #   Gogate 2000 の同期）。既定0＝従来どおり1提示1発話＝挙動不変。
        #   見るのをやめたら打ち切る（follow-inの原則を維持）。
        self.repeat_labels = int(repeat_labels)
        self.repeat_gap_sec = float(repeat_gap_sec)
        self.repeat_timeout_sec = float(repeat_timeout_sec)
        # 【回転の切替・2026-09-02・F2-42】提示中の回転周期[秒]。0以下なら回さず
        #   fixed_yaw_deg の向きで固定（傾き25度は共通）。既定8.0＝従来どおり。
        self.spin_period_sec = float(spin_period_sec)
        self.fixed_yaw_deg = float(fixed_yaw_deg)
        # 【F2-49・2026-09-03・ユーザー指摘】物ごとに「それらしく見える」角度で見せる。
        #   {"toy6": {"yaw": 30, "tilt": 65}, ...}。無い物は fixed_yaw_deg／傾き25度（従来）。
        #   人間の親は皿なら面を、靴なら横顔を子に向ける（物ごとの定番の見せ方）。
        self.present_angles = {str(k): dict(v) for k, v in dict(present_angles or {}).items()}
        # 【一巡シャッフル・2026-09-02・ユーザー承認】cycle_pick=True なら候補を
        #   シャッフルして順に全部見せ、使い切ったらまたシャッフル（1巡の中で回数が均等）。
        #   人間の親は手元のおもちゃを満遍なく見せる。既定False＝従来の毎回一様乱数。
        self.cycle_pick = bool(cycle_pick)
        self._deck = []
        self._repeat_attempt = False     # いまのSHAKEが同じ的の呼び戻しか
        self._repeats_left = 0
        self.n_repeats = 0
        self.shuffle_after = int(shuffle_after)
        self.solo_presentation = bool(solo_presentation)
        # 【F2-9・2026-08-26】振りのバースト化（motionese）。人間の親は乳児相手だと
        #   「動かす→止める」を繰り返し、静止時間の割合が大人相手の約3倍
        #   [Tier2・van Schaik 2020]。振る長さは2往復（2/shake_hz秒）、止まる長さは
        #   shake_pause_sec。比率（静止多め）は文献、秒数自体は恣意値[Tier3]。
        self.shake_burst_sec = 2.0 / self.shake_hz
        self.shake_pause_sec = float(shake_pause_sec)
        # 【F2-9C・2026-08-26】視線への差し出し（follow_gaze）。
        #   実測（gaze_probe・120秒）：学習中の太郎の視線は的の10度以内に7.5%しか
        #   おらず、連続では最大0.4秒しか留まらない＝「太郎が偶然見るのを待つ」
        #   方式は成立しない。親が**太郎のいま見ている方向へ物を持って行き**、
        #   視界に入れてから名付ける方式に変える。人間の親も子の視界に物を
        #   差し出してから名付ける（follow-in labeling の実際の形。Gogate 2000 の
        #   同期命名も物を動かして見せながら言う）[Tier2・親の自然な行動]。
        #   follow_dist   顔から物までの距離[m]（既定0.086=従来のアンカー距離と同じ）
        #   follow_speed  親の手の速さの上限[m/s]（0.5。人間の手の到達運動の
        #                 速度域から。厳密な文献値ではない[Tier3]）
        self.follow_gaze = bool(follow_gaze)
        self.follow_dist = float(follow_dist)
        self.follow_speed = float(follow_speed)
        # 【消失発話・2026-09-05・仕様_M2_消えた瞬間に親が「○○ないね」と言う】
        #   既定False＝従来どおり_PICKで即座に次の的を選ぶ（1ビットも変わらない）。
        #   Trueなら、旧的を隠したまま before 秒待って「○○ないね」を1回言い、
        #   さらに after 秒待ってから次の的を選ぶ（_VANISH状態。update()参照）。
        self.vanish_utterance = bool(vanish_utterance)
        self.vanish_template = str(vanish_template)
        self.vanish_gap_before_sec = float(vanish_gap_before_sec)
        self.vanish_gap_after_sec = float(vanish_gap_after_sec)
        # 【M4c・2026-09-06・仕様_M4c_親は太郎が気づいてから「ないね」と言う】
        #   既定False＝従来どおり vanish_gap_before_sec 固定待ちで言う
        #   （1ビットも変わらない）。Trueなら、太郎の物体ファイルが「消えた」を
        #   立てた合図（object_files.py の env._taro_noticed_gone_time）から
        #   vanish_notice_delay_sec 秒後に言う。気づき待ちが
        #   vanish_notice_max_sec 秒を超えたら（合図が来なくても）諦めて言う
        #   （cause="vanish_timeout"。前半「決めたこと2」）。
        self.vanish_wait_notice = bool(vanish_wait_notice)
        self.vanish_notice_delay_sec = float(vanish_notice_delay_sec)
        self.vanish_notice_max_sec = float(vanish_notice_max_sec)
        # 【短期B・2026-09-06・仕様_短期B】消えたとき「○○ないね」を黙る的の集合。
        #   既定は空集合＝従来どおり全ての的で発話する（1ビットも変わらない）。
        #   ここに載った的は_VANISH状態でshould_speakが真になっても発話を返さず、
        #   代わりにenv.unwrapped._parent_silent_vanishへ(時刻,的)を置く（採点用の
        #   目印。読む側は無い。発話イベント.csvへの記録は無い＝仕様書の指示どおり）。
        self.vanish_silent_targets = set(vanish_silent_targets)
        # 【2026-09-11・K1の場面】差し出して park_after_sec 秒たったら、その時の
        #   視線から park_offset_deg 度だけ横へずらした所に固定する（追従をやめる）。
        #   None（既定）なら1行も通らない＝既定不変。状態は reset() 側。
        self.park_after_sec = (None if park_after_sec is None else float(park_after_sec))
        self.park_offset_deg = float(park_offset_deg)
        self.reset()

    def reset(self):
        """エピソード境界で呼ぶ（前エピソードの状態を持ち越さない。e_orienting.reset()と同じ位置づけ）。"""
        self._state = self._PICK
        self._target = None
        self._shake_t = 0.0
        self._voice_timer = None       # 声への返事の予約もエピソード境界で消す
        # 【親の言い直し・2026-09-03】予約時の正誤判定・的もエピソード境界で消す。
        self._voice_correct = None
        self._voice_target = None
        self.n_voice_responses = 0
        self._hold_t = 0.0
        self._silent_t = 0.0
        self._silent_until = 0.0
        self.n_utterances = 0     # 累計発話回数（testのshuffle判定にも使う）
        self.n_swaps = 0          # 累計入れ替え回数
        # 【F2-9】発話バーストの残り秒数。発話の瞬間に1往復ぶんセットされ、
        #   不応期に入っても使い切るまで振り続ける（_speak参照）。
        self._speech_burst_t = 0.0
        # 【F2-9C】差し出し中の物の現在位置（親の手の位置に相当）。Noneなら未開始。
        self._follow_pos = None
        # 【2026-09-11・K1の場面】差し出してから park_after_sec 秒たったら、
        #   その時の視線から park_offset_deg 度だけ横へずらした所に**固定**する
        #   （追従をやめる）。ユーザーの設計「生成できたらわざと視界の中央から
        #   ずらして、コップを見るように命令する」の『ずらす』にあたる。
        #   None（既定）なら1行も通らない＝既定不変。
        #   水平にずらすのにおもちゃ2個は要らない（位置を直接置くため。
        #   toy の angle_deg は e_toy_env:1716 の if self._toy2 の中でしか効かない）。
        self._park_pos = None
        self._follow_t = 0.0
        # 【F2-9】fit-check用の観測値：注視カウントが始まった回数と、
        #   gaze_hold_sec まで完走した回数（3秒が長すぎないかの判定材料）。
        self.n_hold_starts = 0
        self.n_hold_completes = 0
        self._holding_prev = False
        # 【消失発話・2026-09-05】_VANISH状態の経過秒／発話済みか。
        self._vanish_t = 0.0
        self._vanish_spoken = False
        self._vanish_spoken_t = 0.0
        # 【M4c・2026-09-06】隠した sim 時刻（気づき待ちの経過判定には self._vanish_t
        #   をそのまま使うので、これは記録用途。_enter_pick で更新される）。
        self._vanish_hide_t = 0.0

    # ------------------------------------------------------------------
    def update(self, env):
        """1物理stepぶん進める。発話が起きたstepだけ {"text","target"} を返す（他はNone）。

        env: ToySupineEnv自身（self）。位置の読み書き・_gaze_angle_to・_set_anchor
             ・np_random（乱数はenvのnp_randomを使う＝依頼書の指示）を使う。
        """
        if not self.enabled:
            return None
        # 【発話の動機・2026-08-31】太郎の声への反応。状態機械（PICK/SHAKE/
        #   REFRACTORY）とは独立のタイマーで動く。trainer が発話時に
        #   env._taro_voice_signal を立てる約束（設計_発話の動機.md 技術付録）。
        if self.respond_to_voice:
            if getattr(env, "_taro_voice_signal", False):
                env._taro_voice_signal = False
                if self._voice_timer is None:
                    # 【親の言い直し・2026-09-03】正誤の判定はここ（タイマー予約時）で
                    #   固定する。満了までに的(_target)が変わっても、予約時の判定を使う
                    #   （設計_親の言い直し.md 第2部「同じブロック」）。
                    tgt = self._target or "toy1"
                    taro_text = getattr(env, "_taro_voice_text", "") or ""
                    word = self._single_word(tgt)
                    correct = bool(word) and (
                        taro_text.startswith(word)
                        or taro_text.startswith(word.replace("ー", "")))
                    if (self.voice_respond_prob_correct is None
                            and self.voice_respond_prob_error is None):
                        prob = self.voice_respond_prob
                    else:
                        prob = (self.voice_respond_prob_correct if correct
                                else self.voice_respond_prob_error)
                        if prob is None:
                            prob = self.voice_respond_prob
                    if env.np_random.random() < prob:
                        lo, hi = self.voice_delay_range
                        self._voice_timer = lo + (hi - lo) * float(env.np_random.random())
                        self._voice_correct = correct
                        self._voice_target = tgt
            if self._voice_timer is not None:
                self._voice_timer -= float(env.dt)
                if self._voice_timer <= 0.0:
                    self._voice_timer = None
                    tgt = getattr(self, "_voice_target", None) or self._target or "toy1"
                    correct = bool(self._voice_correct)
                    if self.voice_text_correct is None and self.voice_text_error is None:
                        text = self._pick_text(env, tgt)
                    else:
                        word = self._single_word(tgt)
                        template = self.voice_text_correct if correct else self.voice_text_error
                        text = template.format(X=word) if template else self._pick_text(env, tgt)
                    self._voice_correct = None
                    if text:
                        self.n_voice_responses += 1
                        self.n_utterances += 1
                        return {"text": text, "target": tgt, "cause": "voice",
                                "correct": correct}
        if not getattr(env, "_toy", False):
            return None      # おもちゃが1つも無ければ成立しない
        if not getattr(env, "_toy2", False) and not self.follow_gaze:
            # 【F2-9C・2026-08-26】従来の待ち伏せ方式は「近い方／遠い方」の比較で
            #   注視を判定するため2個要る。差し出し方式(follow_gaze)は的を1個
            #   持って行くだけなので1個でも成立する（産出テスト用＝親が差し出すが
            #   名前は言わない構成で使う。respond_prob=0.0）。
            return None
        dt = float(env.dt)

        # 【F2-9・2026-08-26】1個提示は「振っている間だけ」から「常時」へ強化。
        #   以前は不応期に両方が戻り、目視で「まだ2個同時に出る」と紛らわしかった
        #   （ユーザー報告）。人間側でも、親は次のおもちゃに持ち替えるまで
        #   今の1個を持ち続けるのが自然。的（_target）が次の試行で替わった瞬間に
        #   見える物が入れ替わる＝持ち替えに相当。
        if self.solo_presentation and self._target is not None and self._state != self._VANISH:
            # 透明化の復元→隠し直し（毎step。状態を持たない流儀）。
            # 【消失発話】_VANISH中は下の専用ブロックが復元→全隠しをやるので、
            #   ここでは「旧的だけ見える」形にしない（二重処理を避ける）。
            for g, a0 in getattr(env, "_toy_alpha_backup", {}).items():
                env.model.geom_rgba[g, 3] = a0
            self._hide_other(env)

        if self._state == self._REFRACTORY:
            # 【F2-9】発話バースト：発話直後の1往復ぶんは不応期でも振り続ける
            #   （「言う瞬間に動かす」Gogate 2000 の同期。_speak参照）。
            if self._speech_burst_t > 0.0:
                self._apply_shake(env)
                self._shake_t += dt
                self._speech_burst_t -= dt
            elif self.follow_gaze and self._follow_pos is not None and                     self._target in getattr(env, "_present_slots", {}):
                # 【10択・2026-08-31・目視で発覚】差し出しスロットは環境の毎step退避
                #   に上書きされるため、不応期に地下へ消えて世界が空になっていた
                #   （旧toy1/toy2はアンカー保持があるので見え続ける＝非一貫）。
                #   不応期も親は手に持ったまま＝最後の提示位置に置き続ける。
                _sl = self._slots(env).get(self._target)
                if _sl is not None:
                    env._place(_sl["qadr"], np.array(self._follow_pos, dtype=float))
                    env.data.qpos[_sl["qadr"] + 3:_sl["qadr"] + 7] =                         self._present_quat(env, _sl["shape"])
                    env.data.qvel[_sl["dadr"]:_sl["dadr"] + 6] = 0.0
            self._silent_t += dt
            # 【連呼】間隔が来たら、まだ見ていれば言い直す。見ていなければ打ち切り
            if self._repeats_left > 0 and self._silent_t >= self.repeat_gap_sec:
                _ok = self._gaze_on_target(env)
                if _ok:
                    return self._speak_repeat(env)
                # 見ていなければここでは諦めず、不応期の終わりに呼び戻し（振り直し）へ
            if self._silent_t < self._silent_until:
                return None
            if self._repeats_left > 0:
                # 【連呼・呼び戻し】目をそらしていても持ち替えず、同じ的を振り直して
                #   見返した瞬間に言い直す（人間の親は同じ物を見せ続けて繰り返す）。
                self._state = self._SHAKE
                self._shake_t = 0.0
                self._hold_t = 0.0
                self._holding_prev = False
                self._repeat_attempt = True
            else:
                self._enter_pick(env)

        if self._state == self._VANISH:
            # 【消失発話・2026-09-05】机が空のまま「○○ないね」を1回言う（仕様書 前半）。
            #   毎step：透明化の復元→全スロットを隠し直す（状態を持たない流儀、
            #   solo_presentationブロックと同じ形。的の有無に関わらず全部隠す）。
            for g, a0 in getattr(env, "_toy_alpha_backup", {}).items():
                env.model.geom_rgba[g, 3] = a0
            self._hide_all(env)
            self._vanish_t += dt
            # 【M4c・2026-09-06】仕様_M4c 後半「2. 親」。wait_noticeが偽なら
            #   従来どおり固定待ち（vanish_gap_before_sec）で言う（1ビットも
            #   変わらない）。真なら、太郎の物体ファイルの「気づいた」合図
            #   （object_files.py が立てる env.unwrapped._taro_noticed_gone_time）
            #   から vanish_notice_delay_sec 秒後に言う。合図が
            #   vanish_notice_max_sec 秒たっても来なければ、気づき待ちの時間切れ
            #   として諦めて言う（cause="vanish_timeout"）。
            if not self._vanish_spoken:
                should_speak = False
                cause = "vanish"
                if not self.vanish_wait_notice:
                    should_speak = self._vanish_t >= self.vanish_gap_before_sec
                else:
                    u = getattr(env, "unwrapped", env)
                    noticed = getattr(u, "_taro_noticed_gone_time", None)
                    now = float(env.data.time)
                    if noticed is not None and now >= noticed + self.vanish_notice_delay_sec:
                        should_speak = True
                        cause = "vanish"
                    elif self._vanish_t >= self.vanish_notice_max_sec:
                        should_speak = True
                        cause = "vanish_timeout"
                if should_speak:
                    self._vanish_spoken = True
                    # 【既定不変】wait_notice=偽では従来どおり厳密に
                    #   vanish_gap_before_sec を基準にする（self._vanish_t の
                    #   実際値は dt刻みでこれをわずかに超え得るため、基準に
                    #   使うと after 判定の閾値が従来と1ビットずれてしまう）。
                    self._vanish_spoken_t = (self.vanish_gap_before_sec
                                              if not self.vanish_wait_notice
                                              else self._vanish_t)
                    # 【短期B・2026-09-06】黙る的なら発話を返さない。時間の進み方
                    #   （_vanish_spoken=True・_vanish_spoken_t確定）は上と同じに
                    #   保ち、次の_PICKへ移るタイミングだけ従来どおりにする。
                    #   代わりにenv.unwrapped._parent_silent_vanishへ目印を置く
                    #   （採点用。発話イベント.csvへの記録は無い＝プラグイン非経由）。
                    if self._target in self.vanish_silent_targets:
                        u = getattr(env, "unwrapped", env)
                        u._parent_silent_vanish = (float(env.data.time), self._target)
                        return None
                    word = self._single_word(self._target)
                    text = self.vanish_template.format(word=word)
                    return {"text": text, "target": self._target, "cause": cause}
            if self._vanish_spoken and (
                    self._vanish_t >= self._vanish_spoken_t + self.vanish_gap_after_sec):
                self._state = self._PICK
            return None

        if self._state == self._PICK:
            # 【10択・2026-08-31】候補＝セリフ(utterances)が定義されていて、かつ
            #   スロットが実在するもの全員からランダム。従来シーン（toy1/toy2の
            #   2択・または1個）では候補が同じなので挙動は変わらない。
            _cand = [k for k in sorted(self.utterances) if k in self._slots(env)]
            if not _cand:
                _cand = ["toy1"]
            if self.cycle_pick:
                if not self._deck or any(k not in _cand for k in self._deck):
                    self._deck = list(_cand)
                    perm = env.np_random.permutation(len(self._deck))
                    self._deck = [self._deck[int(i)] for i in perm]
                self._target = self._deck.pop(0)
            else:
                self._target = _cand[int(env.np_random.integers(len(_cand)))]
            self._state = self._SHAKE
            self._shake_t = 0.0
            self._hold_t = 0.0
            self._holding_prev = False   # 試行ごとに注視開始を数え直す（F2-9観測用）
            self._follow_pos = None      # 新しい的は今の視線の先から差し出し直す
            self._park_pos = None        # 【2026-09-11】ずらして固定した位置も解く
            self._follow_t = 0.0
            self._repeat_attempt = False

        # ここに来るのは常に SHAKE 状態（振っている最中）
        self._apply_shake(env)
        holding = self._gaze_on_target(env)
        if holding:
            self._hold_t += dt
        else:
            self._hold_t = 0.0     # 連続でないとカウントしない（「続いたら」の判定）
        # 【F2-9】fit-check用：注視の開始と、gaze_hold_secまでの完走を数える
        #   （3.0秒 [Tier2・N=6] が太郎の飽きと衝突していないかの判定材料）。
        if holding and not self._holding_prev:
            self.n_hold_starts += 1
        self._holding_prev = holding
        self._shake_t += dt

        if self._repeat_attempt:
            if self._hold_t >= min(1.0, self.gaze_hold_sec):
                return self._speak_repeat(env)
            if self._shake_t >= self.repeat_timeout_sec:
                self._repeats_left = 0
                self._repeat_attempt = False
                self._enter_pick(env)   # 呼び戻せなければ諦めて持ち替える
            return None
        if self._hold_t >= self.gaze_hold_sec:
            self.n_hold_completes += 1
            return self._speak(env)
        if self._shake_t >= self.attract_timeout_sec:
            self._enter_pick(env)   # 諦めて黙って次の試行へ（前半の設計意図どおり）
        return None

    # ------------------------------------------------------------------ 内部
    def _slots(self, env):
        """提示できるスロットの表（10択・2026-08-31）。

        {"toy1": {"qadr","dadr","bid","body","shape"}, ...}。
        toy1/toy2は既存の属性から、toy5以降は env._present_slots から組む。
        """
        d = {"toy1": {"qadr": env._toy_qadr, "dadr": env._toy_dadr,
                      "bid": env._toy_bid, "body": "test_object1",
                      "shape": getattr(env, "_toy_shape", "")}}
        if getattr(env, "_toy2", False):
            d["toy2"] = {"qadr": env._obj2_qadr, "dadr": env._obj2_dadr,
                         "bid": env._obj2_bid, "body": "test_object2",
                         "shape": getattr(env, "_toy2_shape", "")}
        for k, sl in getattr(env, "_present_slots", {}).items():
            d[k] = {"qadr": sl["qadr"], "dadr": sl["dadr"], "bid": sl["bid"],
                    "body": sl["body"], "shape": "asis"}
        return d

    def _gaze_on_target(self, env):
        """今の視線が self._target を「見ている」と判定できるか。

        判定：近い方のおもちゃであり、かつ他方との角度差が gaze_margin_deg 以上
        （2おもちゃの間隔24度に対する曖昧域対策。docstring冒頭の導出参照）。
        """
        slots = self._slots(env)
        sl = slots.get(self._target)
        if sl is None:
            return False
        a_t = env._gaze_angle_to(sl["body"])
        if a_t is None:
            return False
        # 【10択・2026-08-31】1個提示（solo）・スロット3個以上・おもちゃ1個の
        #   ときは角度のみで判定（他は退避・透明なので「近い方」の比較は無意味）。
        #   従来の2個並べ（solo無し）だけ margin 判定を残す（後方互換）。
        if self.solo_presentation or len(slots) != 2 or not getattr(env, "_toy2", False):
            return a_t <= self.gaze_deg
        other = "test_object2" if self._target == "toy1" else "test_object1"
        a_o = env._gaze_angle_to(other)
        if a_o is None:
            return False
        return a_t <= self.gaze_deg and (a_o - a_t) >= self.gaze_margin_deg

    def _follow_step(self, env):
        """差し出し（follow_gaze）：物の目標位置＝太郎の視線の先を計算し、
        親の手の速さの上限つきで1stepぶん近づけた現在位置を返す。

        【F2-9C・2026-08-26】視線方向は e_toy_env._gaze_dir と同じ計算
        （片目カメラの-z軸）。目標の高さは床下に沈まないよう5cmで打ち切る
        （視線が下を向いたとき物が床にめり込むのを防ぐだけの安全装置）。
        """
        cid = int(env.model.camera("eye_left").id)
        eye = np.array(env.data.cam_xpos[cid], dtype=float)
        fwd = -np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        goal = eye + fwd * self.follow_dist
        goal[2] = max(goal[2], 0.05)
        # 【2026-09-11・K1の場面】「ずらして固定」。park_after_sec が None なら素通り。
        if self.park_after_sec is not None:
            self._follow_t += float(env.dt)
            if self._park_pos is None and self._follow_t >= self.park_after_sec:
                right = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 0]
                a = np.radians(self.park_offset_deg)
                fwd2 = fwd * np.cos(a) + right * np.sin(a)   # 今の視線から横へ a 度
                self._park_pos = eye + fwd2 * self.follow_dist
                self._park_pos[2] = max(self._park_pos[2], 0.05)
            if self._park_pos is not None:
                return self._park_pos        # 追従をやめてそこに置き続ける
        if self._follow_pos is None:
            self._follow_pos = goal.copy()      # 初回は視線の先に直接現れる
            return self._follow_pos
        step = goal - self._follow_pos
        dist = float(np.linalg.norm(step))
        max_step = self.follow_speed * float(env.dt)
        if dist > max_step > 0.0:
            step = step * (max_step / dist)
        self._follow_pos = self._follow_pos + step
        return self._follow_pos

    def _face_quat(self, env):
        """板の絵の面（ローカル+y）が太郎の顔を向く姿勢（クォータニオン）を返す。

        【F2-9C・2026-08-26】実測（plate_diag.png）：差し出した板の「縁」が太郎に
        向いており、太郎の目には白い縦棒しか映っていなかった。親が絵本を子の顔へ
        向けて見せる動作に相当する回転を毎stepかける。板以外の形（box等）でも
        呼ばれるが、対称な塊では見た目が変わらないだけで害はない。
        """
        import mujoco as _mj
        cid = int(env.model.camera("eye_left").id)
        fwd = -np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        ey = -fwd / (np.linalg.norm(fwd) + 1e-12)      # 絵の面を太郎の方向へ
        ez = np.array([0.0, 0.0, 1.0])
        ez = ez - ey * float(ez @ ey)
        n = np.linalg.norm(ez)
        ez = ez / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        # 実測（plate_diag.png）：ローカル+zを上に向けると絵が上下逆さまに写った
        #   （MuJoCoのcube貼りのUV原点の向きによる）。zを下向きにして正立させる。
        ez = -ez
        ex = np.cross(ey, ez)
        q = np.empty(4)
        _mj.mju_mat2Quat(q, np.stack([ex, ey, ez], axis=1).ravel())
        return q

    def _present_quat(self, env, shape):
        """提示中のおもちゃの向き。板は絵の面を太郎へ（_face_quat）、
        3D物体(asis)は**直立のまま顔（ローカル+x）だけ太郎へ**回す。

        【V1b・2026-08-31・実測で発覚】_face_quat は板のUV補正でローカルzを
        下に向けるため、z-up で組んだ3D物体（犬等）が**逆さま**に提示されていた
        （F/logs/V1bプローブ/図_提示の問題.png。F2-19以降の全走行に影響）。
        人間の親はおもちゃを立てて、顔を子に向けて見せる——その形にする。
        """
        if shape != "asis":
            return self._face_quat(env)
        import mujoco as _mj
        cid = int(env.model.camera("eye_left").id)
        fwd = -np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2]
        ex = -fwd
        ex[2] = 0.0                       # 水平成分だけ＝直立を保つ（ヨー回転のみ）
        n = np.linalg.norm(ex)
        ex = ex / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        ey = np.cross(ez, ex)
        # 【2026-08-31・ユーザー指摘】1アングル固定をやめる。人間の親は物を回して
        #   いろんな面を見せる（ぼうし等は正面固定だと縁の線にしか見えない）。
        #   提示中はゆっくり回し（1周約8秒）、やや手前に傾けて上面も見せる（25度）。
        _ang = self.present_angles.get(str(self._target), {})
        _yaw_deg = float(_ang.get("yaw", self.fixed_yaw_deg))
        _tilt_deg = float(_ang.get("tilt", 25.0))
        _spin = (2.0 * np.pi * float(env.data.time) / self.spin_period_sec
                 if self.spin_period_sec > 0.0 else np.radians(_yaw_deg))
        _c, _s = np.cos(_spin), np.sin(_spin)
        ex2 = _c * ex + _s * ey
        ey2 = -_s * ex + _c * ey
        _tilt = np.radians(_tilt_deg)
        _ct, _st = np.cos(_tilt), np.sin(_tilt)
        ex3 = _ct * ex2 + _st * ez
        ez3 = -_st * ex2 + _ct * ez
        q = np.empty(4)
        _mj.mju_mat2Quat(q, np.stack([ex3, ey2, ez3], axis=1).ravel())
        return q

    def _apply_shake(self, env):
        """今振っているおもちゃの位置に、正弦波の揺れを加える（F1-3aの決定的検証と同条件）。

        揺れの軸は片目カメラのローカルx軸（「右」方向）＝_set_anchorの左右振り分けと同じ軸。
        中心（base）は env._rest_pos / _rest_pos2（変更しない）＝揺れは表示上の上乗せだけで、
        毎stepの _apply_tether / _hold_toy2 が置いた位置を、このメソッドが上書きする形になる
        （step()内の呼び出し順：_hold_toy2の直後にupdate()を呼ぶ設計）。
        """
        cid = int(env.model.camera("eye_left").id)
        right = np.array(env.data.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 0]
        # 【F2-9・2026-08-26】バースト化：連続正弦波 → 「振る(2往復)→止まる」の繰り返し。
        #   人間の親のmotionese（乳児向けは静止時間の割合が約3倍 [Tier2・van Schaik
        #   2020]）に合わせる。静止区間は wob=0＝基準位置に置くだけ。
        #   位相はサイクル（burst+pause）内に折り返す。バースト先頭で sin は必ず
        #   0から始まる（位相の不連続で物が跳ばない）。
        phase = self._shake_t % (self.shake_burst_sec + self.shake_pause_sec)
        if phase < self.shake_burst_sec:
            wob = right * self.shake_amp_m * np.sin(2 * np.pi * self.shake_hz * phase)
        else:
            wob = right * 0.0
        # 【F2-9C・2026-08-26】基準位置の決め方が2通りになった。
        #   follow_gaze=False（既定）：従来どおり固定アンカー（_rest_pos）。
        #   follow_gaze=True：太郎のいま見ている方向の先（顔からfollow_dist）へ、
        #     親の手の速さ（follow_speed）の上限つきで物を追従させる＝差し出し。
        if self.follow_gaze:
            base = self._follow_step(env)
        elif self._target == "toy1":
            base = getattr(env, "_rest_pos", None)
        else:
            base = getattr(env, "_rest_pos2", None)
        if base is None:
            return
        # 【10択・2026-08-31】書き込み先をスロット表から引く（toy1/toy2/toy5〜共通）
        _sl = self._slots(env).get(self._target)
        if _sl is None:
            return
        env._place(_sl["qadr"], np.array(base, dtype=float) + wob)
        if self.follow_gaze:
            env.data.qpos[_sl["qadr"] + 3:_sl["qadr"] + 7] = self._present_quat(env, _sl["shape"])
        env.data.qvel[_sl["dadr"]:_sl["dadr"] + 6] = 0.0

    def _hide_other(self, env):
        """名指ししていない方のおもちゃを視界外へ退避させる（solo_presentation時のみ）。

        _apply_shake と同じく「毎stepの _apply_tether / _hold_toy2 が置いた位置を
        上書きする」流儀。SHAKE状態の間だけ呼ばれるので、不応期に入れば
        次のstepから環境側の通常配置（両方見える）へ自動的に戻る。
        退避先は e_toy_env.FAR_AWAY（未使用物体の既存の退避先）＋オフセット
        （toy2無効時の退避位置 FAR_AWAY+[0.5,0,0] と重ねないため）。
        """
        from run.world.toy_env import FAR_AWAY   # 循環import回避のため関数内で読む
        #   （e_toy_env側の `from parent_labeling import ...` と同じ素の名前で読む
        #    ＝E/scripts がsys.path上にある前提。読み込み済みの同一モジュールを指す）
        # 【10択・2026-08-31】的以外の**全スロット**を退避＋透明化する
        _slots = self._slots(env)
        _others = [(k, v) for k, v in _slots.items() if k != self._target]
        # 【F2-9・2026-08-26】退避先を地面の下2mへ。以前の FAR_AWAY+[1,0,0]（地上5m先）
        #   は第三者視点のViewerで画角に映り込み、「2個同時に見える」ように見えた
        #   （ユーザーの目視報告→ヘッドレス録画で再現→geomダンプで特定）。
        #   太郎の視界からは元々消えていたので学習への影響はない（見た目だけの修正）。
        if not hasattr(env, "_toy_alpha_backup"):
            env._toy_alpha_backup = {}
        for _i, (_k, _v) in enumerate(_others):
            env._place(_v["qadr"], FAR_AWAY + np.array([1.0 + 0.5 * _i, 0.0, -2.0]))
            env.data.qvel[_v["dadr"]:_v["dadr"] + 6] = 0.0
            _adr = env.model.body_geomadr[_v["bid"]]
            _num = env.model.body_geomnum[_v["bid"]]
            for _g in range(_adr, _adr + _num):
                if _g not in env._toy_alpha_backup:
                    env._toy_alpha_backup[_g] = float(env.model.geom_rgba[_g, 3])
                env.model.geom_rgba[_g, 3] = 0.0
        # 【V1b・2026-08-31・実測で発覚】地下2mでも、床が視覚的に遮らない場面では
        #   暗い影として見えていた（図_提示の問題.png③）。透明化を併用する。
        #   復元は update() 冒頭で毎step全geomのαを戻してから隠し直す方式
        #   ＝状態を持たないので取りこぼしが無い。

    def _hide_all(self, env):
        """全スロットを視界外へ退避させる（_VANISH専用。「的が無い」状態を作る）。

        【消失発話・2026-09-05】_hide_other は「的以外」を隠す＝的自体は見える形。
        _VANISH中は的も含めて全部隠す必要があるため、_others の絞り込みを外した
        だけの別メソッドにした（_hide_otherの流儀・退避先はそのまま踏襲）。
        """
        from run.world.toy_env import FAR_AWAY
        _slots = self._slots(env)
        if not hasattr(env, "_toy_alpha_backup"):
            env._toy_alpha_backup = {}
        for _i, (_k, _v) in enumerate(_slots.items()):
            env._place(_v["qadr"], FAR_AWAY + np.array([1.0 + 0.5 * _i, 0.0, -2.0]))
            env.data.qvel[_v["dadr"]:_v["dadr"] + 6] = 0.0
            _adr = env.model.body_geomadr[_v["bid"]]
            _num = env.model.body_geomnum[_v["bid"]]
            for _g in range(_adr, _adr + _num):
                if _g not in env._toy_alpha_backup:
                    env._toy_alpha_backup[_g] = float(env.model.geom_rgba[_g, 3])
                env.model.geom_rgba[_g, 3] = 0.0

    def _enter_pick(self, env=None):
        """次の的を選ぶ（_PICK）前に、旧的があれば_VANISHを挟む。

        【消失発話・2026-09-05・仕様_M2】vanish_utteranceが真かつ旧的
        （self._target）があるときだけ_VANISHへ。既定False・初回提示（旧的なし）
        では従来どおり_PICKへ直行（1ビットも変わらない）。
        【M4c・2026-09-06】vanish_wait_notice が真のときだけ、太郎の「気づいた」
        合図(env.unwrapped._taro_noticed_gone_time)を隠す直前にNoneへ初期化し、
        隠した sim 時刻を self._vanish_hide_t に控える（気づき待ちの判定に使う）。
        env が渡されない呼び出し元（過去のテスト等）を壊さないよう env=None を許容。
        """
        if self.vanish_utterance and self._target is not None:
            self._state = self._VANISH
            self._vanish_t = 0.0
            self._vanish_spoken = False
            if self.vanish_wait_notice and env is not None:
                u = getattr(env, "unwrapped", env)
                u._taro_noticed_gone_time = None
                self._vanish_hide_t = float(env.data.time)
        else:
            self._state = self._PICK

    def _single_word(self, tgt):
        """【親の言い直し・2026-09-03】正誤判定用の単独形。utterances[tgt]が
        [[text, weight], ...] のリストなら最短のtext、文字列ならそのまま
        （設計_親の言い直し.md 第2部の規約）。
        """
        u = self.utterances.get(tgt)
        if isinstance(u, (list, tuple)):
            if not u:
                return ""
            return min((str(text) for text, _ in u), key=len)
        return u or ""

    def _pick_text(self, env, tgt):
        """セリフを1つ選ぶ（2026-08-31・二語文）。

        utterances の値が文字列なら従来どおりそのまま。
        [[セリフ, 重み], ...] のリストなら重みで1つ引く（毎回変わる）。
        実測の裏付け：日本の親は「名詞+動詞（助詞省略）」「名詞+だよ」が典型で、
        裸の名詞単独は18.4%（小椋・浜辺2021ほか、
        doc/文献調査/二語文/2026-08-31_日本語の親の語りかけ.md）。
        """
        u = self.utterances.get(tgt)
        if isinstance(u, (list, tuple)):
            total = sum(float(w) for _, w in u)
            r = float(env.np_random.random()) * total
            for text, w in u:
                r -= float(w)
                if r <= 0.0:
                    return text
            return u[-1][0]
        return u

    def _speak(self, env):
        """注視が確認できた瞬間。respond_probで実際に言うかを決め、言えば不応期へ入る。

        5回に1回（shuffle_after）は、言い終えた直後に左右を入れ替える
        （場所の丸暗記を防ぐ仕掛け。前半の設計意図参照）。
        """
        if env.np_random.random() >= self.respond_prob:
            # 応えないと決めた回（既定respond_prob=1.0では起きない）＝黙って次の試行へ。
            self._state = self._PICK
            return None
        text = self._pick_text(env, self._target)
        result = {"text": text, "target": self._target}
        # 【F2-9・2026-08-26】発話バースト：言う瞬間に振りの位相を0へ戻し、
        #   1往復ぶん（1/shake_hz秒）だけ不応期に食い込んで振り続ける。
        #   人間の親は発話の開始と物の動きの開始を0.06秒の精度で重ねる
        #   [Tier2・Gogate 2000]。シミュレーションでは同一step開始＝ずれ0秒。
        #   前セッションで「見たら止めて言う」と設計しかけたが実測は逆
        #   （動かしながら言う）だったため、こちらに従う（設計_F2-9 第1部③）。
        self._shake_t = 0.0
        self._speech_burst_t = 1.0 / self.shake_hz
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
        self._repeats_left = self.repeat_labels
        if self._repeats_left > 0:
            self._silent_until = max(self._silent_until, self.repeat_gap_sec)
        return result

    def _speak_repeat(self, env):
        """【連呼】注視が続いているときの言い直し。言い回しは混合から引き直す。"""
        text = self._pick_text(env, self._target)
        self._shake_t = 0.0
        self._speech_burst_t = 1.0 / self.shake_hz
        self.n_utterances += 1
        self.n_repeats += 1
        self._repeats_left -= 1
        self._silent_t = 0.0
        self._silent_until = (self.repeat_gap_sec if self._repeats_left > 0
                              else self.refractory_sec)
        self._state = self._REFRACTORY
        self._repeat_attempt = False
        return {"text": text, "target": self._target, "cause": "repeat"}


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
                #   ビルド時点（scene_io.build()）ではtaro（run/taro_setup.py側）が
                #   まだ存在せずhearing設定を見られないため、taroが配線済みに
                #   なる最初の使用時（＝ここ）で止める。
                # 【2026-08-21修正】env.taro は posture_reflex/righting_reflex の
                #   どちらかが有効なときにしか配線されない（run/taro_setup.py:272,307）。
                #   語彙テストは姿勢反射なしで走るため、taroが無い＝hearing無効と
                #   誤診してassertで落ちていた（F1-4hパイロット第2段で実際に発生）。
                #   taroが取れる場合だけ検査し、取れない場合は素通しする
                #   （発話はParentLabelingと同じ info 経路で耳へ届くので、
                #   hearing側の有効確認は実験ログの発話イベントで行う）。
                taro = getattr(env, "taro", None)
                if taro is not None:
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
