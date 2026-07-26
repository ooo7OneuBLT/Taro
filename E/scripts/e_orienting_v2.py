"""視線誘導反射（新版）＝設計図（`E/docs/視線誘導反射_設計図.md`）に基づく実装。

【現行 e_orienting.py からの変更点】
  ステップ1（動き検出）    ： 単フレーム差分 → 複数フレームスケール比較
  ステップ2（中心バイアス）： 6マス分割 → 画素ごとの中心重み
  ステップ3（方向の決定）  ： 全マス合計 → 側方抑制で1箇所を選び、その周辺で重心
  ステップ4（出力）        ： 毎tick連続加算 → 階段状サッケード（未実装）
  ステップ5（IOR）         ： なし → 一度向いた場所を数百ms 抑制（未実装）
  保留                    ： egomotion 補正（残差法は取り除く。遠心性コピーが要る
                            本物の補正は保留、簡易近似はしない）

【ステップ3の設計変更・2026-07-26】
当初は「閾値以上の画素で全体の重心」だったが、これは**離れた2箇所を平均してしまい、
何もない中間を指す**という欠陥があった（人工画像テストで発覚）。
文献（調査B）が言っていたのは population vector 単独ではなく
**側方相互作用によるソフトな winner-take-all** だった：
  ・上丘に側方抑制がある＝実測された事実（Munoz & Istvan 1998、サル上丘で直接記録）
  ・その競合が1つの山に収束する＝数理モデル（Kopecz & Schöner 1995、
    実際のサッケードデータに合う工学的モデル）
  ・⚠️「必ず1つが勝つ」わけではない。標的が近いと人間も中間へサッケードする
    （global effect＝大域効果、実際に観察される現象）。
    メキシカンハット型のカーネルは距離に応じてこの両方を自然に出す。

⚠️実装中。ステップ4・5は未実装。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# 関節への指令を、身体の駆動方式（筋肉2本／モーター1つ）に合った形で書き込む共通の写像。
# 反射がこれを飛ばして直接書くと、筋肉モデルでは負の指令が消えて片方向にしか動けなくなる。
import os as _os, sys as _sys
_CORE_BRAIN = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, _os.pardir,
    "taro_core", "src", "brain"))
if _CORE_BRAIN not in _sys.path:
    _sys.path.insert(0, _CORE_BRAIN)
from spinal_cord.cpg import write_joint_command as _write_joint_command

# ---- ステップ1（動き検出）------------------------------------------------
# frame index の 1・5・20 前と現フレームを比べる（10Hzなら 0.1・0.5・2秒 前に対応）
# [Tier3・ARBITRARY] 人間はフレーム記憶でなく連続的な時間フィルタ。効果のみ近似。
TIME_SCALES = (1, 5, 20)

# ---- ステップ2（中心視野バイアス）----------------------------------------
# 画像中心ほど重みが大きいガウス窓。上丘の中心視野マグニフィケーション
# （対数極座標マッピング、Ottes et al. 1986）の粗い近似。
# [Tier3・ARBITRARY] sigma は「画像端で重み ≈0.3」になるよう選んだ暫定値。
CENTER_BIAS_SIGMA_FRAC = 0.32   # 画像サイズに対する比

# ---- ステップ3（側方抑制＋重心）------------------------------------------
# メキシカンハット：近く（sigma_exc）は助け合い、遠く（sigma_inh）は邪魔し合う。
# [Tier3・ARBITRARY] 機構の存在は Munoz & Istvan 1998 の実測だが、
# 具体的な sigma・重み・反復回数は文献に値がなく、目視で調整する暫定値。
LI_SIGMA_EXC_FRAC = 0.025   # 局所興奮の広がり（画像サイズ比）
LI_SIGMA_INH_FRAC = 0.12    # 近距離抑制の広がり（画像サイズ比）
LI_W_EXC = 1.0
LI_W_INH = 0.9
LI_RATE = 0.5               # 1反復あたりの更新率
LI_N_ITER = 12              # 反復回数

# 【全体抑制・2026-07-26 追加】視野全体に届く抑制。
# 上記のガウス型抑制は届く範囲が約 46画素（sigma 15.4 の3倍）しかなく、
# 30度（64画素）以上離れた2標的が**互いを抑制できず、ただ並んで平均される**
# ことが実測で判明した。
# 【根拠】Munoz & Istvan (1998) J Neurophysiol 79:1193 が上丘で実測したのは
#   ・固視ニューロンとサッケードニューロンの相互抑制
#   ・★左右の上丘の間の抑制性結合（＝視野の反対側どうしが抑制し合う）
#   ＝抑制は局所だけでなく**視野全体に届く**。
# ダイナミック神経場モデルでも「1つの山だけが残る」ことを保証するために
# 全体抑制（global inhibition）を置くのが標準形。
# ⚠️[ARBITRARY] 重みの具体値は文献になく、実測で選ぶ。
LI_W_GLOBAL = 1.2

# 重心を取るときの閾値（勝った山の裾を切る）。最大値に対する比。
CENTROID_THRESH_FRAC = 0.35

# ---- ステップ4（階段状サッケード）----------------------------------------
# 【なぜ間欠出力か】新生児の平滑追従（smooth pursuit）は未成熟で、
# 追跡時間の15%未満・19度/秒で頭打ち（Kremenitzer et al. 1979, PubMed 487885）。
# 実際の定位は跳躍運動（サッケード）であり、しかも
# **第一サッケードは著しく低振幅で目標に届かず、同振幅の追加サッケードが
# 階段状に連続する**（Aslin & Salapatek 1975）。
# ＝毎tick連続的に目を動かす旧実装は smooth pursuit 型で、新生児には合わない。
#
# 【文献で決まっている値】サッケード潜時は多くが 500ms 未満（Aslin & Salapatek 1975）
# ⚠️[ARBITRARY] 以下の具体値は文献に直接の記載がなく、目視で調整する暫定値。
SACCADE_LATENCY = 0.20      # 次のサッケードまでの最小間隔[秒]（<500ms の範囲内）
SACCADE_DURATION = 0.05     # 1発のサッケードが続く時間[秒]
SACCADE_MIN_STRENGTH = 0.02 # これ未満の動きでは撃たない
# 首と目のゲイン。低振幅（hypometric）にするため小さく取り、
# 届かなければ次のサッケードで詰める＝階段状になる。
# ⚠️[ARBITRARY] 旧実装 e_orienting.py から引き継いだ暫定値。
NECK_GAIN = 0.3
EYE_GAIN = 0.15
# apply() 1回あたりの経過時間[秒]。環境の dt に合わせて上書きする。
DEFAULT_DT = 0.01

# ---- 神経ノイズ（対称性を破って決定を生む）--------------------------------
# 【なぜ入れるか・2026-07-26】完全に対称な2標的では、側方抑制だけでは
# 引き分けたまま決着しない（数学的に対称解が安定）。実装テストで
# 45度・55度分離の対称2標的が中央を指し続けることを確認した。
#
# 【学術的根拠】上丘ニューロンの試行間変動が、実際の選択結果と直接結びつく：
#   Kim B, Basso MA (2010) "A Probabilistic Strategy for Understanding Action
#   Selection" J Neurosci 30(6):2340（PMC2841973）
#     ・上丘ニューロンの Fano factor = 1.44（同じ刺激でも発火がばらつく）
#     ・4ニューロンの活動から選択を予測：ベイズ推定 84.76% >
#       勝者総取り 71.11% > 集団ベクトル平均 55.71〜69.47%
#     ＝ばらつきが選択結果を予測する＝ノイズが決定に関与している直接証拠
#   加えて、決定モデル一般（drift-diffusion / race model）でも、
#   対称条件ではノイズ項がないと理論上決着しないことが知られている。
#
# ⚠️[ARBITRARY] ノイズの大きさに文献の直接の推奨値は無い（調査で確認）。
#   Fano factor 1.44 は発火のばらつきの指標であって、
#   この実装の活動量スケールへの変換式は存在しない。
#   「単一標的の精度を崩さず、対称2標的で決着する最小値」を実測で選ぶ。
#
# ⚠️人間の対称2択は純粋な 50:50 ランダムではない（中心窩寄り選好・個体差・
#   コスト依存の系統的偏りが報告されている：Van Heusen 2023、PES研究）。
#   ここではノイズのみで、偏り項は入れない＝人間の近似。
LI_NOISE = 0.05


class OrientingReflexV2:
    """視線誘導反射・新版。段階的に組み立てる。"""

    def __init__(self, model, time_scales=TIME_SCALES, noise=LI_NOISE, seed=None,
                 dt=DEFAULT_DT):
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        # ステップ4（階段状サッケード）の状態
        self._t = 0.0                # apply() が刻む内部時刻[秒]
        self._last_saccade_t = -1e9  # 前回サッケードを撃った時刻
        self._sacc_remaining = 0.0   # 今のサッケードの残り時間[秒]
        self._sacc_h = 0.0           # 今のサッケードの方向（撃った瞬間に固定）
        self._sacc_v = 0.0
        self.n_saccades = 0          # 撃った回数（テスト・観察用）
        self.time_scales = tuple(time_scales)
        self._max_scale = max(self.time_scales)
        self._frame_buffer = []          # 直近フレームのリング（最大 _max_scale+1 枚）
        self.motion_map = None           # ステップ1の出力
        self.biased_map = None           # ステップ2の出力
        self.competed_map = None         # ステップ3の競合後の活動
        self._center_weight = None       # ステップ2の重み（画像サイズが決まってから作る）
        self.h_dir = 0.0
        self.v_dir = 0.0
        self.strength = 0.0              # 反応の強さ（0〜1相当）
        self.n_actuator = int(model.nu)
        # 目・首アクチュエータのインデックス（現行 e_orienting.py と同じ）
        self.neck_idx = {}
        self.eye_idx = {"h": [], "v": []}
        for i in range(model.nu):
            name = model.actuator(i).name
            if name == "act:head_swivel":
                self.neck_idx["h"] = i
            elif name == "act:head_tilt":
                self.neck_idx["v"] = i
            elif "eye" in name and "horizontal" in name:
                self.eye_idx["h"].append(i)
            elif "eye" in name and "vertical" in name:
                self.eye_idx["v"].append(i)

    def reset(self):
        self._frame_buffer = []
        self.motion_map = None
        self.biased_map = None
        self.competed_map = None
        self.h_dir = 0.0
        self.v_dir = 0.0
        self.strength = 0.0
        self._t = 0.0
        self._last_saccade_t = -1e9
        self._sacc_remaining = 0.0
        self._sacc_h = 0.0
        self._sacc_v = 0.0
        self.n_saccades = 0

    # ------------------------------------------------------------
    # 公開インターフェース
    # ------------------------------------------------------------
    def update(self, eye_image):
        """新しい画像で内部状態を更新する。両眼平均を想定するが単眼画像でも動く。"""
        motion = self._detect_motion(eye_image)          # ステップ1
        self.motion_map = motion
        biased = self._apply_center_bias(motion)         # ステップ2
        self.biased_map = biased
        h, v, s = self._select_and_centroid(biased)      # ステップ3
        self.h_dir, self.v_dir, self.strength = h, v, s
        # ステップ4（階段状サッケード）・5（IOR）は未実装

    def apply(self, action, dt=None):
        """action に階段状サッケードを加算して返す（ステップ4）。

        毎tick連続的に加算するのではなく、
          ・前回から SACCADE_LATENCY 経過し、かつ動きが十分なら1発撃つ
          ・撃ったサッケードは SACCADE_DURATION のあいだだけ出力される
          ・その間は方向を固定する（撃った瞬間の h_dir/v_dir）
          ・目標に届かなければ次の潜時のあとにまた撃つ → 階段状
        """
        self._t += self.dt if dt is None else float(dt)

        if self._sacc_remaining <= 0.0:
            # サッケードを撃つか判断する
            ready = (self._t - self._last_saccade_t) >= SACCADE_LATENCY
            if ready and self.strength >= SACCADE_MIN_STRENGTH \
                    and (abs(self.h_dir) > 1e-6 or abs(self.v_dir) > 1e-6):
                self._sacc_h = self.h_dir
                self._sacc_v = self.v_dir
                self._sacc_remaining = SACCADE_DURATION
                self._last_saccade_t = self._t
                self.n_saccades += 1
            else:
                return action     # サッケード中でなければ何も足さない

        # サッケード実行中：固定した方向を出力する
        self._sacc_remaining -= (self.dt if dt is None else float(dt))
        out = np.array(action, dtype=float).copy()
        h, v = self._sacc_h, self._sacc_v
        # ★2026-07-26：ここで `out[i] = clip(out[i] + gain*h, -1, 1)` と直接書いていたのが誤り。
        #   MuscleModel では1関節が2本の筋（前半＝負方向筋・後半＝正方向筋）で駆動され、
        #   各要素は [0, 1] に切り捨てられる。負の指令は消えるので、**首も目も片方向にしか
        #   動けなかった**（目標が反対側にあると永久に追えない）。VOR で同じ誤りが実測で
        #   確認され（眼が可動域の下限に張り付いて戻らない）、こちらも同型と判明した。
        #   共通の写像 write_joint_command を通す。眼球は相反神経支配なので共収縮は0。
        for key, gain in (("h", NECK_GAIN), ("v", NECK_GAIN)):
            if key in self.neck_idx:
                d = h if key == "h" else v
                _write_joint_command(out, self.neck_idx[key], gain * d,
                                     self.n_actuator, co_activation=0.0, additive=True)
        for i in self.eye_idx["h"]:
            _write_joint_command(out, i, EYE_GAIN * h, self.n_actuator,
                                 co_activation=0.0, additive=True)
        for i in self.eye_idx["v"]:
            _write_joint_command(out, i, EYE_GAIN * v, self.n_actuator,
                                 co_activation=0.0, additive=True)
        return out

    # ------------------------------------------------------------
    # ステップ1：複数フレーム比較による動き検出
    # ------------------------------------------------------------
    def _detect_motion(self, image):
        """現フレームと 1・5・20 前のフレームとの差を混ぜて動きマップを返す。

        フレームが揃わないうちは、揃っているスケールだけ使う（穴埋めしない）。
        RGB画像はチャネル平均で単一マップにする。
        ⚠️ON（明転）とOFF（暗転）を区別せず絶対値を取る。文献調査（2026-07-26）で
        「ON優位」を支持する定量値は無く、V1はむしろOFF優位（Jansen 2019, 65:35）、
        行動レベルの非対称もパラダイム依存でバラバラと判明したため、
        **対称（1:1）が根拠のない前提を最小にする**という判断。
        """
        cur = np.asarray(image, dtype=np.float32)
        cur_gray = cur.mean(axis=-1) if cur.ndim == 3 else cur

        self._frame_buffer.append(cur_gray)
        while len(self._frame_buffer) > self._max_scale + 1:
            self._frame_buffer.pop(0)

        motion = np.zeros_like(cur_gray, dtype=np.float32)
        used = 0
        for scale in self.time_scales:
            if len(self._frame_buffer) > scale:
                past = self._frame_buffer[-1 - scale]
                motion += np.abs(cur_gray - past)
                used += 1
        if used == 0:
            return np.zeros_like(cur_gray)
        return motion / used

    # ------------------------------------------------------------
    # ステップ2：中心視野バイアス
    # ------------------------------------------------------------
    def _apply_center_bias(self, motion):
        """画像中心ほど重みが大きいガウス窓を掛ける。"""
        if self._center_weight is None or self._center_weight.shape != motion.shape:
            h, w = motion.shape
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            sigma = CENTER_BIAS_SIGMA_FRAC * max(h, w)
            ys = np.arange(h)[:, None] - cy
            xs = np.arange(w)[None, :] - cx
            self._center_weight = np.exp(-(ys ** 2 + xs ** 2) / (2 * sigma ** 2)).astype(np.float32)
        return motion * self._center_weight

    # ------------------------------------------------------------
    # ステップ3：側方抑制で1箇所を選び、その周辺で重心を取る
    # ------------------------------------------------------------
    def _select_and_centroid(self, activity):
        """メキシカンハット型の側方相互作用を数回まわして競合させ、
        残った活動の重心を (h_dir, v_dir, strength) として返す。

        h_dir: 右が正、左が負（[-1, 1] 相当）
        v_dir: 上が正、下が負（画像座標の y は下向きなので符号を反転）
        strength: 反応の強さ（動きが無ければ 0）
        """
        if activity.max() < 1e-6:
            self.competed_map = np.zeros_like(activity)
            return 0.0, 0.0, 0.0

        h, w = activity.shape
        sigma_exc = LI_SIGMA_EXC_FRAC * max(h, w)
        sigma_inh = LI_SIGMA_INH_FRAC * max(h, w)

        inp = activity / activity.max()      # 入力を [0,1] に正規化
        u = inp.copy()
        # ノイズは確率微分方程式の慣例に従い sqrt(dt) でスケールする
        noise_scale = self.noise * np.sqrt(LI_RATE)
        for _ in range(LI_N_ITER):
            f = np.clip(u, 0, None)          # 発火率（負は出力しない）
            exc = gaussian_filter(f, sigma_exc) * LI_W_EXC
            inh = gaussian_filter(f, sigma_inh) * LI_W_INH
            inh_global = LI_W_GLOBAL * float(f.mean())   # 視野全体に届く抑制
            # du/dt = -u + input + 近傍興奮 - 近距離抑制 - 全体抑制 + ノイズ
            u = u + LI_RATE * (-u + inp + exc - inh - inh_global)
            if noise_scale > 0:
                u = u + noise_scale * self.rng.standard_normal(u.shape)
        u = np.clip(u, 0, None)
        self.competed_map = u

        if u.max() < 1e-6:
            return 0.0, 0.0, 0.0

        # 勝った山の裾を切って重心を取る
        thresh = u.max() * CENTROID_THRESH_FRAC
        mask = u > thresh
        wsum = u[mask].sum()
        if wsum < 1e-9:
            return 0.0, 0.0, 0.0
        ys, xs = np.nonzero(mask)
        cy_map = float((ys * u[mask]).sum() / wsum)
        cx_map = float((xs * u[mask]).sum() / wsum)

        # 画像中心を原点に、[-1, 1] へ正規化
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        h_dir = (cx_map - cx) / cx
        v_dir = -(cy_map - cy) / cy          # 画像の y は下向きなので反転
        strength = float(activity.max())
        return float(h_dir), float(v_dir), strength
