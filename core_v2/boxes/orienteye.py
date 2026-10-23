"""見えた方へ目を向ける反射（1段目 反射、上丘の役）。

段階2・工程3の仮部品（仕様書 §3）：合わせた地図（prio）の最大のマスが、今の注視点
（地図は網膜の座標なので真ん中 13.5,13.5）から MIN_DIST_CELLS マス以上離れていて、前回の
発射から REFRACTORY_TICKS 回（1.2秒）以上たっていれば、そのマスの方向
((x−13.5)/13.5, (13.5−y)/13.5)（[-1,1]、右・上が正）を目の出口へ渡す。撃たない回はゼロ。

抑える強さ（門番から）が 0.5 を超えるときは、目立ちだけの山には撃たない。
**仕様書からの変更（2026-10-15）**：仕様書は「探し物らしさ<0.2 の山へは撃たない」だが、
この係に届くのは合わせた地図だけで、探し物らしさは別には届かない。合わせた地図の値 m と
w=0.9×抑える強さ から、探し物らしさの下限 (m−(1−w))/w（目立ちが最大の1だった場合）を出し、
それが 0.2 未満なら撃たない。目立ちだけの山は m ≤ 1−w なので必ず止まる。

**段階3（2026-10-15〜16）：復帰抑制（一度見た所・物はしばらく見ない）を足した。**生まれつきの部品
（新生児からある。Valenza 1994 ほか、旧太郎 `doc/文献調査/2026-08-19_乳児のIORと馴化.md`）
なので手で決め、学習しない。
* 「もう見た」の印を、場所の点（網膜の座標、行・列）と強さの組で持つ（self.marks）。強さは毎回
  exp(−0.1/IOR_DECAY_S) 倍に薄れる。撃ったら全部の印を目が動く分だけ逆向きにずらし（目を動かす
  指令の写しを使う）、新しい注視点＝真ん中に強さ1の印を置く
* **印は物に付いて動く（2026-10-16、ユーザー決定）**：毎回、目の入口から届いた物ごとの山の
  中心を見て、印から TRACK_CELLS マス以内に物があれば、印をその物の中心へ移す。物が無ければ
  その場に残る（＝場所の印）。最初の版は網膜の場所にだけ印を付けたため、横に動き続ける妨害物が
  自分の印の外へ逃げ、太郎は妨害物を追いかけ直し続けた（段階3①のボール右の場面 1/10）
  - 物を見分けるのは見た目ではなく「つながって動いていること」（場所と時間の連続）。乳児は
    物を個体として追うとき、見た目より場所と時間の連続を使う（Xu & Carey 1996 系）。妨害物は
    点滅して見た目が変わるので、見た目で照合すると取り違える
  - 乳児で「動く物に印が付いて動く」ことを直接確かめた文献は見つけていない（成人は Tipper 1991。
    乳児は場所の復帰抑制が3〜6か月で整い、物の新しさを好む傾向は6か月で出る、Harman ほか 1994）。
    逸脱リスト行き［Tier3・モデル上の選択］
* **目の動いた分は、目の入口の「目の動きの地図」から読む（2026-10-16 作り直し）**：目の入口は毎回、
  前の画像から今の画像へ画面全体がどれだけずれたか（位相相関。MT の役）を、真ん中からずらした
  山（荷札ゼロの組、物の列の最後）として送ってくる。それを decode_ego で戻し、全部の印に足す。
  目を動かす指令の写しではなく実際に目が動いた分なので、跳びが指令より小さくても印は物に残る。
  目だけのテストの問題集で、このずれの誤差は中央値0.5画素・最大でも10画素未満だった（60画素跳んでも）。
  前の版（物の中心の組を見比べる estimate_shift と、確かめ待ちの _pending）は、目の入口が3回に1回しか
  切り出さず、間の回は動いていない位置を流していたため、跳んだ直後に印が取り残された（①の1試行、
  2026-10-16）。目の動きの地図が届かないとき（環境なしのテスト）だけ、撃った回に指令の分でずらす
  ［Tier3・モデル上の選択］
* 印を新しく置くのは「撃った先のマス」（撃った回の座標）。その後、上のずれで目と一緒に動く
* 印が物に付いていける1回の距離 TRACK_CELLS は 1.5 マス（目の動きを別に引いたので、物そのものの
  動きだけ追えばよい）
* **1つの印で、見ている間は慣れ・離れたら復帰抑制（2026-10-16 作り直し、案A）**：撃った先に強さ1の
  印を置くのをやめた。その形だと、着いた物が見ている最中から抑えられ、太郎は1.2秒（撃てない時間）
  ごとに次の物へ移り、1つの物を3秒見続けることが一度も無かった（①の20試行で最長2.8秒・中央値1.2秒）。
  そのため親の名づけ（3秒見たら名前を言う）が1回も起きなかった。今は：
  - 今見ている物（目の入口の物のうち、中心が真ん中から FIX_CELLS マス以内でいちばん近いもの）に印を
    付け、見ている間は強さが 1−exp(−見ている時間/HAB_TAU) で育つ（慣れ・馴化）。見続けるほど
    目立たなくなり、ほかの物が上回ったら移る
  - 目が離れたら、その印は物に付いて動きながら IOR_DECAY_S で薄れる（復帰抑制）。人間の復帰抑制も、
    注意が離れたあとの場所・物にかかる（Posner の形。記憶ベース）
  - 慣れ（馴化）は本来は大脳の働きで、上丘の係に入れるのはモデル上の選択［Tier3］。乳児が1つの物を
    見つめる時間は数秒（記憶ベース・要確認）
  - 目の入口から物が届かないとき（環境なしのテスト）は、真ん中の「場所」に印を付ける
  - **脱馴化（2026-10-16）**：物の位置の値（合わせた地図、印を引く前）が、その物のふだん（指数移動平均
    μ・ふだんの揺れ幅 σ）より max(DISHAB_JUMP, DISHAB_Z×σ) 以上急に上がったら、その物の印（慣れも
    復帰抑制も）を DISHAB_KEEP 倍に下げる。光って動く妨害物は目立ちがいつも大きく揺れるので、揺れ幅で
    割らないと印が消え続ける。乳児は、慣れた物でも動きや出方が変わると、また見るようになる（脱馴化。
    記憶ベース・要確認）［Tier3・値は仮置き］
  - **見続ける力（2026-10-16、案D）**：慣れだけでは、撃てない時間（1.2秒）が明けた瞬間に、より目立つ
    動く妨害物（1.0）が見ている物（ボール約0.76−慣れ0.33）を上回り、見続ける時間は中央値1.2秒のまま
    だった。そこで、ほかの山が「今見ている物の値＋FIX_HOLD×(1−慣れ)×(1−GOAL_RELIEF×抑える強さ)」を
    上回ったときだけ撃つ（撃たない回は "holding"）。最初は離れにくく、慣れが育つほど離れやすい。
    探している間は弱める（探し物へ跳ぶのを邪魔しない）。上丘の真ん中（rostral SC）の注視ニューロンが
    見ている間の跳びを抑える働き、乳児の目が離れにくい傾向（sticky fixation、月齢が低いほど強い）に
    あたる（どちらも記憶ベース・要確認）［Tier3・FIX_HOLD は見続けた時間で決める］
* どこを見るか決める前に、合わせた地図から「印の地図」（印ごとに高さ＝強さ・σ=IOR_SIGMA_CELLS の
  山、重なりは大きい方）× IOR_GAIN ×(1 − GOAL_RELIEF×抑える強さ) を引く（負は0）
* **目的を持って探している間は、さっき見た物への復帰抑制を弱める（2026-10-16）**：上丘の反射は、
  前頭眼野などの目的で上書きされる。抑える強さ（門番 gate3、0〜1）が高いほど印を弱く引く。
  印を8秒持たせたため、言われる前に見たボールに印が約0.5残り、「ボールはどこ？」で探し物へ向かう
  跳びが門番の足切りに落ちていた（②ボール 17/20、ボール左 seed9 は t51〜t68 撃てず）。
  乳児・成人の文献は未確認［Tier3・GOAL_RELIEF=0.7 は仮置き］
**段階4（2026-10-16）：印は「物の名簿」（objfile）へ移した。**上の印・物に付いて動く・目の動きの地図・
慣れ・離れたら復帰抑制・脱馴化は、名簿の行ごとの「強さ」として名簿が持つ（値も同じ）。この係は名簿から
行ごとの印の山と「今見ている行」（列の最後、荷札ゼロ。高さ＝慣れ）を受け取り、
* 印の地図（行の山の大きい方）× IOR_GAIN ×(1 − GOAL_RELIEF×抑える強さ) を合わせた地図から引く
* 見続ける力：ほかの山が「今見ている行の値＋FIX_HOLD×(1−慣れ)×(1−GOAL_RELIEF×抑える強さ)」を上回るときだけ撃つ
だけを行う。撃った指令は目の出口と名簿の両方へ渡す（名簿は撃った先の物へ注意を移す）。
use_ior=False で段階2と1ビットも変わらない（名簿からの入力を使わない）。
"""

import math

import torch

from ..formats import GRID, DTYPE
from .base import Box

MIN_DIST_CELLS = 2.0
REFRACTORY_TICKS = 12          # 1.2秒（脳の1回＝0.1秒）
INHIBIT_ON = 0.5
MIN_SEEK = 0.2
W_GOAL = 0.9
CENTER = (GRID - 1) / 2.0      # 13.5
TICK_S = 0.1

IOR_GAIN = 1.0           # 旧太郎と同じ［仮置き］
GOAL_RELIEF = 0.7        # 目的を持って探している間に印を弱める割合［Tier3・仮置き］
FIX_HOLD = 1.0           # 見続ける力（慣れ0のときの上乗せ）［Tier3・仮置き］


def read_objfile(lst):
    """名簿からの列を (印の地図, 今見ている行の (行, 列, 慣れ) か None) に分ける。"""
    ior = torch.zeros(GRID, GRID, dtype=DTYPE)
    att = None
    if not lst:
        return ior, att
    for tag, m in lst[:-1]:
        ior = torch.maximum(ior, m)
    _, am = lst[-1]
    hab = float(am.max())
    if hab > 0.0:
        r, c = divmod(int(torch.argmax(am)), GRID)
        att = (r, c, hab if hab > 1e-3 + 1e-6 else 0.0)
    return ior, att


class Orienteye(Box):
    BOX_ID = "orienteye"

    def __init__(self, use_ior: bool = True) -> None:
        super().__init__()
        self.use_ior = bool(use_ior)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_fire = -10 ** 9
        self.last_reason = ""     # 記録用
        self.ior = torch.zeros(GRID, GRID, dtype=DTYPE)
        self.last_ior_at_peak = 0.0   # 記録用：選んだマス（引く前の最大）に乗っていた印

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        m = inbox.get("prio")
        inhibit = float(inbox.get("gate3", 0.0))
        att = None
        relief = 1.0 - GOAL_RELIEF * min(max(inhibit, 0.0), 1.0)
        if self.use_ior:
            self.ior, att = read_objfile(inbox.get("objfile"))
            if m is not None:
                raw_peak = int(torch.argmax(m))
                self.last_ior_at_peak = float(self.ior.reshape(-1)[raw_peak])
                m = (m - IOR_GAIN * relief * self.ior).clamp(min=0.0)
        if m is None or float(m.max()) <= 0.0:
            self.last_reason = "empty"
            return out
        r, c = divmod(int(torch.argmax(m)), GRID)
        peak = float(m[r, c])
        if math.hypot(c - CENTER, r - CENTER) < MIN_DIST_CELLS:
            self.last_reason = "centered"
            return out
        if self.t - self.last_fire < REFRACTORY_TICKS:
            self.last_reason = "refractory"
            return out
        if att is not None:
            hr, hc, hab = att
            hold = float(m[hr, hc]) + FIX_HOLD * (1.0 - hab) * relief
            if peak <= hold:
                self.last_reason = "holding"
                return out
        if inhibit > INHIBIT_ON:
            w = W_GOAL * inhibit
            if (peak - (1.0 - w)) / w < MIN_SEEK:
                self.last_reason = "inhibited"
                return out
        self.last_fire = self.t
        self.last_reason = "fire"
        cmd = torch.tensor([(c - CENTER) / CENTER, (CENTER - r) / CENTER], dtype=DTYPE)
        out["eyeout"] = cmd
        if "objfile" in out and self.use_ior:
            out["objfile"] = cmd.clone()
        if "eyein" in out:              # 段階9：指令の写しを目の入口へ（目が動いた分を先に知る）
            out["eyein"] = cmd.clone()
        return out
