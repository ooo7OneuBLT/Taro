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
use_ior=False で段階2と1ビットも変わらない。
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

# 復帰抑制（段階3・仕様書 付録A）
IOR_DECAY_S = 8.0        # 旧太郎 orienting.py:703 は3秒（成人）。①の実測（2026-10-16）で3秒では
                         # 1.2秒後の妨害物（目立ち1.0）に残る印0.67が、コップ（0.19）より上に残すので、
                         # 6.4秒以上が要る。8秒にした［Tier3・仮置き］
IOR_SIGMA_CELLS = 2.0    # 旧太郎 spatial_priority_map.py の「注意の半径の半分」≒2.3マス［仮置き］
IOR_GAIN = 1.0           # 旧太郎と同じ［仮置き］
GOAL_RELIEF = 0.7        # 目的を持って探している間に印を弱める割合［Tier3・仮置き］
HAB_TAU = 6.0            # 見ている間に印が育つ時定数（秒）［Tier3・仮置き、①の見続けた時間で決める］
FIX_HOLD = 1.0           # 見続ける力（慣れ0のときの上乗せ）［Tier3・仮置き］
LEAVE_IOR = 1.0          # 目を離した物の印の強さ（見ていた間の慣れがこれより弱ければ引き上げる）［Tier3・仮置き］
FIX_CELLS = 3.0          # 中心が真ん中からこれ以内の物を「今見ている物」とする（約3マス≒8度）［仮置き］
TRACK_CELLS = 1.5        # 印が物に付いていける1回の距離。妨害物は横に約12度/秒＝約0.6マス/回［仮置き］
MERGE_CELLS = 1.0        # これより近い印は1つにまとめる（強い方を残す）
MAX_MARKS = 8
MARK_MIN = 0.02          # これより弱い印は捨てる

_YS, _XS = torch.meshgrid(torch.arange(GRID, dtype=DTYPE), torch.arange(GRID, dtype=DTYPE),
                          indexing="ij")


def bump_at(r: float, c: float, sigma: float = IOR_SIGMA_CELLS) -> torch.Tensor:
    return torch.exp(-((_XS - c) ** 2 + (_YS - r) ** 2) / (2.0 * sigma ** 2))


def shift_map(m: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """地図を下へ dr・右へ dc マスずらす。はみ出た所は捨て、空いた所はゼロ。"""
    out = torch.zeros_like(m)
    r0, r1 = max(dr, 0), GRID + min(dr, 0)
    c0, c1 = max(dc, 0), GRID + min(dc, 0)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def split_ego(objs):
    """目の入口からの組の列を、物の組と「目の動きの地図」（荷札ゼロ、列の最後）に分ける。"""
    if not objs:
        return [], None
    tag, m = objs[-1]
    if float(torch.linalg.vector_norm(tag)) == 0.0:
        return list(objs[:-1]), m
    return list(objs), None


def object_centers(objs) -> list[tuple[float, float]]:
    """物ごとの (荷札, 地図) から、山の中心（行・列、小数）を出す。山の 0.5 以上の重心。
    荷札ゼロの組（目の動きの地図）は物ではないので除く。"""
    out = []
    for tag, m in objs or []:
        if float(torch.linalg.vector_norm(tag)) == 0.0 or float(m.max()) <= 0.0:
            continue
        w = (m >= 0.5 * float(m.max())).to(DTYPE) * m
        tot = float(w.sum())
        out.append((float((w * _YS).sum()) / tot, float((w * _XS).sum()) / tot))
    return out


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
        self.marks: list[list[float]] = []    # [行, 列, 強さ]（網膜の座標）
        self.ior = torch.zeros(GRID, GRID, dtype=DTYPE)
        self.last_ior_at_peak = 0.0   # 記録用：選んだマス（引く前の最大）に乗っていた印
        self._ego_seen = False        # 目の入口から目の動きの地図が届いているか
        self.last_held = False        # 記録用：今見ている物（場所）に慣れの印を育てたか
        self._attended = None         # 今注意を置いている印（次に撃つまで）
        self._held_now = None         # この回に慣れを育てた印（物が届かないときは真ん中の場所の印）
        self._pick_wait = 0           # 撃ったあと、着いた物を選ぶまで待つ回数
        self._aim = None              # 撃った先（網膜の座標。目の動きの地図で印と一緒にずらす）
        self.last_shift = (0.0, 0.0)  # 記録用：その回に印をずらした分（マス、行・列）

    def _render(self) -> None:
        m = torch.zeros(GRID, GRID, dtype=DTYPE)
        for r, c, a in self.marks:
            m = torch.maximum(m, a * bump_at(r, c))
        self.ior = m

    def _merge(self) -> None:
        kept: list[list[float]] = []
        for mk in sorted(self.marks, key=lambda x: -x[2]):
            if mk[2] < MARK_MIN and mk is not self._attended and mk is not self._held_now:
                # 見始めたばかりの慣れの印は弱くても残す
                continue
            if any(math.hypot(mk[0] - k[0], mk[1] - k[1]) < MERGE_CELLS for k in kept):
                continue
            kept.append(mk)
        self.marks = kept[:MAX_MARKS]

    def _shift_marks(self, dr: float, dc: float) -> None:
        lim = 3 * IOR_SIGMA_CELLS
        for mk in self.marks:
            mk[0] += dr
            mk[1] += dc
        if self._aim is not None:
            self._aim = [self._aim[0] + dr, self._aim[1] + dc]
        self.marks = [mk for mk in self.marks
                      if -lim <= mk[0] <= GRID - 1 + lim and -lim <= mk[1] <= GRID - 1 + lim]
        self.last_shift = (dr, dc)

    def _track(self, objs) -> None:
        """目の動きの地図の分だけ印をずらし、印から TRACK_CELLS マス以内の物の中心へ移す。
        そのあと、今見ている物の印を慣れで育て、ほかの印は復帰抑制として薄れさせる。"""
        self.last_shift = (0.0, 0.0)
        centers = []
        if objs is not None:
            from .eyein import decode_ego
            things, ego = split_ego(objs)
            self._ego_seen = ego is not None
            if ego is not None and float(ego.max()) > 0.0:
                dr, dc = decode_ego(ego)
                if abs(dr) + abs(dc) > 0:
                    self._shift_marks(dr, dc)
            centers = object_centers(things)
            for mk in self.marks:
                if not centers:
                    break
                d, (r, c) = min((math.hypot(r - mk[0], c - mk[1]), (r, c)) for r, c in centers)
                if d <= TRACK_CELLS:
                    mk[0], mk[1] = r, c
        # 今見ている物：撃って着いたあと、真ん中から FIX_CELLS 以内でいちばん近い物を選び、次に撃つまで
        # その物（の印）に注意を置き続ける。動く物が真ん中から外れても、印は物に付いて動くので慣れは育つ。
        # 目の動きの地図は2回遅れで届くので、撃ったあと _pick_wait 回は選ばない。
        # 物が届かないとき（環境なしのテスト）は真ん中の「場所」。
        held = None
        if objs is None:
            fix = (CENTER, CENTER)
            near = [mk for mk in self.marks if math.hypot(mk[0] - CENTER, mk[1] - CENTER) <= TRACK_CELLS]
            held = max(near, key=lambda mk: mk[2]) if near else None
            if held is None:
                held = [CENTER, CENTER, 0.0]
                self.marks.append(held)
            held[0], held[1] = fix
        elif self._pick_wait > 0:
            self._pick_wait -= 1
        elif self._attended is not None and any(mk is self._attended for mk in self.marks):
            held = self._attended
        elif centers:
            # 撃った先（目の動きの地図で一緒にずらした点）にいちばん近い物。跳びが足りず、近くの小さな物の
            # 方が真ん中に近くても、狙った物に注意を置く。撃っていなければ真ん中
            ar, ac = self._aim if self._aim is not None else (CENTER, CENTER)
            d, fix = min((math.hypot(r - ar, c - ac), (r, c)) for r, c in centers)
            if d <= FIX_CELLS:
                near = [mk for mk in self.marks
                        if math.hypot(mk[0] - fix[0], mk[1] - fix[1]) <= TRACK_CELLS]
                held = max(near, key=lambda mk: mk[2]) if near else None
                if held is None:
                    held = [fix[0], fix[1], 0.0]
                    self.marks.append(held)
                held[0], held[1] = fix
                self._attended = held
                self._aim = None
        k = math.exp(-TICK_S / IOR_DECAY_S)
        for mk in self.marks:
            if mk is held:
                look = -HAB_TAU * math.log(max(1.0 - mk[2], 1e-6)) + TICK_S
                mk[2] = 1.0 - math.exp(-look / HAB_TAU)
            else:
                mk[2] *= k
        self.last_held = held is not None
        self._held_now = held

    def _mark_fire(self, r: int, c: int) -> None:
        """撃った回：新しい印は置かない（着いた物には、見ている間に慣れの印が育つ）。目が動いた分は
        次の回から目の動きの地図で足す。地図が届いていないとき（環境なしのテスト）だけ、指令の分で
        ずらす（今見ていた場所の印が、離れた場所に残る）。"""
        if self._attended is not None and any(mk is self._attended for mk in self.marks):
            # 離れる物の印を LEAVE_IOR まで強める（復帰抑制は、離れた物に見ていた長さによらずかかる）
            self._attended[2] = max(self._attended[2], LEAVE_IOR)
        if not self._ego_seen:
            self._shift_marks(CENTER - r, CENTER - c)
        self._attended = None
        self._pick_wait = 2 if self._ego_seen else 0
        self._aim = [float(r), float(c)] if self._ego_seen else None
        self._merge()
        self._render()

    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        m = inbox.get("prio")
        inhibit = float(inbox.get("gate3", 0.0))
        if self.use_ior:
            self._track(inbox.get("eyein"))
            self._merge()
            self._render()
            if m is not None:
                raw_peak = int(torch.argmax(m))
                self.last_ior_at_peak = float(self.ior.reshape(-1)[raw_peak])
                gain = IOR_GAIN * (1.0 - GOAL_RELIEF * min(max(inhibit, 0.0), 1.0))
                m = (m - gain * self.ior).clamp(min=0.0)
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
        if self.use_ior and self._attended is not None and any(mk is self._attended for mk in self.marks):
            hr = min(max(int(round(self._attended[0])), 0), GRID - 1)
            hc = min(max(int(round(self._attended[1])), 0), GRID - 1)
            relief = 1.0 - GOAL_RELIEF * min(max(inhibit, 0.0), 1.0)
            hold = float(m[hr, hc]) + FIX_HOLD * (1.0 - self._attended[2]) * relief
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
        if self.use_ior:
            self._mark_fire(r, c)
        out["eyeout"] = torch.tensor([(c - CENTER) / CENTER, (CENTER - r) / CENTER], dtype=DTYPE)
        return out
