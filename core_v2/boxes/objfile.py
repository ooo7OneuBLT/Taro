"""物の名簿（頭頂葉・頭頂間溝の役。段階4、2026-10-16、ユーザー承認の仕様書と設計図）。

仕様書 `2026-10-16_段階4_仕様書_物の名簿.md`。目の入口が見つけた物を1つずつ名簿の行に書き、
「もう見た（復帰抑制）」「慣れ（馴化）」「名前を結び付ける物」を物ごとに1か所で持つ。
段階3では、この3つが目を向ける反射（印）・意味の辞書（直前1秒の窓）・目の入口にばらばらにあり、
目が物を割るたびにずれた。ここに集めたので、目を向ける反射は「いつ撃つか」、意味の辞書は
「覚える・思い出す」だけに戻った。

人間との対応（記憶ベース・要確認）：
* 頭頂葉は、見えている物を少数（乳児で約3個、大人で約4個）1つずつ追う（object file。
  Kahneman・Treisman・Gibbs 1992、乳児は Feigenson・Carey 2003）→ 行は最大 MAX_ROWS（4）
* 同じ物かは、見た目より場所と動きのつながりで決める（Xu・Carey 1996）→ 行は場所で追う
* 一緒に動く面は1つの物（4か月児、Kellman・Spelke 1983、共通の動き）→ 割れたかけらは、
  隣り合って一緒に動いたら1行にまとめる。目が跳んでいる間の動きは使わない（跳びの間は動きの
  感じ方が抑えられる＝サッカード抑制。目の動きの見積もりの誤差は全部の物に同じだけ乗るので、
  使うと止まった物どうしまで「一緒に動いた」ことになる）

**生まれつき（手で決める）**：行の数・場所で追うやり方・まとめる決まり・慣れと薄れ方の形と数字。
**経験で変わる**：どの物が名簿にいるか、それぞれの見た目の平均・慣れ・もう見た。学習するNNではない。

1つの行（Row）は、1つ以上の「かけら」（Part、目の入口の1つの物を追ったもの）の集まり。
行の強さ（印）は段階3の orienteye の印と同じ意味：見ている間は 1−exp(−見ている時間/HAB_TAU) で
育ち（慣れ）、離れたら LEAVE_IOR まで上げてから IOR_DECAY_S で薄れる（復帰抑制）。

受け取るもの：
* eyein     : 物ごとの (荷札, 地図)、最後に目の動きの地図（荷札ゼロ）
* prio      : 合わせた地図（脱馴化で、行の位置の値の「ふだん」と比べる）
* orienteye : 撃った指令の写し（撃たない回はゼロ）。撃った先の物に注意を移す
出すもの：
* orienteye : 行ごとの (荷札, 印の山＝強さ×σ IOR_SIGMA_CELLS の山)。列の最後に必ず
              (荷札ゼロ, 今見ている行の山＝高さは慣れ、下限0.001) を1つ足す。今見ている行が無ければ
              地図はゼロ（目の入口の目の動きの地図と同じ「列の最後の荷札ゼロ＝特別な組」の約束）
* surprise  : 段階8（2026-10-18）。席1〜SEATS の順に (行の荷札, 場所の山)。山の高さ＝1：見えている、
              0.5：視野の中にいるのに ABSENT_TICKS 回（目が跳んだ回は数えない）続けて見えない。視野の外の行は地図ゼロ、
              空席は荷札も地図もゼロ。そのあとに (荷札ゼロ, 今見ている行の山。視野の外なら地図ゼロ) と、
              列の最後に必ず (荷札ゼロ, 目立ちが急に下がった所の山) を足す
              （行の位置の目立ちが「ふだん」より急に下がり、そのあと見えていない行。脱馴化と同じ数字で下がりを見る）。
              席は、新しい行が、空いてからいちばん長い席に座る（行が消えたその回に同じ席へ別の物が座らないので、
              驚きの係は席ごとに同じ物を追える。行は最大 MAX_ROWS なので、SEATS＝MAX_OBJECTS−2＝6 席のうち2席以上は空いている）
* semantic  : 段階6（2026-10-17）から列。行ごとに (荷札, 山)。山の高さ＝直前 LOOK_WINDOW_S（3秒）に名簿の物を
              見ていた時間のうち、その行を見ていた割合（合わせて1。見ていない行は載せない）。そのあとに
              (今見ている行の荷札, ゼロの地図)、列の最後に必ず (名前を結び付ける行の荷札, ゼロの地図) を足す
              （無ければ荷札もゼロ。今見ている行は2026-10-17から、意味の辞書が聞いた語の名残りのあいだ毎回数えるため）。
              意味の辞書は覚え方Bで「今見ている行」、覚え方Aで最後の組を使う
"""

from __future__ import annotations

import functools
import math

import torch

from ..formats import DTYPE, GRID, MAX_OBJECTS, TAG_DIM, make_tag
from .base import Box

TICK_S = 0.1
CENTER = (GRID - 1) / 2.0

# 段階3の orienteye から移した数字（意味も値も同じ）
IOR_DECAY_S = 8.0        # もう見たの印が薄れる時定数（秒）［Tier3・仮置き、段階3①の実測で決めた］
IOR_SIGMA_CELLS = 2.0    # 印の山の広がり［仮置き］
HAB_TAU = 6.0            # 見ている間に慣れが育つ時定数（秒）［Tier3・仮置き］
LEAVE_IOR = 1.0          # 目を離した行の印の強さ［Tier3・仮置き］
DISHAB_JUMP = 0.3        # 脱馴化（段階3と同じ）［Tier3・仮置き］
DISHAB_Z = 3.0
DISHAB_KEEP = 0.5
DISHAB_ALPHA = 0.2
FIX_CELLS = 3.0          # 撃った先からこれ以内の行を「今見ている行」にする（約8度）［仮置き］
TRACK_CELLS = 1.5        # かけらが1回に追える距離（目の動きを引いたあと）［仮置き］
MARK_MIN = 0.02          # これより弱い印は 0 とみなす

# 段階4で新しく決めた数字
MAX_ROWS = 4             # 名簿の行の数（乳児3〜大人4）［Tier2・文献の範囲内で仮置き］
KEEP_S = 2.0             # 視野の中にあるのに見えないまま、これだけたったら、かけらを消す［Tier3・仮置き］
OFF_MARGIN = 3 * IOR_SIGMA_CELLS   # 視野の外へこれ以上出たかけらは消す（段階3の印と同じ）
NEAR_CELLS = 4.0         # まとめる候補：かけらどうしの中心がこれ以内（隣り合う）［仮置き］
MOVE_WINDOW = 10         # 一緒に動いたかを見る回数（1秒）
MOVE_MIN = 0.15          # 1回あたりの平均の動き（マス）がこれ以上で「動いた」［Tier3・仮置き］
COMOVE_COS = 0.7         # 2つのかけらの動きの向きの揃い方（窓の中の内積の正規化）［Tier3・仮置き］
EGO_STILL = 0.5          # 目の動きがこれ（マス）を超えた回は、物の動きに数えない（サッカード抑制）
ATTACH_CELLS = 3.0       # 追っている物のかけらからこれ以内に現れたかけらは、その物の一部（約8度）［Tier3・仮置き］
ABSENT_TICKS = 3         # これだけ見えていない行は、新しい物に席を譲る（目の入口の切り出しは3回に1回）
MIN_LOOK_S = 1.0         # 名前を結び付けるには、少なくともこれだけ続けて見ていた行［Tier3・仮置き］

# 段階6で新しく決めた数字
LOOK_WINDOW_S = 3.0      # 名前を聞く前のこの秒数に、どの物をどれだけ見ていたか（子は名づけの約3秒前から
                         # 見ている物と語を結び付けやすい、Yu・Smith 2012。記憶ベース）［Tier2］

# 段階8で新しく決めた数（新しい手合わせの数は無い）
SEATS = MAX_OBJECTS - 2  # 驚きの係へ渡す席の数（列の最後の2つは、今見ている行の山と、目立ちの下がりの地図）

_YS, _XS = torch.meshgrid(torch.arange(GRID, dtype=DTYPE), torch.arange(GRID, dtype=DTYPE),
                          indexing="ij")


@functools.lru_cache(maxsize=4096)
def _bump(r: float, c: float, sigma: float) -> torch.Tensor:
    return torch.exp(-((_XS - c) ** 2 + (_YS - r) ** 2) / (2.0 * sigma ** 2))


def bump_at(r: float, c: float, sigma: float = IOR_SIGMA_CELLS) -> torch.Tensor:
    """山の地図。同じ場所の山は作り直さず使い回す（値は同じ。受け取った側はその場で書き換えない約束、base.py）。"""
    return _bump(float(r), float(c), float(sigma))


def split_ego(objs):
    """目の入口からの組の列を、物の組と「目の動きの地図」（荷札ゼロ、列の最後）に分ける。"""
    if not objs:
        return [], None
    tag, m = objs[-1]
    if float(torch.linalg.vector_norm(tag)) == 0.0:
        return list(objs[:-1]), m
    return list(objs), None


def object_center(m: torch.Tensor) -> tuple[float, float]:
    """地図の山の中心（行・列、小数）。山の 0.5 以上の重心。"""
    w = (m >= 0.5 * float(m.max())).to(DTYPE) * m
    tot = float(w.sum())
    return float((w * _YS).sum()) / tot, float((w * _XS).sum()) / tot


def in_view(r: float, c: float) -> bool:
    return 0.0 <= r <= GRID - 1 and 0.0 <= c <= GRID - 1


class Part:
    """目の入口の1つの物を追ったもの（行のかけら）。"""

    def __init__(self, r: float, c: float, tag: torch.Tensor, t: int) -> None:
        self.r, self.c = r, c
        self.tag_sum = make_tag(tag).clone()
        self.n_seen = 1
        self.last_seen = t
        self.moves: list[tuple[float, float] | None] = []   # 1回ごとの物そのものの動き（None＝数えない回）

    @property
    def tag(self) -> torch.Tensor:
        return make_tag(self.tag_sum)


class Row:
    """名簿の1行＝1つの物。"""

    def __init__(self, part: Part, number: int) -> None:
        self.number = number
        self.parts = [part]
        self.strength = 0.0      # 慣れ（見ている間）／もう見た（離れたあと）
        self.look_ticks = 0      # 今回続けて見ている回数
        self.sal = None          # 脱馴化：(ふだんの値, ふだんの揺れ幅)
        self.seat = -1           # 段階8：驚きの係へ渡す席
        self.unseen = 0          # 段階8：続けて見えていない回数（目が跳んだ回は数えない）
        self.dropped = False     # 段階8：最後に見えたあと、行の位置の目立ちが急に下がった

    @property
    def pos(self) -> tuple[float, float]:
        return (sum(p.r for p in self.parts) / len(self.parts),
                sum(p.c for p in self.parts) / len(self.parts))

    @property
    def tag(self) -> torch.Tensor:
        """かけらの見た目を、見えた回数で重みを付けて平均した荷札。"""
        return make_tag(torch.stack([p.tag_sum for p in self.parts]).sum(0))

    @property
    def last_seen(self) -> int:
        return max(p.last_seen for p in self.parts)

    def place_only(self) -> bool:
        """物が届かないとき（環境なしのテスト）の「場所」の行か。"""
        return all(p.n_seen == 0 for p in self.parts)


def comove(a: Part, b: Part) -> bool:
    """2つのかけらが、窓の中で同じ向きに動いたか（両方が動いた回だけ数える）。"""
    pairs = [(u, v) for u, v in zip(a.moves[-MOVE_WINDOW:], b.moves[-MOVE_WINDOW:])
             if u is not None and v is not None]
    if len(pairs) < MOVE_WINDOW // 2:
        return False
    n = len(pairs)
    ea = sum(math.hypot(*u) for u, _ in pairs) / n
    eb = sum(math.hypot(*v) for _, v in pairs) / n
    if ea < MOVE_MIN or eb < MOVE_MIN:
        return False
    dot = sum(u[0] * v[0] + u[1] * v[1] for u, v in pairs)
    na = math.sqrt(sum(u[0] ** 2 + u[1] ** 2 for u, _ in pairs))
    nb = math.sqrt(sum(v[0] ** 2 + v[1] ** 2 for _, v in pairs))
    return dot / max(na * nb, 1e-9) >= COMOVE_COS


class Objfile(Box):
    BOX_ID = "objfile"

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.rows: list[Row] = []
        self._next_number = 1
        self.attended: Row | None = None   # 今注意を置いている行（次に撃つまで）
        self.named: Row | None = None      # 最後に MIN_LOOK_S 以上続けて見た行（名前を結び付ける行）
        self._held_now: Row | None = None
        self._pick_wait = 0
        self._aim: list[float] | None = None
        self._ego_seen = False
        self._look_hist: list[Row | None] = []   # 直前 LOOK_WINDOW_S のあいだ、毎回見ていた行
        # 記録用
        self.last_shift = (0.0, 0.0)
        self.n_dishab = 0
        self.n_merge = 0
        self.last_merges: list[tuple[int, int]] = []
        self._seat_free_t = [-(10 ** 9) + i for i in range(SEATS)]   # 席が空いた回（はじめは席の番号順）
        self._ego_moving = False

    # ---- 行とかけら ----
    def _remove_row(self, rw: Row) -> None:
        self.rows.remove(rw)
        if rw.seat >= 0:
            self._seat_free_t[rw.seat] = self.t

    def _take_seat(self, rw: Row) -> None:
        used = {r.seat for r in self.rows if r is not rw}
        free = [i for i in range(SEATS) if i not in used]
        rw.seat = min(free, key=lambda i: self._seat_free_t[i])

    def _new_row(self, part: Part, captured: bool = True) -> Row | None:
        """新しい物に行を作る。席（MAX_ROWS）が埋まっていれば、もう見えなくなった行（ABSENT_TICKS 回以上
        見えていない行。印の無い行を優先し、その中で一番長く見えていない行）と入れ替える。今見えている物の
        行は、新しく現れた物に席を譲らない（追っている物は、余分な物が現れても追い続ける）。
        ただし注意を向けた先（captured＝撃った先・真ん中）の物は、見えている行とも入れ替えて席を取る。"""
        if len(self.rows) >= MAX_ROWS:
            cands = [rw for rw in self.rows if rw is not self.attended and rw is not self.named]
            if not captured:
                cands = [rw for rw in cands if self.t - rw.last_seen >= ABSENT_TICKS]
            if not cands:
                return None
            victim = min(cands, key=lambda rw: (rw.strength >= MARK_MIN, rw.last_seen))
            self._remove_row(victim)
        row = Row(part, self._next_number)
        self._next_number += 1
        self._take_seat(row)
        self.rows.append(row)
        return row

    def _parts(self):
        for rw in self.rows:
            for p in rw.parts:
                yield rw, p

    def _shift_all(self, dr: float, dc: float) -> None:
        for _, p in self._parts():
            p.r += dr
            p.c += dc
        if self._aim is not None:
            self._aim = [self._aim[0] + dr, self._aim[1] + dc]
        self.last_shift = (dr, dc)

    def _match(self, things, ego_moving: bool) -> None:
        """今の回の物を、かけらに1対1で割り当てる（近い組から）。割り当たらなかった物は新しい行。"""
        dets = []
        for tag, m in things:
            if float(torch.linalg.vector_norm(tag)) == 0.0 or float(m.max()) <= 0.0:
                continue
            dets.append((object_center(m), tag))
        parts = list(self._parts())
        pairs = sorted(
            (math.hypot(d[0][0] - p.r, d[0][1] - p.c), i, j)
            for i, d in enumerate(dets) for j, (_, p) in enumerate(parts))
        used_d, used_p = set(), set()
        for dist, i, j in pairs:
            if dist > TRACK_CELLS:
                break
            if i in used_d or j in used_p:
                continue
            used_d.add(i)
            used_p.add(j)
            (r, c), tag = dets[i]
            p = parts[j][1]
            p.moves.append(None if ego_moving else (r - p.r, c - p.c))
            p.r, p.c = r, c
            p.tag_sum = p.tag_sum + make_tag(tag)
            p.n_seen += 1
            p.last_seen = self.t
        for j, (_, p) in enumerate(parts):
            if j not in used_p:
                p.moves.append(None)
            del p.moves[:-MOVE_WINDOW]
        ar, ac = self._aim if self._aim is not None else (CENTER, CENTER)
        for i, ((r, c), tag) in enumerate(dets):
            if i in used_d:
                continue
            part = Part(r, c, tag, self.t)
            # 追っている物のすぐそば（ATTACH_CELLS 以内）に現れたかけらは、その物の一部とする
            # （接している面は1つの物＝まとまりの原則。目が物を割っても、割れたかけらは同じ行に入る）
            near = [(math.hypot(r - q.r, c - q.c), rw) for rw, q in self._parts()
                    if not rw.place_only()]
            d, rw = min(near, key=lambda x: x[0], default=(99.0, None))
            if rw is not None and d <= ATTACH_CELLS:
                part.moves = [None] * len(rw.parts[0].moves)
                rw.parts.append(part)
                continue
            self._new_row(part, captured=math.hypot(r - ar, c - ac) <= FIX_CELLS)

    def _forget(self) -> None:
        """視野の中で KEEP_S 見えないかけら・視野の外へ出すぎたかけらを消す。かけらの無い行も消す。"""
        lim = OFF_MARGIN
        keep_ticks = round(KEEP_S / TICK_S)
        for rw in list(self.rows):
            rw.parts = [p for p in rw.parts
                        if -lim <= p.r <= GRID - 1 + lim and -lim <= p.c <= GRID - 1 + lim
                        and not (in_view(p.r, p.c) and p.n_seen > 0 and self.t - p.last_seen > keep_ticks)]
            if not rw.parts:
                self._remove_row(rw)
                for name in ("attended", "named", "_held_now"):
                    if getattr(self, name) is rw:
                        setattr(self, name, None)

    def _merge(self) -> None:
        """隣り合って一緒に動いた、別の行のかけらどうしを1行にまとめる（共通の動き）。"""
        self.last_merges = []
        changed = True
        while changed:
            changed = False
            for a in self.rows:
                for b in self.rows:
                    if a is b or a.number > b.number:
                        continue
                    if any(math.hypot(p.r - q.r, p.c - q.c) <= NEAR_CELLS and comove(p, q)
                           for p in a.parts for q in b.parts):
                        # 番号の若い方（先に名簿に入った方）に寄せる。印は強い方、見ている回数は長い方
                        a.parts += b.parts
                        a.strength = max(a.strength, b.strength)
                        a.look_ticks = max(a.look_ticks, b.look_ticks)
                        for name in ("attended", "named", "_held_now"):
                            if getattr(self, name) is b:
                                setattr(self, name, a)
                        self._remove_row(b)
                        self.n_merge += 1
                        self.last_merges.append((a.number, b.number))
                        changed = True
                        break
                if changed:
                    break

    def _pick_attended(self, have_objs: bool) -> Row | None:
        """今見ている行を決め、その行の慣れを育てる。ほかの行は薄れさせる。"""
        held = None
        if not have_objs:
            # 物が届かないとき（環境なしのテスト）は、真ん中の「場所」の行
            near = [rw for rw in self.rows
                    if math.hypot(rw.pos[0] - CENTER, rw.pos[1] - CENTER) <= TRACK_CELLS]
            held = max(near, key=lambda rw: rw.strength) if near else None
            if held is None:
                p = Part(CENTER, CENTER, torch.zeros(TAG_DIM, dtype=DTYPE), self.t)
                p.n_seen = 0
                held = self._new_row(p)
            if held is not None:
                held.parts = [held.parts[0]]
                held.parts[0].r, held.parts[0].c = CENTER, CENTER
            self.attended = held
        elif self._pick_wait > 0:
            self._pick_wait -= 1
        elif self.attended is not None and self.attended in self.rows:
            held = self.attended
        else:
            ar, ac = self._aim if self._aim is not None else (CENTER, CENTER)
            seen = [rw for rw in self.rows if rw.last_seen == self.t]
            if seen:
                d, held = min(((min(math.hypot(p.r - ar, p.c - ac) for p in rw.parts), rw) for rw in seen),
                              key=lambda x: x[0])
                if d <= FIX_CELLS:
                    self.attended = held
                    self._aim = None
                else:
                    held = None
        k = math.exp(-TICK_S / IOR_DECAY_S)
        for rw in self.rows:
            if rw is held:
                look = -HAB_TAU * math.log(max(1.0 - rw.strength, 1e-6)) + TICK_S
                rw.strength = 1.0 - math.exp(-look / HAB_TAU)
                rw.look_ticks += 1
                if rw.look_ticks * TICK_S >= MIN_LOOK_S and not rw.place_only():
                    self.named = rw
            else:
                rw.strength *= k
                rw.look_ticks = 0
        self._held_now = held
        return held

    def _dishabituate(self, m: torch.Tensor) -> None:
        for rw in self.rows:
            r0, c0 = rw.pos
            r = min(max(int(round(r0)), 0), GRID - 1)
            c = min(max(int(round(c0)), 0), GRID - 1)
            v = float(m[r, c])
            if rw.sal is None:
                rw.sal = (v, 0.0)
                continue
            mu, dev = rw.sal
            if v - mu >= max(DISHAB_JUMP, DISHAB_Z * dev):
                rw.strength *= DISHAB_KEEP
                self.n_dishab += 1
            # 段階8：同じ線で、急に下がったのも見る（周辺で消えたことを安く知らせる道。目が跳んだ回は見ない）
            if mu - v >= max(DISHAB_JUMP, DISHAB_Z * dev) and rw.last_seen != self.t and not self._ego_moving:
                rw.dropped = True
            dev = (1 - DISHAB_ALPHA) * dev + DISHAB_ALPHA * abs(v - mu)
            mu = (1 - DISHAB_ALPHA) * mu + DISHAB_ALPHA * v
            rw.sal = (mu, dev)

    def _on_fire(self, cmd: torch.Tensor) -> None:
        """目を向ける反射が撃った（1回遅れで届く）：離れる行に復帰抑制をかけ、撃った先に注意を移す。"""
        h, v = float(cmd[0]), float(cmd[1])
        r, c = CENTER - v * CENTER, CENTER + h * CENTER
        if self.attended is not None and self.attended in self.rows:
            self.attended.strength = max(self.attended.strength, LEAVE_IOR)
            self.attended.look_ticks = 0
        if not self._ego_seen:
            # 目の動きの地図が届かないとき（環境なしのテスト）は、指令の分でずらす
            self._shift_all(CENTER - r, CENTER - c)
            self._aim = None
            self._pick_wait = 0
        else:
            self._aim = [r, c]
            self._pick_wait = 1    # 撃った回から数えて段階3と同じ2回待つ（指令は1回遅れで届く）
        self.attended = None

    # ---- 1回 ----
    def step(self, inbox):
        out = self.passthrough()
        self.t += 1
        self.last_shift = (0.0, 0.0)
        objs = inbox.get("eyein")
        cmd = inbox.get("orienteye")
        have_objs = objs is not None
        if have_objs:
            things, ego = split_ego(objs)
            self._ego_seen = ego is not None
            ego_moving = False
            if ego is not None and float(ego.max()) > 0.0:
                from .eyein import decode_ego
                dr, dc = decode_ego(ego)
                if abs(dr) + abs(dc) > 0:
                    self._shift_all(dr, dc)
                ego_moving = math.hypot(dr, dc) > EGO_STILL
            self._match(things, ego_moving)
            self._ego_moving = ego_moving
            for rw in self.rows:
                if rw.last_seen == self.t:
                    rw.unseen, rw.dropped = 0, False
                elif not ego_moving:
                    rw.unseen += 1
        if cmd is not None and float(torch.linalg.vector_norm(cmd)) > 0.0:
            self._on_fire(cmd)
        self._forget()
        if have_objs:
            self._merge()
        held = self._pick_attended(have_objs)
        self._look_hist.append(held)
        del self._look_hist[:-int(round(LOOK_WINDOW_S / TICK_S))]
        m = inbox.get("prio")
        if m is not None and DISHAB_KEEP < 1.0:
            self._dishabituate(m)
        for rw in self.rows:
            if rw.strength < MARK_MIN and rw is not held:
                rw.strength = 0.0
        if "orienteye" in out:
            lst = [(rw.tag, rw.strength * bump_at(*rw.pos)) for rw in self.rows]
            att = torch.zeros(GRID, GRID, dtype=DTYPE)
            if held is not None:
                # 高さ＝慣れ。見始め（慣れ0）でも場所が分かるよう、高さの下限を 0.001 にする
                att = max(held.strength, 1e-3) * bump_at(*held.pos)
            lst.append((torch.zeros(TAG_DIM, dtype=DTYPE), att))
            out["orienteye"] = lst
        if "surprise" in out:
            out["surprise"] = self.surprise_list()
        if "semantic" in out:
            lst = [(rw.tag, share * bump_at(*rw.pos)) for rw, share in self.look_shares()]
            zero_tag, zero_map = torch.zeros(TAG_DIM, dtype=DTYPE), torch.zeros(GRID, GRID, dtype=DTYPE)
            named = self.named.tag if self.named is not None and self.named in self.rows else zero_tag
            lst.append((held.tag if held is not None else zero_tag, zero_map))
            lst.append((named, zero_map))
            out["semantic"] = lst
        return out

    def look_shares(self) -> list[tuple[Row, float]]:
        """名簿にいる行ごとに、直前 LOOK_WINDOW_S のうち名簿の物を見ていた時間の中で、その行を見ていた割合
        （0より大きい行だけ。合わせて1）。何も見ていなかった時間（画面と画面のあいだ）で薄めない。"""
        looked = [h for h in self._look_hist if h is not None and h in self.rows]
        out = []
        for rw in self.rows:
            k = sum(1 for h in looked if h is rw)
            if k > 0:
                out.append((rw, k / len(looked)))
        return out

    def surprise_list(self) -> list:
        """段階8：驚きの係へ渡す列（席の順、そのあと今見ている行の山、最後に目立ちの下がりの地図）。"""
        zero_tag, zero_map = torch.zeros(TAG_DIM, dtype=DTYPE), torch.zeros(GRID, GRID, dtype=DTYPE)
        seats = [(zero_tag, zero_map)] * SEATS
        drop = zero_map
        for rw in self.rows:
            if rw.place_only() or rw.seat < 0:
                continue
            r, c = rw.pos
            m = zero_map
            if 0 < rw.unseen and self._gone_out(rw):
                pass                             # 動いていた物が、その向きのまま視野の外へ出た（見えなくなったのではない）
            elif in_view(r, c):
                m = (1.0 if rw.unseen < ABSENT_TICKS else 0.5) * bump_at(r, c)
                if rw.dropped and rw.unseen >= ABSENT_TICKS:
                    drop = torch.maximum(drop, bump_at(r, c))
            seats[rw.seat] = (rw.tag, m)
        held = self._held_now
        hm = zero_map
        if held is not None and held in self.rows and not held.place_only() and in_view(*held.pos):
            hm = bump_at(*held.pos)
        return seats + [(zero_tag, hm), (zero_tag, drop)]

    def _gone_out(self, rw: Row) -> bool:
        """段階8：見えなくなった行が、見えていた最後に動いていた（1回あたり MOVE_MIN マス以上）なら、同じ速さで
        動き続けたとして、見えなくなってからの回数ぶん進めた所が視野の外か（動く物は同じ道を進み続けるという
        生まれつきの見込み、Spelke の連続性。動きの数え方は「一緒に動いた」と同じ：目が跳んだ回は数えない）。"""
        mv = [m for p in rw.parts for m in p.moves[-MOVE_WINDOW:] if m is not None][-3:]
        if not mv:
            return False
        dr = sum(m[0] for m in mv) / len(mv)
        dc = sum(m[1] for m in mv) / len(mv)
        if math.hypot(dr, dc) < MOVE_MIN:
            return False
        r, c = rw.pos
        return not in_view(r + dr * rw.unseen, c + dc * rw.unseen)

    # ---- 記録用 ----
    def ior_map(self) -> torch.Tensor:
        m = torch.zeros(GRID, GRID, dtype=DTYPE)
        for rw in self.rows:
            m = torch.maximum(m, rw.strength * bump_at(*rw.pos))
        return m

    def summary(self) -> str:
        """名簿を1行の文字にする：番号:かけら数:強さ@行,列"""
        return " ".join(f"{rw.number}:{len(rw.parts)}:{rw.strength:.2f}@{rw.pos[0]:.0f},{rw.pos[1]:.0f}"
                        + ("*" if rw is self._held_now else "") for rw in self.rows)
