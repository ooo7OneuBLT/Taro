"""段階3〜4：復帰抑制・慣れ・見続ける力の確かめ。環境なしで回る。
段階4から印は物の名簿（objfile）が持ち、目を向ける反射（orienteye）は名簿の出力を引いて撃つだけ。
Pair は、脳と同じく係どうしの矢印に1回ずつ遅れを入れて2つを回す。"""

import math

import torch

from core_v2.boxes.eyein import ego_map
from core_v2.boxes.objfile import HAB_TAU, IOR_DECAY_S, LEAVE_IOR, Objfile, bump_at
from core_v2.boxes.orienteye import REFRACTORY_TICKS, Orienteye
from core_v2.formats import DTYPE, GRID, TAG_DIM

ZERO2 = torch.zeros(2)


def shift_map(m: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """地図を下へ dr・右へ dc マスずらす。はみ出た所は捨て、空いた所はゼロ。"""
    out = torch.zeros_like(m)
    r0, r1 = max(dr, 0), GRID + min(dr, 0)
    c0, c1 = max(dc, 0), GRID + min(dc, 0)
    if r1 > r0 and c1 > c0:
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


class Pair:
    def __init__(self, use_ior=True):
        self.of, self.oe = Objfile(), Orienteye(use_ior=use_ior)
        self.of_out = None
        self.cmd = ZERO2

    def step(self, prio, eyein=None, gate=0.0):
        inbox = {"prio": prio, "orienteye": self.cmd}
        if eyein is not None:
            inbox["eyein"] = eyein
        of_out = self.of.step(inbox)
        oe_in = {"prio": prio, "gate3": torch.tensor(gate)}
        if self.of_out is not None:
            oe_in["objfile"] = self.of_out
        out = self.oe.step(oe_in)
        self.of_out = of_out["orienteye"]
        self.cmd = out["objfile"]
        return out


def two_peaks():
    """右に高い山（行13・列19）、左に低い山（行13・列8）。どちらを見ても、もう片方は視野に入る。"""
    m = torch.zeros(GRID, GRID, dtype=DTYPE)
    m[13, 19] = 1.0
    m[13, 8] = 0.8
    return m


def run(pair, world, ticks):
    eye = [13.5, 13.5]
    fired = []
    for _ in range(ticks):
        dr, dc = int(round(13.5 - eye[0])), int(round(13.5 - eye[1]))
        out = pair.step(shift_map(world, dr, dc))
        cmd = out["eyeout"]
        if float(torch.linalg.vector_norm(cmd)) > 0:
            c = 13.5 + float(cmd[0]) * 13.5
            r = 13.5 - float(cmd[1]) * 13.5
            eye = [eye[0] + (r - 13.5), eye[1] + (c - 13.5)]
            fired.append((round(eye[0]), round(eye[1])))
    return fired


def test_without_ior_goes_back_to_the_same_peak():
    fired = run(Pair(use_ior=False), two_peaks(), 3 * REFRACTORY_TICKS + 1)
    assert [c for _, c in fired] == [19]


def test_ior_moves_on_to_the_second_peak():
    cols = [c for _, c in run(Pair(), two_peaks(), 4 * REFRACTORY_TICKS)]
    assert cols[:2] == [19, 8]          # 見ているうちに慣れて、2番目の山へ移る


def test_ior_fades_and_allows_return():
    cols = [c for _, c in run(Pair(), two_peaks(), int(6 * IOR_DECAY_S / 0.1))]
    assert cols[:2] == [19, 8]
    assert len(cols) >= 3 and cols[2] == 19     # 印が薄れて、右の山へ戻れる


def test_mark_shifts_with_the_eye():
    """見ている間は真ん中（の場所）に慣れが育ち、撃つと（目の動きの地図が無いので指令の分）
    離れた場所に残り、LEAVE_IOR まで上がる＝復帰抑制。"""
    of = Objfile()
    for _ in range(20):                # 2秒見る
        of.step({"orienteye": ZERO2})
    m = of.ior_map()
    r, c = divmod(int(torch.argmax(m)), GRID)
    assert (r, c) in {(13, 13), (13, 14), (14, 13), (14, 14)}
    assert abs(float(m.max()) - (1 - math.exp(-2.0 / HAB_TAU))) < 0.05
    cmd = torch.tensor([(19 - 13.5) / 13.5, 0.0])   # 右へ約6マス跳んだ
    of.step({"orienteye": cmd})
    m = of.ior_map()
    r, c = divmod(int(torch.argmax(m)), GRID)
    assert 7 <= c <= 9 and float(m.max()) > 0.9 * LEAVE_IOR


def run_moving(pair, ticks, speed=0.5):
    """動く目立つ物（値1.0、行13を右へ）と、止まった弱い物（値0.4、行13・列6）。"""
    eye = [13.5, 13.5]
    fired = []
    for t in range(ticks):
        mover = (13.0, 17.0 + speed * t)
        if mover[1] > 22:
            mover = (13.0, 22.0 - (mover[1] - 22.0) % 10)
        still = (13.0, 6.0)
        to_ret = lambda p: (p[0] - eye[0] + 13.5, p[1] - eye[1] + 13.5)
        mr, sr = to_ret(mover), to_ret(still)
        prio = torch.maximum(bump_at(*mr, sigma=1.0), 0.4 * bump_at(*sr, sigma=1.0))
        objs = [(torch.ones(TAG_DIM), bump_at(*mr, sigma=1.0)), (-torch.ones(TAG_DIM), bump_at(*sr, sigma=1.0))]
        out = pair.step(prio, objs)
        cmd = out["eyeout"]
        if float(torch.linalg.vector_norm(cmd)) > 0:
            dc, dr = float(cmd[0]) * 13.5, -float(cmd[1]) * 13.5
            eye = [eye[0] + dr, eye[1] + dc]
            dm = abs(eye[1] - mover[1]) + abs(eye[0] - mover[0])
            ds = abs(eye[1] - still[1]) + abs(eye[0] - still[0])
            fired.append("mover" if dm < ds else "still")
    return fired


def test_mark_follows_a_moving_object():
    fired = run_moving(Pair(), 20 * REFRACTORY_TICKS, speed=0.1)
    assert fired[0] == "mover"
    assert "still" in fired           # 動く物に慣れが育って、やがて止まった弱い物へ移る


def test_mark_stays_on_target_when_the_eye_undershoots():
    """目の跳びが指令の0.6倍でも、目の動きの地図で行をずらすので、慣れは着いた物に育つ。
    まわりの小さな物へは移らない。目の入口の出力は1回遅れで届く。"""
    world = {"target": (13.0, 20.0), "other": (13.0, 7.0),
             "clutter1": (15.0, 17.0), "clutter2": (11.0, 10.0), "clutter3": (17.0, 12.0)}
    eye = [13.5, 13.5]
    prev_eye = list(eye)
    pair = Pair()
    pending = None

    def step():
        nonlocal pending, prev_eye
        ret = {k: (p[0] - eye[0] + 13.5, p[1] - eye[1] + 13.5) for k, p in world.items()}
        dx_px, dy_px = -(eye[1] - prev_eye[1]) * 8.0, -(eye[0] - prev_eye[0]) * 8.0
        prev_eye = list(eye)
        now = [(make(k), bump_at(*p, sigma=1.0)) for k, p in ret.items()]
        now.append((torch.zeros(TAG_DIM), ego_map(dx_px, dy_px)))
        delivered, pending = pending, now
        prio = torch.maximum(bump_at(*ret["target"], sigma=1.0), 0.6 * bump_at(*ret["other"], sigma=1.0))
        return pair.step(prio, delivered if delivered is not None else []), ret

    def make(k):
        g = torch.Generator().manual_seed(hash(k) % 1000)
        return torch.randn(TAG_DIM, generator=g)

    pair.oe.last_fire = 0
    step()
    step()
    pair.oe.last_fire = -100
    out, _ = step()
    cmd = out["eyeout"]
    assert float(torch.linalg.vector_norm(cmd)) > 0     # 目立つ target へ撃つ
    eye[1] += 0.6 * float(cmd[0]) * 13.5
    pair.oe.last_fire = pair.oe.t + 100
    for _ in range(12):
        _, ret = step()
    m = pair.of.ior_map()
    tr, tc = ret["target"]
    r, c = divmod(int(torch.argmax(m)), GRID)
    assert abs(r - tr) <= 1 and abs(c - tc) <= 1
    assert float(m[round(tr), round(tc)]) > 0.1
    cr, cc = ret["clutter1"]
    assert float(m[round(cr), round(cc)]) < 0.1


def test_goal_weakens_ior_on_a_seen_object():
    """目的を持って探している間（抑える強さ1）は、さっき見た物の印を弱く引くので撃てる。"""
    prio = 0.7 * bump_at(9.0, 6.0, sigma=1.0)
    lst = [(torch.ones(TAG_DIM), 0.5 * bump_at(9.0, 6.0)), (torch.zeros(TAG_DIM), torch.zeros(GRID, GRID))]

    def fire(gate):
        return float(torch.linalg.vector_norm(
            Orienteye().step({"prio": prio, "gate3": torch.tensor(gate), "objfile": lst})["eyeout"])) > 0
    assert fire(0.0)                   # 目的なし：0.7−0.5=0.2 でも撃つ
    assert fire(1.0)                   # 目的あり：印を弱めて撃てる
    import core_v2.boxes.orienteye as oe
    keep, oe.GOAL_RELIEF = oe.GOAL_RELIEF, 0.0
    try:
        assert not fire(1.0)           # 弱めなければ足切りで止まる
    finally:
        oe.GOAL_RELIEF = keep


def test_holding_keeps_looking_until_habituated():
    """今見ている行（慣れ0）があると、少し目立つ別の山があっても撃たない（見続ける力）。"""
    prio = torch.maximum(0.6 * bump_at(13.0, 13.0, sigma=1.0), 0.9 * bump_at(13.0, 20.0, sigma=1.0))
    att = 1e-3 * bump_at(13.0, 13.0)
    lst = [(torch.ones(TAG_DIM), att), (torch.zeros(TAG_DIM), att)]
    out = Orienteye().step({"prio": prio, "gate3": torch.zeros(()), "objfile": lst})
    assert float(torch.linalg.vector_norm(out["eyeout"])) == 0
