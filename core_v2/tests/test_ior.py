"""段階3：復帰抑制（orienteye、仕様書 付録A）の確かめ。環境なしで回る。"""

import torch

from core_v2.boxes.orienteye import (IOR_DECAY_S, REFRACTORY_TICKS, Orienteye, shift_map)
from core_v2.formats import DTYPE, GRID


def two_peaks():
    """右に高い山（行13・列19）、左に低い山（行13・列8）。どちらを見ても、もう片方は視野に入る。"""
    m = torch.zeros(GRID, GRID, dtype=DTYPE)
    m[13, 19] = 1.0
    m[13, 8] = 0.8
    return m


def run(box, world, ticks):
    """「世界の地図」を目の位置に合わせて網膜の地図へ写しながら回す。撃ったら目を動かす。
    eye=(行, 列) は世界の中で今見ているマス。返り値は撃った先（世界の座標）の列。"""
    eye = [13.5, 13.5]
    fired = []
    for _ in range(ticks):
        dr, dc = int(round(13.5 - eye[0])), int(round(13.5 - eye[1]))
        retina = shift_map(world, dr, dc)
        out = box.step({"prio": retina, "gate3": torch.zeros(())})
        cmd = out["eyeout"]
        if float(torch.linalg.vector_norm(cmd)) > 0:
            c = 13.5 + float(cmd[0]) * 13.5
            r = 13.5 - float(cmd[1]) * 13.5
            eye = [eye[0] + (r - 13.5), eye[1] + (c - 13.5)]
            fired.append((round(eye[0]), round(eye[1])))
    return fired


def test_without_ior_goes_back_to_the_same_peak():
    box = Orienteye(use_ior=False)
    fired = run(box, two_peaks(), 3 * REFRACTORY_TICKS + 1)
    # 右の山へ跳んだあとは、真ん中に山があるので撃たない（段階2の挙動）
    assert [c for _, c in fired] == [19]


def test_ior_moves_on_to_the_second_peak():
    box = Orienteye(use_ior=True)
    fired = run(box, two_peaks(), 2 * REFRACTORY_TICKS + 1)
    cols = [c for _, c in fired]
    assert cols[:2] == [19, 8]          # 見た所には印があるので、2番目の山へ移る


def test_ior_fades_and_allows_return():
    box = Orienteye(use_ior=True)
    fired = run(box, two_peaks(), int(6 * IOR_DECAY_S / 0.1))
    cols = [c for _, c in fired]
    assert cols[:2] == [19, 8]
    assert len(cols) >= 3 and cols[2] == 19     # 印が薄れて、右の山へ戻れる


def test_mark_shifts_with_the_eye():
    box = Orienteye(use_ior=True)
    box._mark_fire(13, 19)            # 右へ約6マス跳んだ：印は真ん中
    r, c = divmod(int(torch.argmax(box.ior)), GRID)
    assert (r, c) in {(13, 13), (13, 14), (14, 13), (14, 14)}
    box._mark_fire(13, 8)             # 左へ約6マス跳んだ：前の印は右へ約6マスずれる
    right = box.ior.clone()
    right[:, :18] = 0
    r, c = divmod(int(torch.argmax(right)), GRID)
    assert 18 <= c <= 21 and float(right.max()) > 0.9


def test_shift_map():
    m = torch.zeros(GRID, GRID, dtype=DTYPE)
    m[5, 5] = 1.0
    s = shift_map(m, 2, -3)
    assert float(s[7, 2]) == 1.0 and float(s.sum()) == 1.0
    assert float(shift_map(m, 0, -10).sum()) == 0.0


def run_moving(box, ticks, speed=0.5):
    """世界に、動く目立つ物（値1.0、行13を右へ speed マス/回）と、止まった弱い物（値0.4、行13・列6）。
    物ごとの (荷札, 地図) も目の入口から届く形で渡す。返り値は撃った先の物の名前の列。"""
    from core_v2.boxes.orienteye import bump_at
    eye = [13.5, 13.5]
    fired = []
    for t in range(ticks):
        mover = (13.0, 17.0 + speed * t)
        if mover[1] > 22:                       # 端まで行ったら戻る（往復）
            mover = (13.0, 22.0 - (mover[1] - 22.0) % 10)
        still = (13.0, 6.0)
        to_ret = lambda p: (p[0] - eye[0] + 13.5, p[1] - eye[1] + 13.5)
        mr, sr = to_ret(mover), to_ret(still)
        prio = torch.maximum(bump_at(*mr, sigma=1.0), 0.4 * bump_at(*sr, sigma=1.0))
        objs = [(torch.ones(64), bump_at(*mr, sigma=1.0)), (torch.ones(64), bump_at(*sr, sigma=1.0))]
        out = box.step({"prio": prio, "gate3": torch.zeros(()), "eyein": objs})
        cmd = out["eyeout"]
        if float(torch.linalg.vector_norm(cmd)) > 0:
            dc, dr = float(cmd[0]) * 13.5, -float(cmd[1]) * 13.5
            eye = [eye[0] + dr, eye[1] + dc]
            dm = abs(eye[1] - mover[1]) + abs(eye[0] - mover[0])
            ds = abs(eye[1] - still[1]) + abs(eye[0] - still[0])
            fired.append("mover" if dm < ds else "still")
    return fired


def test_mark_follows_a_moving_object():
    """動く物は印の外へ逃げない：見たあとは、止まった弱い物の方へ移る。"""
    fired = run_moving(Orienteye(use_ior=True), 3 * REFRACTORY_TICKS + 1)
    assert fired[0] == "mover"
    assert "still" in fired[1:3]


def test_mark_stays_on_target_when_the_eye_undershoots():
    """目の跳びが指令より小さくても（0.6倍）、目の動きの地図（実際に目が動いた分）で印をずらすので、
    印は撃った先の物に残る。まわりに机のマスのような小さな物があっても移らない
    （段階3①の診断と作り直し、2026-10-16）。目の動きの地図は、目の入口と同じく1回遅れで届く。"""
    from core_v2.boxes.eyein import ego_map
    from core_v2.boxes.orienteye import bump_at
    world = {"target": (13.0, 20.0), "other": (13.0, 7.0),
             "clutter1": (15.0, 17.0), "clutter2": (11.0, 10.0), "clutter3": (17.0, 12.0)}
    eye = [13.5, 13.5]
    prev_eye = list(eye)
    box = Orienteye(use_ior=True)
    pending = None                            # 目の入口の出力は次の回に届く

    def step():
        nonlocal pending, prev_eye
        ret = {k: (p[0] - eye[0] + 13.5, p[1] - eye[1] + 13.5) for k, p in world.items()}
        # 画面の中身のずれ（画素）＝目の動きの逆向き×8画素/マス
        dx_px, dy_px = -(eye[1] - prev_eye[1]) * 8.0, -(eye[0] - prev_eye[0]) * 8.0
        prev_eye = list(eye)
        now = [(torch.ones(64), bump_at(*p, sigma=1.0)) for p in ret.values()]
        now.append((torch.zeros(64), ego_map(dx_px, dy_px)))
        delivered, pending = pending, now
        prio = torch.maximum(bump_at(*ret["target"], sigma=1.0), 0.6 * bump_at(*ret["other"], sigma=1.0))
        return box.step({"prio": prio, "gate3": torch.zeros(()), "eyein": delivered}), ret

    box.last_fire = 0                        # 最初の2回は撃たない
    step()
    step()
    box.last_fire = -100
    out, _ = step()
    cmd = out["eyeout"]
    assert float(torch.linalg.vector_norm(cmd)) > 0     # 目立つ target へ撃つ
    eye[1] += 0.6 * float(cmd[0]) * 13.5                # 指令の 0.6 倍しか動かない
    for _ in range(4):
        _, ret = step()
    tr, tc = ret["target"]
    assert float(box.ior[round(tr), round(tc)]) > 0.9   # 着いた物に印が残る
    cr, cc = ret["clutter1"]
    assert float(box.ior[round(cr), round(cc)]) < 0.5   # 近くの小さな物へは移らない


def test_ego_map_shifts_marks_by_the_real_eye_motion():
    """目の動きの地図を読むと、印は画面の中身と同じだけずれる（指令の分ではない）。"""
    from core_v2.boxes.eyein import ego_map
    box = Orienteye(use_ior=True)
    box.marks = [[13.0, 13.0, 1.0]]
    box._track([(torch.zeros(64), ego_map(-40.0, 16.0))])   # 中身が左へ5マス・下へ2マス
    assert abs(box.marks[0][0] - 15.0) < 0.6 and abs(box.marks[0][1] - 8.0) < 0.6
