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
