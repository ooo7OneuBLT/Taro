"""段階4：物の名簿（objfile）の確かめ。環境なしで回る（仕様書 §7 の A・B の小さな形）。"""

import math

import torch

from core_v2.boxes.eyein import ego_map
from core_v2.boxes.objfile import (KEEP_S, MAX_ROWS, MIN_LOOK_S, Objfile, bump_at)
from core_v2.formats import TAG_DIM

ZERO2 = torch.zeros(2)


def tag(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TAG_DIM, generator=g)


def objs(things, ego=(0.0, 0.0)):
    """things: [(見た目の種, 行, 列)]。最後に目の動きの地図（中身のずれ、画素）を足す。"""
    lst = [(tag(s), bump_at(r, c, sigma=1.0)) for s, r, c in things]
    lst.append((torch.zeros(TAG_DIM), ego_map(*ego)))
    return lst


def numbers(of):
    return sorted(rw.number for rw in of.rows)


def test_rows_keep_their_number_across_a_saccade():
    """目が跳んで画面の中身が左へ5マスずれても、同じ物は同じ番号のまま。"""
    of = Objfile()
    for _ in range(3):
        of.step({"eyein": objs([(1, 13, 10), (2, 13, 18)]), "orienteye": ZERO2})
    before = {rw.number: rw.pos for rw in of.rows}
    of.step({"eyein": objs([(1, 13, 5), (2, 13, 13)], ego=(-40.0, 0.0)), "orienteye": ZERO2})
    after = {rw.number: rw.pos for rw in of.rows}
    assert sorted(before) == sorted(after) and len(after) == 2
    for n in before:
        assert abs(after[n][1] - (before[n][1] - 5)) < 0.6


def test_a_missed_frame_does_not_erase_the_row():
    of = Objfile()
    for _ in range(3):
        of.step({"eyein": objs([(1, 13, 10), (2, 13, 18)]), "orienteye": ZERO2})
    n = numbers(of)
    of.step({"eyein": objs([(1, 13, 10)]), "orienteye": ZERO2})     # 2つ目を拾い損ねた
    assert numbers(of) == n
    of.step({"eyein": objs([(1, 13, 10), (2, 13, 18)]), "orienteye": ZERO2})
    assert numbers(of) == n                                         # 戻ったら同じ行に
    for _ in range(int(KEEP_S / 0.1) + 2):                          # 見えないまま KEEP_S たつと消える
        of.step({"eyein": objs([(1, 13, 10)]), "orienteye": ZERO2})
    assert len(of.rows) == 1


def shaking(of, ticks, parts, others=(), still_parts=False):
    """parts のかけらを一緒に左右に振る（1.5Hz、振幅1.2マス）。others は止まったまま。"""
    for t in range(ticks):
        dx = 0.0 if still_parts else 1.2 * math.sin(2 * math.pi * 1.5 * t * 0.1)
        things = [(s, r, c + dx) for s, r, c in parts] + list(others)
        of.step({"eyein": objs(things), "orienteye": ZERO2})


def test_split_pieces_that_move_together_become_one_row():
    """コップが胴と底の線に割れていても、親が振って一緒に動けば1行にまとまる。止まったボールはまとめない。"""
    of = Objfile()
    shaking(of, 3, [(1, 12, 14), (2, 15.5, 14)], others=[(3, 13, 17)], still_parts=True)
    assert len(of.rows) == 3
    shaking(of, 20, [(1, 12, 14), (2, 15.5, 14)], others=[(3, 13, 17)])
    assert len(of.rows) == 2 and of.n_merge == 1
    cup = [rw for rw in of.rows if len(rw.parts) == 2][0]
    want = torch.nn.functional.normalize(
        torch.nn.functional.normalize(tag(1), dim=0) + torch.nn.functional.normalize(tag(2), dim=0), dim=0)
    assert float(cup.tag @ want) > 0.99      # 見た目は2つのかけらの平均


def test_touching_pieces_are_one_row_from_the_start():
    """目が物を2つのかけらに割っても、接している（ATTACH_CELLS 以内）かけらは最初から同じ行（まとまり）。"""
    of = Objfile()
    shaking(of, 5, [(1, 12, 14), (2, 14.5, 14)], others=[(3, 13, 20)], still_parts=True)
    assert len(of.rows) == 2 and max(len(rw.parts) for rw in of.rows) == 2


def test_tracked_objects_keep_their_seats():
    """席が埋まっていると、視野の端に現れた余分な物は、見えている物の席を奪わない。
    真ん中（注意を向けた先）に現れた物は席を取る。"""
    of = Objfile()
    four = [(s, 5 + 6 * (s % 2), 3 + 6 * s) for s in range(4)]
    for _ in range(3):
        of.step({"eyein": objs(four), "orienteye": ZERO2})
    n = numbers(of)
    of.step({"eyein": objs(four + [(9, 26, 26)]), "orienteye": ZERO2})
    assert numbers(of) == n
    of.step({"eyein": objs(four + [(10, 13.5, 13.5)]), "orienteye": ZERO2})
    assert len(of.rows) == MAX_ROWS and numbers(of) != n


def test_still_neighbours_are_not_merged():
    of = Objfile()
    shaking(of, 30, [(1, 12, 14), (2, 15.5, 14)], still_parts=True)
    assert len(of.rows) == 2 and of.n_merge == 0


def test_a_mover_passing_a_still_object_is_not_merged():
    """横に動く妨害物が、止まった物のそばを通っても、まとめない（止まった物は動いていない）。"""
    of = Objfile()
    for t in range(30):
        of.step({"eyein": objs([(1, 13, 8 + 0.6 * t % 12), (2, 14, 13)]), "orienteye": ZERO2})
    assert of.n_merge == 0


def test_eye_motion_is_not_object_motion():
    """目が跳び続け、目の動きの見積もりが8割しか合っていなくても（残りのずれは全部の物に同じだけ乗る）、
    跳んだ回の動きは数えないので、止まった物どうしはまとめない（サッカード抑制）。"""
    of = Objfile()
    col = 14.0
    for t in range(30):
        d = 3.0 if t % 2 else -3.0                       # 中身が左右へ3マスずつずれる
        col += d
        of.step({"eyein": objs([(1, 12, col), (2, 15.5, col)], ego=(0.8 * 8.0 * d, 0.0)),
                 "orienteye": ZERO2})
    assert len(of.rows) == 2 and of.n_merge == 0
    import core_v2.boxes.objfile as m
    keep, m.EGO_STILL = m.EGO_STILL, 99.0                # 抑えなければ、まとめてしまう
    try:
        of = Objfile()
        col = 14.0
        for t in range(30):
            d = 3.0 if t % 2 else -3.0
            col += d
            of.step({"eyein": objs([(1, 12, col), (2, 15.5, col)], ego=(0.8 * 8.0 * d, 0.0)),
                     "orienteye": ZERO2})
        assert of.n_merge == 1
    finally:
        m.EGO_STILL = keep


def test_the_name_goes_to_the_object_looked_at_long_enough():
    """コップを1秒以上見たあと、ボールへ0.3秒だけ移っても、名前を結び付ける行はコップのまま。"""
    of = Objfile()
    for _ in range(round(MIN_LOOK_S / 0.1) + 3):           # 真ん中のコップを見る
        out = of.step({"eyein": objs([(1, 13.5, 13.5), (2, 13.5, 20)]), "orienteye": ZERO2})
    cup_tag = torch.nn.functional.normalize(tag(1), dim=0)
    assert float(out["semantic"] @ cup_tag) > 0.99
    of.step({"eyein": objs([(1, 13.5, 13.5), (2, 13.5, 20)]), "orienteye": torch.tensor([(20 - 13.5) / 13.5, 0.0])})
    for k in range(3):                                      # 目がボールへ（中身が左へ6.5マス）
        ego = (-52.0, 0.0) if k == 0 else (0.0, 0.0)
        out = of.step({"eyein": objs([(1, 13.5, 7), (2, 13.5, 13.5)], ego=ego), "orienteye": ZERO2})
    assert float(out["semantic"] @ cup_tag) > 0.99


def test_roster_is_limited():
    of = Objfile()
    of.step({"eyein": objs([(s, 4 + 4 * s, 4 + 3 * s) for s in range(6)]), "orienteye": ZERO2})
    assert len(of.rows) == MAX_ROWS
