"""目の入口を軽くした「間の回は前の物を追う」の確かめ（重い視覚は作らない）。"""
import numpy as np

from core_v2.boxes.eyein import normalize_rows, scene_similarity, track

N = 16
SCALE = 224.0 / N


def grid_with(objs, seed=0):
    """背景はランダム、物 (i, j, 見た目) の所だけその見た目のマスを 2×2 置いた特徴。"""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(N * N, 384)).astype(np.float32)
    for i, j, a in objs:
        for di in (0, 1):
            for dj in (0, 1):
                f[(i + di) * N + (j + dj)] = a + 0.05 * rng.normal(size=384)
    return normalize_rows(f)


def det_at(i, j, a):
    return {"pos": ((j + 1) * SCALE, (i + 1) * SCALE), "appearance": a, "area": 4}


def test_still_object_does_not_move():
    a = np.random.default_rng(1).normal(size=384).astype(np.float32)
    f = grid_with([(6, 6, a)])
    d = det_at(6, 6, a)
    out = track([d], f, N, prev_feats=f)
    assert len(out) == 1
    assert np.allclose(out[0]["pos"], d["pos"])


def test_moving_object_is_followed():
    a = np.random.default_rng(2).normal(size=384).astype(np.float32)
    f0 = grid_with([(6, 6, a)], seed=3)
    f1 = grid_with([(6, 8, a)], seed=3)     # 右へ2マス（28画素）
    d = det_at(6, 6, a)
    out = track([d], f1, N, prev_feats=f0)
    assert len(out) == 1
    assert abs(out[0]["pos"][0] - (d["pos"][0] + 2 * SCALE)) < 3.0
    assert abs(out[0]["pos"][1] - d["pos"][1]) < 3.0


def test_lost_object_is_dropped():
    a = np.random.default_rng(4).normal(size=384).astype(np.float32)
    f0 = grid_with([(6, 6, a)], seed=5)
    f1 = grid_with([], seed=5)
    assert track([det_at(6, 6, a)], f1, N, prev_feats=f0) == []


def test_scene_jump_is_noticed():
    f0 = grid_with([], seed=6)
    f1 = grid_with([], seed=7)
    assert scene_similarity(f0, f0) > 0.99
    assert scene_similarity(f0, f1) < 0.9
