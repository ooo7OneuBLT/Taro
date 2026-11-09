"""学習の小さな計算のテスト。pytest でも、`python tests/test_rl.py` でも回る。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import jax.numpy as jnp
import numpy as np

from rl_utils import compute_gae, prev_inputs


def test_gae_cuts_only_at_game_end():
    # 6歩、1つの並列。歩2で1回目が成功（点1）、歩5でゲームの終わり（点1）。価値はすべて0、最後の価値は5
    r = jnp.array([[0.0], [0.0], [1.0], [0.0], [0.0], [1.0]])
    v = jnp.zeros((6, 1))
    last = jnp.array([5.0])
    game_end = jnp.array([[0], [0], [0], [0], [0], [1]])
    ep_end = jnp.array([[0], [0], [1], [0], [0], [1]])
    a_game = np.asarray(compute_gae(r, v, game_end, last, 0.9, 1.0))[:, 0]
    a_ep = np.asarray(compute_gae(r, v, ep_end, last, 0.9, 1.0))[:, 0]
    # ゲームの終わりでだけ切ると、歩0の値に歩5の点が流れてくる（0.9^2 + 0.9^5）
    assert abs(a_game[0] - (0.9**2 + 0.9**5)) < 1e-5
    # 回の終わりで切ると流れない（0.9^2 だけ）
    assert abs(a_ep[0] - 0.9**2) < 1e-5
    # どちらも、ゲームの終わりの先の価値（5）は入らない
    assert abs(a_game[5] - 1.0) < 1e-5


def test_prev_done_marks_the_right_step():
    # 前の歩で回が終わった（done=True）なら、次の歩の印は 1。新しいゲームの始まりでは 0 に消す
    pa, pr, pd = prev_inputs(jnp.array([False, False, True]), jnp.array([3, 4, 2]), jnp.array([0.0, 0.9, 0.9]), jnp.array([True, False, True]))
    assert np.asarray(pd).tolist() == [1.0, 0.0, 0.0]
    assert np.asarray(pa).tolist() == [3, 4, -1]
    assert np.allclose(np.asarray(pr), [0.0, 0.9, 0.0])


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n, flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", n, repr(e), flush=True)
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
