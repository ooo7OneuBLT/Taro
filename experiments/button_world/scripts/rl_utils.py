"""学習で使う小さな計算（テストしやすいように ppo_vanilla.py から分けた）。"""
import jax
import jax.numpy as jnp


def compute_gae(rewards, values, cuts, last_value, gamma, lam):
    """GAE。rewards・values・cuts は (T, B)。cuts が 1 の歩の後ろで割引の和を切る。

    RL² の形では cuts を「ゲームの終わり」（new_game）にする。1ゲーム（4回）の中では、回が終わっても
    点が前の回へ流れる（前の回で効く色を確かめた行動に、後の回の点が届く）。
    """
    cuts = cuts.astype(jnp.float32)

    def body(c, x):
        adv_next, v_next = c
        r, v, d = x
        delta = r + gamma * v_next * (1 - d) - v
        adv = delta + gamma * lam * (1 - d) * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(body, (jnp.zeros_like(last_value), last_value), (rewards, values, cuts), reverse=True)
    return adv


def prev_inputs(resets, prev_a, prev_r, prev_d):
    """GRU に入れる「前の行動・前の点・前の歩で回が終わったか」。新しいゲームの始まり（resets）では消す。"""
    return (
        jnp.where(resets, -1, prev_a),
        jnp.where(resets, 0.0, prev_r),
        jnp.where(resets, 0.0, prev_d.astype(jnp.float32)),
    )
