"""環境だけの速さを測る。ランダムな行動、描画込み（--no_render で描画なし）。1つの設定を1プロセスで測る。

python bench.py --n_envs 1024 --P 5
出力: 1行の JSON（毎秒の歩数、GPU メモリの最大、RAM の最大）。
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

from button_world import ButtonWorld, Config

p = argparse.ArgumentParser()
p.add_argument("--n_envs", type=int, required=True)
p.add_argument("--P", type=int, required=True)
p.add_argument("--T", type=int, default=64, help="1回の scan の歩数")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--no_render", action="store_true")
args = p.parse_args()

env = ButtonWorld(Config(P=args.P))
N, T = args.n_envs, args.T


def step_one(key, state, action):
    if args.no_render:
        # 描画を飛ばす版: render を呼ばずに状態だけ進める
        obs, s2, r, d, info = env.step(key, state, action)
        return jnp.zeros((), jnp.int32), s2, r, d
    obs, s2, r, d, info = env.step(key, state, action)
    return obs.astype(jnp.int32).sum(), s2, r, d


vstep = jax.vmap(step_one)


@jax.jit
def run(key, states):
    def body(carry, _):
        key, states, acc = carry
        key, ka, ks = jax.random.split(key, 3)
        actions = jax.random.randint(ka, (N,), 0, 5)
        chk, states, r, d = vstep(jax.random.split(ks, N), states, actions)
        return (key, states, acc + chk.sum() + r.sum().astype(jnp.int32) + d.sum()), None

    (key, states, acc), _ = jax.lax.scan(body, (key, states, jnp.zeros((), jnp.int32)), None, T)
    return key, states, acc


if args.no_render:
    # render を呼ばない step を作る（jit が描画を消せるように、観察を使わない）
    env.render = lambda state: jnp.zeros((), jnp.uint8)

key = jax.random.PRNGKey(0)
_, states = jax.vmap(env.reset)(jax.random.split(key, N))
t0 = time.time()
key, states, acc = run(key, states)
acc.block_until_ready()
compile_s = time.time() - t0
t0 = time.time()
for _ in range(args.reps):
    key, states, acc = run(key, states)
acc.block_until_ready()
dt = time.time() - t0
stats = jax.devices()[0].memory_stats() or {}
print(
    json.dumps(
        {
            "n_envs": N,
            "P": args.P,
            "render": not args.no_render,
            "obs": f"{9 * args.P}x{9 * args.P}x3",
            "steps_per_sec": round(N * T * args.reps / dt),
            "compile_s": round(compile_s, 1),
            "gpu_peak_gb": round(stats.get("peak_bytes_in_use", 0) / 2**30, 3),
            "ram_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
        }
    ),
    flush=True,
)
