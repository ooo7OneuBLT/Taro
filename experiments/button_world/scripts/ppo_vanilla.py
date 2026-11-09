"""工夫なしのバニラモデルで、ボタンのゲームを学ばせる（動作テストと速さの計測）。評価もこのファイルで行う。

モデル（RL² / Wang 2018 の標準の形）:
  画面 → 小さな CNN（3層、32ch）→ Dense(256)。前の行動（one-hot）と前の報酬をつないで GRU(256) → 行動と価値。標準の PPO。
- 記憶（GRU）と「前の行動・前の報酬」は、新しいゲームの始まり（info["new_game"]）でだけ消す。同じゲームの4回の間は持ち越す。
- 価値の見積もりは回（エピソード）の終わりで切る。

学習: python ppo_vanilla.py --minutes 18 --out <出力先>
評価: python ppo_vanilla.py --eval <保存したモデル> --themes 2 --out <出力先>
"""
import argparse
import csv
import functools
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_bytes, to_bytes
from flax.training.train_state import TrainState

from button_world import ButtonWorld, Config

sys.path.insert(0, os.path.dirname(__file__))
from rl_utils import compute_gae, prev_inputs  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n_envs", type=int, default=1024)
p.add_argument("--n_steps", type=int, default=256, help="1回の rollout の歩数＝記憶を通して勾配が流れる窓（2026-11-09 に 64→256）")
p.add_argument("--P", type=int, default=5)
p.add_argument("--minutes", type=float, default=18.0, help="この時間で止める（最初の jit を除く）")
p.add_argument("--max_updates", type=int, default=100000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--mb_envs", type=int, default=0, help="1回の勾配計算で使う並列の数。0 なら 4096÷n_steps（1回の勾配計算の画面の枚数を約4096にそろえる）")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save_every_min", type=float, default=5.0)
p.add_argument("--eval", default="", help="保存したモデルを評価する（学習しない）")
p.add_argument("--themes", type=int, default=-1, help="評価で使う絵柄（-1 なら練習用 0・1）")
p.add_argument("--eval_steps", type=int, default=600)
p.add_argument("--out", required=True)
p.add_argument("--preset", default="", help="ゲームの設定の組（例 train_v1）。空なら初期値")
p.add_argument("--views", default="0,1", help="学習に使う視点（0 上から、1 横から）")
p.add_argument("--total_steps", type=float, default=0, help="この歩数で学習を終える。学習率もここで 0 になるよう直線で下げる（0 なら一定・時間で止める）")
args = p.parse_args()

GAMMA, LAMBDA, CLIP, ENT, VF, MAX_GRAD, EPOCHS = 0.99, 0.95, 0.2, 0.01, 0.5, 0.5, 4
MB_ENVS = args.mb_envs or max(1, 4096 // args.n_steps)
MINIBATCHES = max(1, args.n_envs // MB_ENVS)
N_ACT = 5
os.makedirs(args.out, exist_ok=True)


class ScannedGRU(nn.Module):
    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, None], jnp.zeros_like(carry), carry)
        carry, y = nn.GRUCell(features=carry.shape[-1])(carry, ins)
        return carry, y


class Net(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, h, obs, prev_a, prev_r, prev_d, resets):
        """obs (T,B,H,W,3) uint8、prev_a (T,B) int（-1 は無し）、prev_r (T,B)、prev_d (T,B) 前の歩で回が終わったか、resets (T,B)。"""
        T, B = obs.shape[:2]
        x = obs.reshape((T * B,) + obs.shape[2:]).astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2))(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2))(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=(1, 1))(x))
        x = x.reshape((T * B, -1))
        x = nn.relu(nn.Dense(self.hidden)(x)).reshape(T, B, self.hidden)
        a_oh = jax.nn.one_hot(prev_a, N_ACT)  # -1 は全部 0
        x = jnp.concatenate([x, a_oh, prev_r[..., None], prev_d[..., None]], axis=-1)
        h, y = ScannedGRU()(h, (x, resets))
        logits = nn.Dense(N_ACT, kernel_init=nn.initializers.orthogonal(0.01))(y)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(y)[..., 0]
        return h, logits, value


themes = ((0, 1), (0, 1)) if args.themes < 0 else ((args.themes,), (args.themes,))
VIEWS = tuple(int(v) for v in args.views.split(","))


def make_cfg(**kw):
    base = dict(P=args.P, themes=themes, views=VIEWS)
    base.update(kw)
    return Config.preset(args.preset, **base) if args.preset else Config(**base)


env = ButtonWorld(make_cfg())
N, T = args.n_envs, args.n_steps
net = Net(args.hidden)
rng = jax.random.PRNGKey(args.seed)
rng, k_init, k_reset = jax.random.split(rng, 3)
obs0, env_state = jax.vmap(env.reset)(jax.random.split(k_reset, N))
h0 = jnp.zeros((N, args.hidden))
pa0 = -jnp.ones((N,), jnp.int32)
pr0 = jnp.zeros((N,))
pd0 = jnp.zeros((N,))
params = net.init(k_init, h0, obs0[None], pa0[None], pr0[None], pd0[None], jnp.ones((1, N), bool))
TOTAL_UPDATES = int(args.total_steps // (N * T)) if args.total_steps else 0
if TOTAL_UPDATES:
    # 学習率を、決めた更新の回数で 0 になるよう直線で下げる（1回の更新は EPOCHS×MINIBATCHES 回の勾配計算）
    lr = optax.linear_schedule(args.lr, 0.0, TOTAL_UPDATES * EPOCHS * MINIBATCHES)
else:
    lr = args.lr
tx = optax.chain(optax.clip_by_global_norm(MAX_GRAD), optax.adam(lr, eps=1e-5))
train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
n_params = sum(x.size for x in jax.tree.leaves(params))


def rollout(params, c, n, keep_obs=True):
    """n 歩ぶん動かして、軌跡を返す。c = (env_state, obs, resets, prev_a, prev_r, prev_d, h, rng)。
    評価では keep_obs=False にして画面を記録しない（長く回すとメモリが足りなくなるため）。"""

    def env_step(c, _):
        env_state, obs, resets, pa, pr, pd, h, rng = c
        pa, pr, pd = prev_inputs(resets, pa, pr, pd)
        rng, ka, ks = jax.random.split(rng, 3)
        h, logits, value = net.apply(params, h, obs[None], pa[None], pr[None], pd[None], resets[None])
        logits, value = logits[0], value[0]
        action = jax.random.categorical(ka, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(N), action]
        obs2, env_state, reward, done, info = jax.vmap(env.step)(jax.random.split(ks, N), env_state, action)
        tr = dict(obs=obs if keep_obs else jnp.zeros((), jnp.uint8), resets=resets, prev_a=pa, prev_r=pr, prev_d=pd, action=action, logp=logp, value=value, reward=reward, done=done,
                  game_end=info["new_game"], success=info["success"], view=info["view"], ep=info["episode_in_game"], first=info["first_touch"],
                  pressed_any=(info["n_good"] + info["n_bad"]) > 0)
        return (env_state, obs2, info["new_game"], action, reward, done.astype(jnp.float32), h, rng), tr

    return jax.lax.scan(env_step, c, None, n)


def episode_metrics(tr):
    """終わった回の成功率・点・最初に踏んだ物・回ごとの成功率を、視点別に。"""
    d = tr["done"]
    out = {"episodes": d.sum()}
    for v, name in ((0, "top"), (1, "side")):
        m = d & (tr["view"] == v)
        n = jnp.maximum(m.sum(), 1)
        out[f"{name}_n"] = m.sum()
        out[f"{name}_success"] = (tr["success"] & m).sum() / n
        out[f"{name}_score"] = (tr["reward"] * m).sum() / n
        out[f"{name}_nopress"] = (m & ~tr["pressed_any"]).sum() / n  # 1つもボタンを押さずに終わった回
        for k, fname in ((0, "good"), (1, "bad"), (2, "deco"), (-1, "none")):
            out[f"{name}_first_{fname}"] = (m & (tr["first"] == k)).sum() / n
        for e in range(4):
            me = m & (tr["ep"] == e)
            n_e = jnp.maximum(me.sum(), 1)
            out[f"{name}_success_ep{e + 1}"] = (tr["success"] & me).sum() / n_e
            out[f"{name}_score_ep{e + 1}"] = (tr["reward"] * me).sum() / n_e
            btn_first = me & ((tr["first"] == 0) | (tr["first"] == 1))
            out[f"{name}_goodshare_ep{e + 1}"] = (me & (tr["first"] == 0)).sum() / jnp.maximum(btn_first.sum(), 1)
    return out


@jax.jit
def update(carry):
    train_state, env_c = carry
    h_start = env_c[6]
    env_c, traj = rollout(train_state.params, env_c, T)
    env_state, obs, resets, pa, pr, pd, h, rng = env_c
    pa_n, pr_n, pd_n = prev_inputs(resets, pa, pr, pd)
    _, _, last_value = net.apply(train_state.params, h, obs[None], pa_n[None], pr_n[None], pd_n[None], resets[None])
    last_value = last_value[0]
    # 割引の和はゲームの終わり（new_game）でだけ切る（RL² の形。回の終わりでは切らない）
    adv = compute_gae(traj["reward"], traj["value"], traj["game_end"], last_value, GAMMA, LAMBDA)
    targets = adv + traj["value"]

    def loss_fn(params, idx):
        _, logits, value = net.apply(params, h_start[idx], traj["obs"][:, idx], traj["prev_a"][:, idx], traj["prev_r"][:, idx], traj["prev_d"][:, idx], traj["resets"][:, idx])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, traj["action"][:, idx][..., None], -1)[..., 0]
        ratio = jnp.exp(logp - traj["logp"][:, idx])
        a = adv[:, idx]
        a = (a - a.mean()) / (a.std() + 1e-8)
        pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
        v_old = traj["value"][:, idx]
        v_clip = v_old + jnp.clip(value - v_old, -CLIP, CLIP)
        vl = 0.5 * jnp.maximum((value - targets[:, idx]) ** 2, (v_clip - targets[:, idx]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + VF * vl - ENT * ent, (pg, vl, ent)

    def epoch(c, _):
        ts, rng = c
        rng, kp = jax.random.split(rng)
        perm = jax.random.permutation(kp, N).reshape(MINIBATCHES, -1)

        def mb(ts, idx):
            (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(ts.params, idx)
            return ts.apply_gradients(grads=g), (l,) + aux

        ts, losses = jax.lax.scan(mb, ts, perm)
        return (ts, rng), losses

    rng, k = jax.random.split(env_c[7])
    env_c = env_c[:7] + (rng,)
    (train_state, _), losses = jax.lax.scan(epoch, (train_state, k), None, EPOCHS)
    m = episode_metrics(traj)
    m["entropy"] = losses[3].mean()
    m["value_loss"] = losses[2].mean()
    return (train_state, env_c), m


def save(params, name):
    path = os.path.join(args.out, name)
    with open(path + ".tmp", "wb") as f:
        f.write(to_bytes(jax.device_get(params)))
    os.replace(path + ".tmp", path)
    return path


env_c = (env_state, obs0, jnp.ones((N,), bool), pa0, pr0, pd0, h0, rng)

if args.eval:
    with open(args.eval, "rb") as f:
        params = from_bytes(params, f.read())
    out = {}
    for v, name in ((0, "上から"), (1, "横から")):
        ev = ButtonWorld(make_cfg(views=(v,)))
        env = ev  # rollout は env を参照する
        o, st = jax.vmap(ev.reset)(jax.random.split(jax.random.PRNGKey(100 + v), N))
        c = (st, o, jnp.ones((N,), bool), pa0, pr0, pd0, h0, jax.random.PRNGKey(200 + v))
        _, tr = jax.jit(lambda c: rollout(params, c, args.eval_steps, keep_obs=False))(c)
        m = {k: float(x) for k, x in episode_metrics(tr).items()}
        key = "top" if v == 0 else "side"
        out[name] = {
            "回数": int(m[f"{key}_n"]),
            "成功率": round(m[f"{key}_success"], 4),
            "1回の平均の点": round(m[f"{key}_score"], 4),
            "最初に踏んだ物": {lab: round(m[f"{key}_first_{k}"], 4) for lab, k in (("効くボタン", "good"), ("効かないボタン", "bad"), ("飾り", "deco"), ("なし", "none"))},
            "1つもボタンを押さずに終わった回": round(m[f"{key}_nopress"], 4),
            "ゲームの中の回ごとの成功率（1〜4回目）": [round(m[f"{key}_success_ep{e}"], 4) for e in range(1, 5)],
            "回ごとの平均の点（1〜4回目）": [round(m[f"{key}_score_ep{e}"], 4) for e in range(1, 5)],
            "最初に踏んだボタンのうち効くボタンの割合（1〜4回目）": [round(m[f"{key}_goodshare_ep{e}"], 4) for e in range(1, 5)],
        }
    out["絵柄"] = "試し用の2" if args.themes == 2 else ("練習用の0・1" if args.themes < 0 else str(args.themes))
    with open(os.path.join(args.out, f"eval_themes{args.themes}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    sys.exit(0)

steps_per_update = N * T
print(json.dumps({"n_params": int(n_params), "n_envs": N, "n_steps": T, "P": args.P, "steps_per_update": steps_per_update}), flush=True)
carry = (train_state, env_c)
t0 = time.time()
carry, m = update(carry)
jax.block_until_ready(m)
compile_s = time.time() - t0
print(f"最初の更新（jit 込み）{compile_s:.1f}秒", flush=True)

rows = []
t_start = time.time()
next_save = args.save_every_min * 60
u = 1
with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
    w = None
    max_u = TOTAL_UPDATES if TOTAL_UPDATES else args.max_updates
    while u < max_u and time.time() - t_start < args.minutes * 60:
        t1 = time.time()
        carry, m = update(carry)
        m = jax.device_get(m)
        dt = time.time() - t1
        u += 1
        row = {"update": u, "steps": u * steps_per_update, "wall_s": round(time.time() - t_start, 1), "sps": round(steps_per_update / dt)}
        row.update({k: round(float(v), 4) for k, v in m.items()})
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
        w.writerow(row)
        f.flush()
        rows.append(row)
        if time.time() - t_start > next_save:
            save(carry[0].params, f"params_{u * steps_per_update:010d}.msgpack")
            next_save += args.save_every_min * 60
        if u % 20 == 0:
            print(f"更新{u} 歩{row['steps']:,} 毎秒{row['sps']:,} 成功 上{row['top_success']:.3f} 横{row['side_success']:.3f}"
                  f" 回別(上) {row['top_success_ep1']:.2f}/{row['top_success_ep2']:.2f}/{row['top_success_ep3']:.2f}/{row['top_success_ep4']:.2f}"
                  f" 回別(横) {row['side_success_ep1']:.2f}/{row['side_success_ep2']:.2f}/{row['side_success_ep3']:.2f}/{row['side_success_ep4']:.2f}"
                  f" エントロピー{row['entropy']:.2f}", flush=True)

save(carry[0].params, "params_final.msgpack")
stats = jax.devices()[0].memory_stats() or {}
last = rows[-10:] if rows else []
summary = {
    "n_params": int(n_params),
    "n_envs": N,
    "updates": u,
    "steps": u * steps_per_update,
    "wall_s_after_jit": round(time.time() - t_start, 1),
    "compile_s": round(compile_s, 1),
    "sps_median": int(np.median([r["sps"] for r in rows])) if rows else None,
    "gpu_peak_gb": round(stats.get("peak_bytes_in_use", 0) / 2**30, 2),
    "ram_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
}
if last:
    for k in ("top_success", "side_success", "top_score", "side_score"):
        summary[f"last10_{k}"] = round(float(np.mean([r[k] for r in last])), 4)
with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False), flush=True)
