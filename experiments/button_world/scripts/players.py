"""手書きのプレイヤーで、効かないボタンのルールを比べる（学習はしない）。

プレイヤー（ゲームの真の状態を読み、道は最短路。押すつもりのないボタンはなるべくよけて通る）:
  0 総当たりの人: その回でまだ押していないボタンのうち、一番近いものへ行って押す。ゲームの中の記憶なし
  1 覚える人: ゲームの中で、当たりの色と外れの色を覚える。当たりが分かれば最初からそこへ。
              分からなければ、外れと分かった色を除いて一番近いボタンへ
  2 答えを知っている人（上限）: 最初から効く色へまっすぐ
  3 でたらめ（基準）
扉が開いたら、扉へ最短路で向かう。

python players.py --out /mnt/c/claude/AI/button_world/out/players
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from button_world import ButtonWorld, Config

p = argparse.ArgumentParser()
p.add_argument("--n_envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=1200)
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

H = W = 9
NC = H * W
BIG = 9999
ITERS = 64
PLAYERS = ["総当たりの人", "覚える人", "答えを知っている人", "でたらめ"]
RULES = ["none", "reset_to_start", "end_episode"]
VIEWS = {0: "上から", 1: "横から"}
CELLS = jnp.stack([jnp.arange(NC) // W, jnp.arange(NC) % W], axis=-1).astype(jnp.int32)
ACTS = jnp.arange(5)


def make_runner(env, nb, avoid):
    """avoid=True なら、押すつもりのないボタンをよけて道を探す（よけられない時は踏む）。
    ルールが none の時は外れを踏んでも損がないので、よけない（遠回りになるため）。"""
    def transitions(state):
        """T[c, a] = マス c で行動 a をした後のマス（ゲームの本当の動き。はしご・落下も含む）。"""

        def one(cell):
            st = state._replace(agent=cell)
            return jax.vmap(lambda a: env._move(st, a))(ACTS)

        nxt = jax.vmap(one)(CELLS)  # (81,5,2)
        return nxt[..., 0] * W + nxt[..., 1]

    def forward(T, cur, expandable):
        """cur からの歩数。expandable でないマス（ボタン）には入れるが、その先へは進まない。"""
        f = jnp.full((NC,), BIG).at[cur].set(0)

        def body(_, f):
            ok = expandable | (jnp.arange(NC) == cur)
            vals = jnp.broadcast_to(jnp.where(ok & (f < BIG), f + 1, BIG)[:, None], (NC, 5))
            return f.at[T.reshape(-1)].min(vals.reshape(-1))

        return jax.lax.fori_loop(0, ITERS, body, f)

    def backward(T, target, blocked):
        """各マスから target への歩数。blocked のマスには入らない（target は除く）。"""
        d = jnp.where(jnp.arange(NC) == target, 0, BIG)
        bad = blocked & (jnp.arange(NC) != target)

        def body(_, d):
            nd = d[T]  # (81,5)
            nd = jnp.where(bad[T], BIG, nd)
            best = jnp.minimum(d, jnp.min(nd, axis=1) + 1)
            best = jnp.where(bad, BIG, best)  # 入らないマスは「届かない」にする（その1歩を選ばないように）
            return jnp.where(jnp.arange(NC) == target, 0, best)

        return jax.lax.fori_loop(0, ITERS, body, d)

    def choose(state, ps, key, ptype):
        tried, known_good, known_bad = ps
        T = transitions(state)
        cur = state.agent[0] * W + state.agent[1]
        bidx = state.btn_pos[:, 0] * W + state.btn_pos[:, 1]
        colors = state.game.btn_colors
        good_mask = jnp.arange(nb) == state.game.good
        is_btn = jnp.zeros((NC,), bool).at[bidx].set(True)
        untried = ~tried
        learner_ok = untried & ~known_bad[colors] & ((known_good < 0) | (colors == known_good))
        cand = jnp.where(ptype == 0, untried, jnp.where(ptype == 1, learner_ok, good_mask))
        f = forward(T, cur, ~(is_btn & avoid))  # avoid の時だけ、ほかのボタンの先へは進まない
        dist = jnp.where(cand, f[bidx], BIG)
        pick = jnp.argmin(dist)
        has_target = dist[pick] < BIG
        door = state.door[0] * W + state.door[1]
        target = jnp.where(state.door_open, door, bidx[pick])
        reachable = state.door_open | has_target
        d = backward(T, target, is_btn & avoid)
        d_free = backward(T, target, jnp.zeros((NC,), bool))
        dd = jnp.where(d[cur] < BIG, d, d_free)
        q = dd[T[cur]]
        a_plan = jnp.argmin(q)
        a_rand = jax.random.randint(key, (), 0, 5)
        use_plan = (ptype != 3) & reachable & (q[a_plan] < BIG)
        return jnp.where(use_plan, a_plan, a_rand)

    def step_one(state, ps, gs, gt, key, ptype):
        ka, ks = jax.random.split(key)
        a = choose(state, ps, ka, ptype)
        _, nxt, r, done, info = env.step(ks, state, a)
        tried, known_good, known_bad = ps
        eb = info["entered_button"]
        hit = eb >= 0
        ebc = jnp.maximum(eb, 0)
        color = state.game.btn_colors[ebc]
        is_good = ebc == state.game.good
        tried = jnp.where(hit, tried.at[ebc].set(True), tried)
        known_good = jnp.where(hit & is_good, color, known_good)
        known_bad = jnp.where(hit & ~is_good, known_bad.at[color].set(True), known_bad)
        tried = jnp.where(done, jnp.zeros_like(tried), tried)
        newg = info["new_game"]
        known_good = jnp.where(newg, -1, known_good)
        known_bad = jnp.where(newg, jnp.zeros_like(known_bad), known_bad)
        gs = gs + jnp.where(done, r, 0.0)
        gt = gt + jnp.where(done, info["t"], 0)
        out = dict(done=done, success=info["success"], reward=r, t=info["t"], ep=info["episode_in_game"], first=info["first_touch"],
                   wrong_end=info["wrong_end"], game_done=newg, game_score=gs, game_steps=gt)
        gs = jnp.where(newg, 0.0, gs)
        gt = jnp.where(newg, 0, gt)
        return nxt, (tried, known_good, known_bad), gs, gt, out

    vstep = jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0, None))

    @jax.jit
    def run(key, ptype):
        N = args.n_envs
        k0, key = jax.random.split(key)
        _, states = jax.vmap(env.reset)(jax.random.split(k0, N))
        ps = (jnp.zeros((N, nb), bool), -jnp.ones((N,), jnp.int32), jnp.zeros((N, 6), bool))
        gs, gt = jnp.zeros((N,)), jnp.zeros((N,), jnp.int32)

        def body(c, k):
            states, ps, gs, gt = c
            states, ps, gs, gt, out = vstep(states, ps, gs, gt, jax.random.split(k, N), ptype)
            return (states, ps, gs, gt), out

        _, outs = jax.lax.scan(body, (states, ps, gs, gt), jax.random.split(key, args.steps))
        return outs

    return run


def summarize(o):
    o = {k: np.asarray(v) for k, v in o.items()}
    d = o["done"]
    res = {"回数": int(d.sum())}
    per_ep = []
    for e in range(4):
        m = d & (o["ep"] == e)
        n = max(int(m.sum()), 1)
        succ = m & o["success"]
        steps = o["t"][succ]
        btn_first = m & ((o["first"] == 0) | (o["first"] == 1))
        timeout = m & ~o["success"] & ~o["wrong_end"]
        hist = np.histogram(steps, bins=np.arange(0, 65, 5))[0].tolist() if len(steps) else []
        per_ep.append({
            "回": e + 1,
            "回数": int(m.sum()),
            "成功率": round(float(succ.sum() / n), 4),
            "点の平均": round(float(o["reward"][m].mean()) if m.any() else 0.0, 4),
            "成功した回の歩数_平均": round(float(steps.mean()), 2) if len(steps) else None,
            "成功した回の歩数_中央値": float(np.median(steps)) if len(steps) else None,
            "成功した回の歩数_分布（5歩ごと、0〜60）": hist,
            "最初に押したボタンが当たりの割合": round(float((m & (o["first"] == 0)).sum() / max(int(btn_first.sum()), 1)), 4),
            "時間切れの割合": round(float(timeout.sum() / n), 4),
            "外れで終わった割合": round(float((m & o["wrong_end"]).sum() / n), 4),
        })
    res["回ごと"] = per_ep
    g = o["game_done"]
    res["ゲーム数"] = int(g.sum())
    res["4回の合計点_平均"] = round(float(o["game_score"][g].mean()), 4) if g.any() else None
    res["4回の合計歩数_平均"] = round(float(o["game_steps"][g].mean()), 2) if g.any() else None
    return res


results = {}
timing = {}
t_all = time.time()
for nb in (3, 6):
    for rule in RULES:
        for view in (0, 1):
            env = ButtonWorld(Config(P=3, views=(view,), n_buttons=nb, wrong_button=rule))
            run = make_runner(env, nb, rule != "none")
            name = f"{rule}・ボタン{nb}個・{VIEWS[view]}"
            results[name] = {}
            for ptype, pname in enumerate(PLAYERS):
                t0 = time.time()
                outs = run(jax.random.PRNGKey(1000 + ptype), jnp.int32(ptype))
                jax.block_until_ready(outs["done"])
                dt = time.time() - t0
                results[name][pname] = summarize(outs)
                timing[f"{name}・{pname}"] = round(dt, 2)
                r = results[name][pname]
                print(name, pname, "ゲーム", r["ゲーム数"], "合計点", r["4回の合計点_平均"],
                      "回ごとの点", [x["点の平均"] for x in r["回ごと"]], f"{dt:.1f}秒", flush=True)
timing["全部"] = round(time.time() - t_all, 1)
with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
    json.dump({"結果": results, "計測にかかった秒数（各行は jit の初回を含む）": timing,
               "条件": {"並列": args.n_envs, "歩数": args.steps, "1回の長さ": 60, "1ゲームの回数": 4, "P": 3}}, f, ensure_ascii=False, indent=1)
print("全部", timing["全部"], "秒", flush=True)
