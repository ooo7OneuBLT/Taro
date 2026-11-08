"""見た目の確認と、ランダムに動いた時の基準の数字。

出力（out/ 以下）:
- first_frames.png  各視点×各絵柄（練習用0・1、試し用2）の最初の1枚を P=5 と P=10 で並べた物
- similarity_knob.png / similarity_table.json  形の似方つまみ（画素の差で決める）の組と差
- random_top.mp4, random_side.mp4  ランダムに動く1ゲーム分（4回）
- random_stats.json  ランダムに動いた時の成功率、最初に踏んだ物、踏んだ回数など
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from button_world import ButtonWorld, Config
from button_world import sprites as S
from button_world.env import features_for_similarity, similarity_table

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
FONT = ImageFont.truetype(FONT_PATH, 18)
SMALL = ImageFont.truetype(FONT_PATH, 14)
VIEW_JA = {0: "上から", 1: "横から"}
TOUCH_JA = {-1: "なし", 0: "効くボタン", 1: "効かないボタン", 2: "ボタンでない物"}
COLOR_JA = ["赤", "橙", "黄", "緑", "青", "紫"]


def big(img, size):
    return Image.fromarray(np.asarray(img)).resize((size, size), Image.NEAREST)


def feat_text(f):
    return "・".join(lst[v] for lst, v in zip((S.OUTERS, S.RINGS, S.HEIGHTS, S.MARKS), f))


def first_frames():
    size, pad, head = 240, 10, 28
    cols = [(v, t) for v in range(2) for t in range(3)]
    canvas = Image.new("RGB", (70 + len(cols) * (size + pad) + pad, 2 * (size + pad + head) + pad), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    for r, P in enumerate((5, 10)):
        d.text((pad, pad + r * (size + pad + head) + head + size // 2), f"P={P}\n{9 * P}×{9 * P}", font=FONT, fill=(0, 0, 0))
        for c, (v, t) in enumerate(cols):
            env = ButtonWorld(Config(P=P, views=(v,), themes=((t,), (t,))))
            obs, _ = env.reset(jax.random.PRNGKey(3 + c))
            x = 70 + pad + c * (size + pad)
            y = pad + r * (size + pad + head)
            d.text((x, y), f"{VIEW_JA[v]}・絵柄{t}" + ("（試し用）" if t == 2 else ""), font=FONT, fill=(0, 0, 0))
            canvas.paste(big(obs, size), (x, y + head))
    canvas.save(os.path.join(OUT, "first_frames.png"))


def similarity_knob():
    cfg = Config()
    tr, far = cfg.train_features, cfg.far_features
    rows, d0 = similarity_table(tr, far)
    levels = [1.0, 0.75, 0.5, 0.25, 0.0]
    table = []
    for sim in levels:
        f = features_for_similarity(tr, far, sim)
        d = [r["diff"] for r in rows if r["features"] == f][0]
        table.append(dict(つまみ=f"{int(sim * 100)}%", 特徴=feat_text(f), 同じ特徴の数=sum(a == b for a, b in zip(f, tr)), 画素の差=round(d, 2), 差から見た似方=f"{round(100 * (1 - d / d0))}%"))
    with open(os.path.join(OUT, "similarity_table.json"), "w", encoding="utf-8") as fo:
        json.dump({"練習側（横から）": feat_text(tr), "差0%の時の画素の差": round(d0, 2), "表": table, "48通りの差": [dict(特徴=feat_text(r["features"]), 画素の差=round(r["diff"], 2)) for r in sorted(rows, key=lambda r: r["diff"])]}, fo, ensure_ascii=False, indent=1)
    size, pad = 110, 10
    canvas = Image.new("RGB", (pad + 2 * (size + pad) + 420, 40 + len(levels) * (size + pad)), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 12), "練習側（横から）", font=SMALL, fill=(0, 0, 0))
    d.text((pad + size + pad, 12), "試す側（上から）", font=SMALL, fill=(0, 0, 0))
    for i, (sim, row) in enumerate(zip(levels, table)):
        env = ButtonWorld(Config(P=10, similarity=sim))
        y = 40 + i * (size + pad)
        for c, (v, th) in enumerate([(1, 0), (0, 0)]):
            vt = v * S.N_THEMES + th
            s = S.shape_index(*env.features[(v, th)])
            img = np.asarray(env.table)[S.layout()["btn0"] + ((vt * S.N_SHAPES + s) * S.N_COLORS + 2) * 2 + 0]
            canvas.paste(big(img, size), (pad + c * (size + pad), y))
        x = pad + 2 * (size + pad)
        d.text((x, y + 8), f"つまみ {row['つまみ']}（画素の差 {row['画素の差']}、差から見た似方 {row['差から見た似方']}）", font=SMALL, fill=(0, 0, 0))
        d.text((x, y + 34), f"上から: {row['特徴']}", font=SMALL, fill=(60, 60, 60))
        d.text((x, y + 56), f"同じ特徴の数: {row['同じ特徴の数']}/4", font=SMALL, fill=(60, 60, 60))
    canvas.save(os.path.join(OUT, "similarity_knob.png"))
    return table, d0


def random_video(view, name, seed=7):
    env = ButtonWorld(Config(P=5, views=(view,)))
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs, s = env.reset(k)
    frames = []

    def frame(img, text):
        canvas = Image.new("RGB", (360, 400), (25, 25, 25))
        canvas.paste(big(img, 360), (0, 0))
        ImageDraw.Draw(canvas).text((6, 366), text, font=SMALL, fill=(235, 235, 235))
        return np.asarray(canvas)

    good = COLOR_JA[int(s.game.btn_colors[s.game.good])]
    frames.append(frame(obs, f"{VIEW_JA[view]} 回1 歩0 効く色={good}"))
    ep = 0
    while True:
        key, ka, ks = jax.random.split(key, 3)
        a = int(jax.random.randint(ka, (), 0, 5))
        obs_next, s2, r, done, info = step(ks, s, a)
        txt = f"{VIEW_JA[view]} 回{ep + 1} 歩{int(info['t'])} 効く色={good} 最初={TOUCH_JA[int(info['first_touch'])]}"
        if bool(done):
            txt += "  成功" if bool(info["success"]) else "  時間切れ"
            end_state = s._replace(agent=info["agent"], door_open=info["door_open"], pressed=info["pressed"])
            for _ in range(8):
                frames.append(frame(env.render(end_state), txt))
            ep += 1
            if bool(info["new_game"]):
                break
            frames.append(frame(obs_next, f"{VIEW_JA[view]} 回{ep + 1} 歩0 効く色={good}"))
        else:
            frames.append(frame(obs_next, txt))
        s = s2
    imageio.mimsave(os.path.join(OUT, name), frames, fps=8, macro_block_size=8)
    return len(frames)


def random_stats(n_envs=4096, steps=1200):
    out = {}
    for view in (0, 1):
        env = ButtonWorld(Config(P=3, views=(view,)))
        _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(view), n_envs))

        @jax.jit
        def run(key, states):
            def body(carry, _):
                key, states = carry
                key, ka, ks = jax.random.split(key, 3)
                a = jax.random.randint(ka, (n_envs,), 0, 5)
                _, states, r, done, info = jax.vmap(env.step)(jax.random.split(ks, n_envs), states, a)
                return (key, states), (done, info["success"], info["first_touch"], r, info["n_good"], info["n_bad"], info["n_deco"], info["steps_open_to_enter"], info["door_open"])

            _, ys = jax.lax.scan(body, (key, states), None, steps)
            return ys

        done, succ, touch, r, ng, nb, nd, ote, opened = map(np.asarray, run(jax.random.PRNGKey(10 + view), states))
        n = done.sum()
        t = touch[done]
        ote_s = ote[done & succ]
        out[VIEW_JA[view]] = {
            "回数": int(n),
            "成功率": round(float(succ.sum() / n), 4),
            "扉が開いた割合": round(float(opened[done].mean()), 4),
            "最初に踏んだ物の割合": {TOUCH_JA[k]: round(float((t == k).mean()), 4) for k in (-1, 0, 1, 2)},
            "1回あたり入った回数（平均）": {"効くボタン": round(float(ng[done].mean()), 3), "効かないボタン": round(float(nb[done].mean()), 3), "ボタンでない物": round(float(nd[done].mean()), 3)},
            "成功した回の、扉が開いてから入るまでの歩数（中央値）": float(np.median(ote_s)) if len(ote_s) else None,
            "1回あたりの平均の点": round(float(r.sum() / n), 4),
        }
    with open(os.path.join(OUT, "random_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    first_frames()
    table, d0 = similarity_knob()
    print("差0%の画素の差", round(d0, 2))
    for row in table:
        print(row)
    print("frames", random_video(0, "random_top.mp4"), random_video(1, "random_side.mp4"))
    print(json.dumps(random_stats(), ensure_ascii=False, indent=1))
