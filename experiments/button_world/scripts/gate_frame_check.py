"""関門の前の確かめ: 押す1歩前・外れを押した瞬間・当たりを押した瞬間の画面を、上から・横からで1枚に並べる。

python gate_frame_check.py <出力の置き場> [P]
学習と同じ設定（preset gate_v1、P=5）で描き、拡大して並べる。ボタンの色が見えるかを目で確かめるため。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from button_world import ButtonWorld, Config

out = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.makedirs(out, exist_ok=True)
font = ImageFont.truetype("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", 15)
COLOR_JA = ["赤", "橙", "黄", "緑", "青", "紫"]


def find(env, want_good):
    step = jax.jit(env.step)
    for seed in range(400):
        obs, s = env.reset(jax.random.PRNGKey(seed))
        key = jax.random.PRNGKey(5000 + seed)
        for _ in range(60):
            key, ka, ks = jax.random.split(key, 3)
            o2, s2, r, d, info = step(ks, s, jax.random.randint(ka, (), 0, 5))
            eb = int(info["entered_button"])
            if eb >= 0:
                good = eb == int(s.game.good)
                if good == want_good:
                    return np.asarray(obs), np.asarray(o2), COLOR_JA[int(s.game.btn_colors[eb])]
                break
            if bool(d):
                break
            obs, s = o2, s2
    raise RuntimeError("見つからなかった")


size = 300
cols = ["押す1歩前", "外れを押した瞬間", "当たりを押した瞬間"]
canvas = Image.new("RGB", (3 * (size + 10) + 10, 2 * (size + 30) + 10), (245, 245, 245))
d = ImageDraw.Draw(canvas)
for row, view in enumerate((0, 1)):
    env = ButtonWorld(Config.preset("gate_v1", P=P, views=(view,), layout_pool=512))
    before, wrong, wcol = find(env, False)
    _, good, gcol = find(env, True)
    y = 10 + row * (size + 30)
    name = "上から" if view == 0 else "横から"
    for c, (img, label) in enumerate(((before, cols[0]), (wrong, f"{cols[1]}（{wcol}）"), (good, f"{cols[2]}（{gcol}）"))):
        x = 10 + c * (size + 10)
        d.text((x, y), f"{name}: {label}", font=font, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(img).resize((size, size), Image.NEAREST), (x, y + 20))
canvas.save(os.path.join(out, "frame_check.png"))
print("saved")
