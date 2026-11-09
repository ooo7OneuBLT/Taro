"""③の確かめ: 押したら終わりで外れを押した時に見せる1枚に、押したボタンの上の自分が写っているか（上から・横から）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from button_world import ButtonWorld, Config

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
font = ImageFont.truetype("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", 16)
rows = []
for view in (0, 1):
    env = ButtonWorld(Config(P=10, wrong_button="end_episode", views=(view,), layout_pool=512))
    step = jax.jit(env.step)
    found = None
    for seed in range(200):
        obs, s = env.reset(jax.random.PRNGKey(seed))
        key = jax.random.PRNGKey(1000 + seed)
        for t in range(60):
            key, ka, ks = jax.random.split(key, 3)
            o2, s2, r, d, info = step(ks, s, jax.random.randint(ka, (), 0, 5))
            if bool(info["ending"]):
                found = (obs, o2, int(info["entered_button"]))
                break
            if bool(d):
                break
            obs, s = o2, s2
        if found:
            break
    before, shown, b = found
    rows.append((view, np.asarray(before), np.asarray(shown)))
size = 360
canvas = Image.new("RGB", (2 * size + 30, 2 * (size + 30) + 10), (245, 245, 245))
d = ImageDraw.Draw(canvas)
for i, (view, before, shown) in enumerate(rows):
    y = 10 + i * (size + 30)
    name = "上から" if view == 0 else "横から"
    d.text((10, y), f"{name}: 押す1歩前", font=font, fill=(0, 0, 0))
    d.text((size + 20, y), f"{name}: 外れを押した瞬間（この1枚を見せてから終わる）", font=font, fill=(0, 0, 0))
    canvas.paste(Image.fromarray(before).resize((size, size), Image.NEAREST), (10, y + 22))
    canvas.paste(Image.fromarray(shown).resize((size, size), Image.NEAREST), (size + 20, y + 22))
canvas.save(os.path.join(out, "wrong_frame_check.png"))
print("saved")
