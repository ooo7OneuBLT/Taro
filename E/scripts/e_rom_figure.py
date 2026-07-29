"""関節の可動域を「絵」で見せる — どこまで動けるはずかを目で確かめる。

【なぜ要るか、2026-07-29】ユーザーの指摘：
> なんか肩の可動域とかひじの可動域がおかしかったりしないかな
> 視覚的に可動域が分からないから何とも言えないんだけど

数値（-28〜+118度）だけでは、それが人間として妥当かどうか判断できない。
⇒ 各関節を可動域の端まで動かした姿を撮って並べる。

⚠️落とし穴チェックリスト 項70「位置・姿勢の問題は数値より先に絵を撮る」。
  2026-07-28 はリクライニングの姿勢を数値だけで1時間・7回失敗し、
  真横からの画像1枚で即座に解決した。

使い方:
    .venv/Scripts/python.exe E/scripts/e_rom_figure.py
    E_SCENE=名前 で環境を変えられる
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402
import mujoco           # noqa: E402
import e_scene          # noqa: E402

SCENE = os.environ.get("E_SCENE", "リーチング_リクライニング60度")
OUT_DIR = os.path.join(_ROOT, "E", "logs", "rom")
SIZE = int(os.environ.get("E_ROM_SIZE", "420"))
SIDE = os.environ.get("E_ROM_SIDE", "right")

# 撮る関節と日本語名。人間の可動域（成人の参考値）も併記して妥当性を見る
#   ⚠️参考値は成人の一般的な臨床値で、乳児の実測ではない。目安として添える。
JOINTS = [
    ("shoulder_horizontal", "肩：水平の前後（内側へ／外側へ）", "水平内転135/外転45"),
    ("shoulder_ad_ab", "肩：開き（横へ上げる）", "外転180/内転45"),
    ("shoulder_rotation", "肩：ひねり（内旋／外旋）", "内旋70/外旋90"),
    ("elbow", "ひじ：曲げ伸ばし", "屈曲145/伸展0〜5"),
]


def shot(model, data, renderer, cam):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam)
    return renderer.render().copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from PIL import Image, ImageDraw
    scene = e_scene.load(SCENE)
    scene["fingerprint"] = None
    env, _ = e_scene.build(scene, orient=False, vor=False, seed=0, verbose=False)
    u = env.unwrapped
    m, d = u.model, u.data
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), SIZE)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), SIZE)
    q0 = d.qpos.copy()

    # ★真上から見下ろすカメラ（腕の動きが一番よく分かる向き）
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array(d.xpos[int(m.body("upper_body").id)], dtype=float)
    cam.distance = 0.75
    cam.elevation = -55.0
    cam.azimuth = 200.0
    ren = mujoco.Renderer(m, SIZE, SIZE)

    print("=" * 76)
    print(f" 関節の可動域を絵で見る（{SCENE} / {'右' if SIDE=='right' else '左'}腕）")
    print("=" * 76)
    rows = []
    for base, jp, human in JOINTS:
        jid = next((j for j in range(m.njnt)
                    if (m.joint(j).name or "").split(":")[-1] == f"{SIDE}_{base}"), None)
        if jid is None:
            continue
        qa = int(m.jnt_qposadr[jid])
        lo, hi = np.degrees(m.jnt_range[jid])
        cur = float(np.degrees(q0[qa]))
        imgs = []
        for deg in (lo, cur, hi):
            d.qpos[:] = q0
            d.qpos[qa] = np.radians(deg)
            imgs.append((deg, shot(m, d, ren, cam)))
        d.qpos[:] = q0
        rows.append((jp, human, lo, hi, cur, imgs))
        print(f"  {jp:<28} {lo:+7.1f} 〜 {hi:+7.1f} 度  （幅{hi-lo:.0f}度）"
              f"  いま{cur:+.0f}度   参考:成人 {human}")
    ren.close()
    env.close()

    # ---- 画像を並べる ----
    pad, header, label_h = 8, 46, 26
    cell = SIZE
    W = pad + (cell + pad) * 3
    H = header + sum(label_h + cell + pad for _ in rows) + pad
    canvas = Image.new("RGB", (W, H), (250, 250, 252))
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 10),
            f"関節の可動域（{'右' if SIDE=='right' else '左'}腕）"
            f"  左=最小 / 中央=いまの姿勢 / 右=最大",
            fill=(20, 20, 30))
    dr.text((pad, 28),
            "※成人の臨床参考値を併記。乳児の実測ではないので目安",
            fill=(110, 110, 120))
    y = header
    for jp, human, lo, hi, cur, imgs in rows:
        dr.text((pad, y + 4), f"{jp}   可動域 {lo:+.0f}〜{hi:+.0f}度"
                              f"（幅{hi-lo:.0f}度）   参考:成人 {human}",
                fill=(30, 30, 40))
        y += label_h
        for i, (deg, im) in enumerate(imgs):
            x = pad + i * (cell + pad)
            canvas.paste(Image.fromarray(im), (x, y))
            tag = ["最小", "いま", "最大"][i]
            dr.rectangle([x, y, x + 78, y + 18], fill=(255, 255, 255))
            dr.text((x + 4, y + 4), f"{tag} {deg:+.0f}度", fill=(180, 30, 30))
        y += cell + pad
    path = os.path.join(OUT_DIR, f"可動域_{SIDE}.png")
    canvas.save(path)
    print(f"\n  → 画像を保存: {os.path.relpath(path, _ROOT)}")
    return path


if __name__ == "__main__":
    main()
