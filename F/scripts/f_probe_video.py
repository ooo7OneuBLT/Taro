# -*- coding: utf-8 -*-
"""下見走行の視界ダンプから、ユーザー目視用の動画を作る（2026-09-03・F2-49）。

  左：第三者視点（保存した関節状態を世界XMLに戻して描き直す。体の見た目は素の体）
  右：太郎の左目の実入力（学習に使われた画像そのもの・224px）
  下：親の発話（発話イベント.csv の該当歩に字幕）

    .venv/Scripts/python.exe F/scripts/f_probe_video.py <dump_dir> <world_xml> <events_csv> <out_mp4> [fps]
"""
import os, sys, io, csv, glob, subprocess, shutil, tempfile
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 18)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 14)


def load_events(path):
    """歩 → 親の発話（複数あれば連結）。列名は 'step' と発話らしい列を探す。"""
    ev = {}
    if not path or not os.path.exists(path):
        return ev
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    if not rows:
        return ev
    cols = rows[0].keys()
    sc = next((c for c in cols if c.lower() in ("step", "tick", "歩")), None)
    tc = next((c for c in cols if c in ("text", "utterance", "発話", "セリフ", "words", "label")), None)
    wc = next((c for c in cols if c in ("speaker", "who", "話者")), None)
    if sc is None or tc is None:
        print("イベント列が見つからない:", list(cols)); return ev
    for r in rows:
        try:
            s = int(float(r[sc]))
        except Exception:
            continue
        who = (r.get(wc) or "") if wc else ""
        ev.setdefault(s, []).append(("%s:%s" % (who, r[tc])) if who else r[tc])
    return ev


def main(dump, xml, events, out, fps=10):
    if xml.endswith(".json"):
        # 実験ファイルを渡されたら、本走行と同じ手順（run/main.py と同じ build）で環境を組む。
        #   素の世界XMLだと座位の支え・体の設定が無く寝転んだ絵になる（2026-09-03 に確認）
        import json
        from run.plugins.common import scene as scene_mod
        spec = json.load(io.open(xml, encoding="utf-8"))
        env, _sc, _h = scene_mod.build(spec["scene"], taro=spec["taro"],
                                       seed=int(spec["run"].get("seed", 0)), verbose=False)
        env.reset(seed=int(spec["run"].get("seed", 0)))
        m, d = env.unwrapped.model, env.unwrapped.data
    else:
        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)     # 描画バッファが500px幅だと足りない
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
    ren = mujoco.Renderer(m, height=448, width=560)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    steps = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(dump, "step*_obs_out.eye_left.npy")))
    ev = load_events(events)
    tmp = tempfile.mkdtemp(prefix="f49video_")
    W, H = 560 + 448, 448 + 60
    recent = ""
    for i, s in enumerate(steps):
        eye = np.load(os.path.join(dump, "step%04d_obs_out.eye_left.npy" % s))
        q = np.load(os.path.join(dump, "step%04d_qpos.npy" % s))
        if q.shape[0] == m.nq:
            d.qpos[:] = q
            mp = os.path.join(dump, "step%04d_mocap_pos.npy" % s)
            if os.path.exists(mp) and m.nmocap:
                d.mocap_pos[:] = np.load(mp).reshape(m.nmocap, 3)
            rg = os.path.join(dump, "step%04d_geom_rgba.npy" % s)
            if os.path.exists(rg):
                m.geom_rgba[:] = np.load(rg).reshape(m.ngeom, 4)
            mujoco.mj_forward(m, d)
            head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
            look = d.xpos[head] if head >= 0 else d.qpos[:3]
            cam.lookat[:] = [look[0] + 0.15, look[1], look[2] - 0.05]
            cam.distance = 1.1; cam.azimuth = 150; cam.elevation = -12
            ren.update_scene(d, camera=cam)
            third = ren.render()
        else:
            third = np.zeros((448, 560, 3), np.uint8)
        canvas = Image.new("RGB", (W, H), (20, 20, 20))
        canvas.paste(Image.fromarray(third), (0, 0))
        canvas.paste(Image.fromarray(eye).resize((448, 448), Image.NEAREST), (560, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((6, 4), "第三者視点（保存状態から再描画）", fill=(255, 255, 255), font=FONT_S)
        dr.text((566, 4), "太郎の左目の実入力 224px", fill=(255, 255, 255), font=FONT_S)
        if s in ev:
            recent = "　".join(ev[s]); recent_step = s
        if recent and s - recent_step > int(1.0 * fps):     # 1秒で消す（次の物に切り替わった後に残さない）
            recent = ""
        dr.text((6, 448 + 6), "歩 %4d   %5.1f 秒" % (s, s / float(fps)), fill=(200, 200, 200), font=FONT)
        if recent:
            dr.text((220, 448 + 6), "親「%s」" % recent, fill=(255, 230, 120), font=FONT)
        canvas.save(os.path.join(tmp, "f%05d.png" % i))
    ren.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
    Image.open(os.path.join(tmp, "f%05d.png" % (len(steps) // 2))).save(out.replace(".mp4", "_代表コマ.png"))
    shutil.rmtree(tmp, ignore_errors=True)
    print("保存:", out, "コマ数", len(steps), "発話のある歩", len(ev))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 10)
