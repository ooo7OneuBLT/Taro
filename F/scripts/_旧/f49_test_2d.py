# -*- coding: utf-8 -*-
"""F2-49 2D画像テスト（2026-09-03・ユーザー問い「3Dで学習して2D画像で答えられるか」）。

初見個体の**2D画像**（GSO同梱のサムネイル＝スキャンの描画画像・白背景。カメラ写真ではない）を
薄い板に貼り、親が持つ位置に立てて太郎の左目に見せ、f49_test と同じ物差し（GRU自力）で採点する。
人間側：15〜18ヶ月児は写真で覚えた語を実物へ転移できる（Ganea et al. 2008・逆方向も成立）。

    .venv/Scripts/python.exe F/scripts/f49_test_2d.py <model.pt> ...
出力: F/logs/F2-49_実物スキャン/test_2d/ テスト48_2D_<モデル名>.csv・図_2Dテスト48枚と答え_<モデル名>.png
"""
import os, sys, io, json, csv, warnings, glob
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco, torch
from PIL import Image, ImageDraw
import f_gen_f49 as G
import f49_test as T
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.vision_backends import get_backend

OUT = "F/logs/F2-49_実物スキャン/test_2d"
PHOTO_DIR = "F/assets/gso_photos"
BOARD_HALF = 0.075            # 板の半辺[m]（15cm角＝3Dの最長辺と同じ）
YAW_OFFSETS = (-30, 0, 30)


def photo_png(src, name, k=0):
    """サムネイル jpg → png（MuJoCoのテクスチャはPNGのみ）。YCBは写真が無いので None。"""
    if src != "gso":
        return None
    jpg = "F/assets/gso/%s/thumbnails/%d.jpg" % (name, k)
    if not os.path.exists(jpg):
        return None
    os.makedirs(PHOTO_DIR, exist_ok=True)
    png = "%s/%s_%d.png" % (PHOTO_DIR, name, k)
    if not os.path.exists(png):
        im = Image.open(jpg).convert("RGB")
        w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((512, 512))
        im.save(png)
    return os.path.abspath(png).replace(os.sep, "/")


def build_photo_world(idx, xml_out):
    base_sc = json.load(io.open(G.BASE_SCENE, encoding="utf-8"))
    src = io.open(base_sc["world"]["xml"], encoding="utf-8").read()
    assets, info = [], []
    for k, (word, s, key) in enumerate(G.WORDS):
        name, unseen = T.test_individual(s, key, idx)
        png = photo_png(s, name)
        if png:
            assets.append('<texture name="tex_p%d" type="cube" file="%s"/><material name="mat_p%d" texture="tex_p%d"/>' % (k, png, k, k))
            g = '<geom type="box" size="0.003 %.3f %.3f" material="mat_p%d" contype="0" conaffinity="0"/>' % (BOARD_HALF, BOARD_HALF, k)
        else:
            g = '<geom type="box" size="0.003 %.3f %.3f" rgba="1 1 1 1" contype="0" conaffinity="0"/>' % (BOARD_HALF, BOARD_HALF)
        info.append((word, name, unseen, png is not None))
        src = G.replace_body(src, G.SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for k in range(5, 13):
        b = "test_object%d" % k
        if b not in G.SLOT_BODIES:
            src = G.remove_body(src, b)
    io.open(xml_out, "w", encoding="utf-8", newline="\n").write(src)
    return info


def render_stimuli():
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    stim = []
    for idx in T.TEST_IDX:
        r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
        sc = json.load(io.open(r1, encoding="utf-8"))
        xml = "MIMo/mimoEnv/assets/f49_8way_photo%d.xml" % (idx + 1)
        info = build_photo_world(idx, xml)
        sc["name"] = "座位_12ヶ月_F2-49_2D写真_個体%d_%s" % (idx + 1, G.DATE)
        sc["note"] = "%s F2-49 2D画像テスト用（初見個体%dの写真を板に貼る）。学習には使わない。" % (G.DATE, idx + 1)
        sc["world"]["xml"] = xml
        io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
        env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False)
        env.reset(seed=0)
        u = env.unwrapped; m, d = u.model, u.data
        dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
        qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
        qadr["toy1"] = int(u._toy_qadr)
        cid = int(m.camera("eye_left").id)
        ren = mujoco.Renderer(m, height=224, width=224)
        for k, (word, s, key) in enumerate(G.WORDS):
            slot = G.SLOT_KEYS[k]
            name, unseen, has = info[k][1], info[k][2], info[k][3]
            if not has:
                continue
            for off in YAW_OFFSETS:
                for a in qadr.values():
                    d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
                mujoco.mj_forward(m, d)
                eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
                goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
                a = qadr[slot]
                d.qpos[a:a + 3] = goal
                d.qpos[a + 3:a + 7] = present_quat(u, off, 0)      # 板の面を太郎へ（直立）
                mujoco.mj_forward(m, d)
                ren.update_scene(d, camera="eye_left")
                stim.append((word, idx + 1, unseen, off, ren.render().copy(), name))
        ren.close(); env.close()
    return stim


def main(models):
    os.makedirs(OUT, exist_ok=True)
    stim = render_stimuli()
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    vecs = [np.asarray(be.encode(s[4], s[4]), dtype=np.float32) for s in stim]
    for mp in models:
        tag = os.path.splitext(os.path.basename(mp))[0]
        b, vp, pv = T.load_model(mp)
        rows = []
        for (word, ind, unseen, yaw, img, name), vec in zip(stim, vecs):
            said, conf = T.greedy(b, vp, pv, vec)
            rows.append({"正解": word, "個体": ind, "初見": unseen, "角度": yaw, "素材": name,
                         "GRUの発話": said, "先頭音の自信": round(conf, 2), "正解?": T.correct(said, word)})
        words = sorted({r["正解"] for r in rows}, key=lambda w: [x[0] for x in G.WORDS].index(w))
        per = {w: (sum(r["正解?"] for r in rows if r["正解"] == w), sum(r["正解"] == w for r in rows)) for w in words}
        total = sum(v[0] for v in per.values()); n = len(rows)
        with io.open("%s/テスト_2D_%s.csv" % (OUT, tag), "w", encoding="utf-8", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        T.sheet(stim, rows, "%s/図_2Dテスト%d枚と答え_%s.png" % (OUT, n, tag),
                "%s  2D画像 合計 %d/%d" % (tag, total, n))
        print("%s: 2D %d/%d  " % (tag, total, n) + "  ".join("%s %d/%d" % (w, per[w][0], per[w][1]) for w in words))


if __name__ == "__main__":
    main(sys.argv[1:] or ["F/models/F2-49_r8_seed98_%s.pt" % G.DATE])
