# -*- coding: utf-8 -*-
"""「新しさ」の対照実験（2026-09-03）：写真が高かったのはノイズのせいではないか。

F2-58 では、知っている8語（初見の個体）が 0.29、ユーザーの写真が 0.70〜0.94 だった。
だが写真は背景が雑然としていて、GSO のレンダリングは背景が単純なので、
**「知らない」ではなく「複雑」を測っている**恐れがある（ユーザーの指摘）。

対照：**同じレンダリング条件で、語彙に無い物**を見せる。
語彙に無い物＝積み木・タクシー・消防車など（8語の選定から外れたGSO素材）。

    .venv/Scripts/python.exe F/scripts/f58b_novelty_control.py
出力: F/logs/F2-58_新しさ/図_対照_語彙にない物.png
"""
import os, sys, io, json, glob, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
import f49_test as T
from f58_novelty_probe import novelty
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.vision_backends import get_backend

OUT = "F/logs/F2-58_新しさ"
# 語彙に無い物（8語の選定から外れた素材。同じ GSO・同じ描画条件）
UNKNOWN = [("つみき", "50_BLOCKS"), ("つみき", "CASTLE_BLOCKS"), ("つみき", "FAIRY_TALE_BLOCKS"),
           ("くるま", "CITY_TAXI_POLICE_CAR"), ("くるま", "FIRE_TRUCK"), ("くるま", "BABY_CAR")]
YAW = [-30, 0, 30]


def build_world(xml_out):
    """語彙に無い物を枠に差し込んだ世界。8語版と同じ手順・同じ枠。"""
    base_sc = json.load(io.open(G.BASE_SCENE, encoding="utf-8"))
    src = io.open(base_sc["world"]["xml"], encoding="utf-8").read()
    assets = []
    for k, (word, name) in enumerate(UNKNOWN):
        a, g = G.asset_and_geom("u%d" % k, "gso", name)
        assets.append(a)
        src = G.replace_body(src, G.SLOT_BODIES[k], g)
    src = src.replace("</asset>", "".join(assets) + "</asset>", 1)
    for k in range(5, 13):
        b = "test_object%d" % k
        if b not in G.SLOT_BODIES[:len(UNKNOWN)]:
            src = G.remove_body(src, b)
    io.open(xml_out, "w", encoding="utf-8", newline="\n").write(src)


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    st = torch.load("F/models/F2-49c_r3_seed93_%s.pt" % G.DATE, map_location="cpu", weights_only=False)
    protos = [np.asarray(v, dtype=np.float64) for v in st["lexicon"]["proto"].values()]

    xml = "MIMo/mimoEnv/assets/f58_unknown.xml"
    build_world(xml)
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    sc["name"] = "座位_12ヶ月_F2-58_語彙にない物_%s" % G.DATE
    sc["note"] = "新しさの対照。語彙に無い物を同じ条件で描く。学習には使わない。"
    sc["world"]["xml"] = xml
    sc["world"]["present_slots"] = G.SLOT_BODIES[1:len(UNKNOWN)]
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    cid = int(m.camera("eye_left").id)
    ren = mujoco.Renderer(m, height=224, width=224)
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    rows, imgs = [], []
    for k, (word, name) in enumerate(UNKNOWN):
        slot = G.SLOT_KEYS[k]
        for off in YAW:
            for a in qadr.values():
                d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
            mujoco.mj_forward(m, d)
            eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
            goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
            a = qadr[slot]; d.qpos[a:a + 3] = goal
            d.qpos[a + 3:a + 7] = present_quat(u, off, -20)
            mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
            img = ren.render().copy()
            vec = np.asarray(be.encode(img, img), dtype=np.float32)
            n, s = novelty(vec, protos)
            rows.append((word, name, off, n))
            if off == 0:
                imgs.append((name, img, n))
    ren.close(); env.close()
    vals = [r[3] for r in rows]
    print("語彙に無い物（同じ描画条件・%d枚）：新しさ 平均 %.3f ± %.3f" % (len(vals), np.mean(vals), np.std(vals)))
    for word, name, off, n in rows:
        if off == 0:
            print("   %-8s %-32s %.3f" % (word, name[:32], n))

    # 既知8語（F2-58の実測）と並べる
    stim = T.render_stimuli()
    known = []
    for word, idx, unseen, yaw, img, name in stim:
        vec = np.asarray(be.encode(img, img), dtype=np.float32)
        known.append(novelty(vec, protos)[0])
    photos = []
    for p in sorted(glob.glob("F/assets/user_photos/converted/*.jpg")):
        im = Image.open(p).convert("RGB"); w, h = im.size; s0 = min(w, h)
        im = im.crop(((w - s0) // 2, (h - s0) // 2, (w - s0) // 2 + s0, (h - s0) // 2 + s0)).resize((224, 224))
        arr = np.asarray(im, dtype=np.uint8)
        photos.append(novelty(np.asarray(be.encode(arr, arr), dtype=np.float32), protos)[0])
    print("既知8語 平均 %.3f ／ 語彙に無い物 平均 %.3f ／ 実写真 平均 %.3f"
          % (np.mean(known), np.mean(vals), np.mean(photos)))

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    groups = [("知っている8語\n（初見の個体）", known, "#2b6cb0"),
              ("語彙にない物\n（同じ描画条件）", vals, "#805ad5"),
              ("実写真\n（背景あり）", photos, "#c53030")]
    for i, (nm, dat, col) in enumerate(groups):
        a.scatter(np.full(len(dat), i) + np.random.RandomState(0).uniform(-.08, .08, len(dat)),
                  dat, alpha=.6, s=26, color=col)
        a.plot([i - .22, i + .22], [np.mean(dat)] * 2, color="k", lw=2)
    a.set_xticks(range(3)); a.set_xticklabels([g[0] for g in groups], fontsize=10)
    a.set_ylabel("新しさ（1 − 一番近い記憶との似方）"); a.set_ylim(0, 1.0); a.grid(alpha=.3, axis="y")
    a.set_title("背景を揃えても分かれるか（横棒は平均）", fontsize=12)
    b = ax[1]
    n = len(imgs)
    sheet = Image.new("RGB", (112 * n, 140), (255, 255, 255))
    for i, (name, img, nv) in enumerate(imgs):
        sheet.paste(Image.fromarray(img).resize((112, 112)), (112 * i, 0))
    b.imshow(np.asarray(sheet)); b.axis("off")
    b.set_title("語彙にない物（太郎の左目）\n" + "  ".join("%.2f" % v[2] for v in imgs), fontsize=10)
    fig.tight_layout()
    p = OUT + "/図_対照_語彙にない物.png"; fig.savefig(p, dpi=110); print("図:", p)


if __name__ == "__main__":
    main()
