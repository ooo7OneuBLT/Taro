# -*- coding: utf-8 -*-
"""新しさの物差しは、物体ファイルで平均すると強くなるか（2026-09-04）。

F2-58：1枚だけで測ると、既知0.288・未知0.411で分布が重なる（弱い）。
今日の議論：物体ファイルは「今見ているこの物」を複数コマぶん平均できる
（`ObjectFile.appearance`のEMA）。これは既存の語彙の記憶（学習を通じた
多くの機会をまたぐ平均）とは別物で、今この瞬間の1回の注視の中での平均。

学習はやり直さない。既存の学習済みモデル（F2-49c_r3）と、既存の描画
（1個体あたり3角度・`f49_test.YAW_OFFSETS`）を「同じ物を数コマ見続けた」列として
そのまま使う。1枚目だけの新しさ（従来）と、3枚を物体ファイルで平均した後の
新しさ（今回）を比べる。

    .venv/Scripts/python.exe F/scripts/f62_novelty_via_object_files.py
出力: F/logs/F2-62_物体ファイルで平均した新しさ/図_1枚 vs 平均.png
"""
import os, sys, io, glob, warnings
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
import f58b_novelty_control as U
from f58_novelty_probe import novelty
from senses.object_detector import patch_features, detect
from brain.object_files import ObjectFileSystem

OUT = "F/logs/F2-62_物体ファイルで平均した新しさ"


def track_sequence(model, protos, imgs):
    """同じ物を写した数コマを物体ファイルに通し、(1枚目だけの新しさ, 平均後の新しさ) を返す。"""
    ofs = ObjectFileSystem(appearance_weight=0.6, max_dist=110.0, max_missed=20, reappear_gap=4, gate=0.9)
    first_novelty = None
    fid = None
    for img in imgs:
        p, n = patch_features(model, img)
        dets = detect(p, n, thresh=0.55)
        if not dets:
            continue
        r = ofs.step(dets)
        if first_novelty is None:
            v = np.asarray(dets[0]["appearance"], dtype=np.float32)
            first_novelty = novelty(v, protos)[0]
        if r["created"]:
            fid = r["created"][0]
        if r["matched"]:
            fid = r["matched"][0][0]
    if fid is None or not ofs.files:
        return first_novelty, first_novelty
    f = next((x for x in ofs.files if x.id == fid), ofs.files[-1])
    avg_novelty = novelty(f.appearance, protos)[0]
    return first_novelty, avg_novelty


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    st = torch.load("F/models/F2-49c_r3_seed93_%s.pt" % G.DATE, map_location="cpu", weights_only=False)
    protos = [np.asarray(v, dtype=np.float64) for v in st["lexicon"]["proto"].values()]
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    # ① 既知8語：個体ごとに3角度をまとめる（同じ個体＝同じ物を見続けた列として扱う）
    stim = T.render_stimuli()
    groups = {}
    for word, idx, unseen, yaw, img, name in stim:
        groups.setdefault((word, idx), []).append((yaw, img))
    known_first, known_avg = [], []
    for (word, idx), items in groups.items():
        imgs = [im for _, im in sorted(items, key=lambda t: t[0])]     # 角度の順に「見続けた」とみなす
        f1, favg = track_sequence(model, protos, imgs)
        if f1 is not None:
            known_first.append(f1); known_avg.append(favg)
    print("既知8語（個体単位・%d件）：1枚目 平均%.3f±%.3f ／ 物体ファイルで平均後 平均%.3f±%.3f"
          % (len(known_first), np.mean(known_first), np.std(known_first), np.mean(known_avg), np.std(known_avg)))

    # ② 語彙に無い物：積み木・車（各3角度）
    xml = "MIMo/mimoEnv/assets/f58_unknown.xml"
    U.build_world(xml)
    import json
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    sc["name"] = "座位_12ヶ月_F2-62_物体ファイルで平均した新しさ_%s" % G.DATE
    sc["note"] = "新しさの物差しの検証。学習には使わない。"
    sc["world"]["xml"] = xml
    sc["world"]["present_slots"] = G.SLOT_BODIES[1:len(U.UNKNOWN)]
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    from run.plugins.common import scene as scene_mod
    from f_present_angle_probe import present_quat
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    dist = float(sc["world"]["parent_labeling"].get("follow_dist", 0.3))
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    cid = int(m.camera("eye_left").id)
    ren = mujoco.Renderer(m, height=224, width=224)
    unknown_first, unknown_avg = [], []
    for k, (word, name) in enumerate(U.UNKNOWN):
        slot = G.SLOT_KEYS[k]
        imgs = []
        for off in U.YAW:
            for a in qadr.values():
                d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
            mujoco.mj_forward(m, d)
            eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
            goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
            a = qadr[slot]; d.qpos[a:a + 3] = goal
            d.qpos[a + 3:a + 7] = present_quat(u, off, -20)
            mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
            imgs.append(ren.render().copy())
        f1, favg = track_sequence(model, protos, imgs)
        if f1 is not None:
            unknown_first.append(f1); unknown_avg.append(favg)
    ren.close(); env.close()
    print("語彙に無い物（%d件）：1枚目 平均%.3f±%.3f ／ 物体ファイルで平均後 平均%.3f±%.3f"
          % (len(unknown_first), np.mean(unknown_first), np.std(unknown_first), np.mean(unknown_avg), np.std(unknown_avg)))

    def overlap(a, b):
        """2群の分布がどれだけ重なるか（既知の上端と未知の下端の差。負なら重なる）。"""
        return float(min(b) - max(a))

    gap_first = overlap(known_first, unknown_first)
    gap_avg = overlap(known_avg, unknown_avg)
    print("分布の隙間（正なら重ならない）：1枚目 %.3f ／ 平均後 %.3f" % (gap_first, gap_avg))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    groups_plot = [("既知\n1枚目", known_first, "#90b4d8"), ("既知\n平均後", known_avg, "#2b6cb0"),
                  ("未知\n1枚目", unknown_first, "#e0a5a5"), ("未知\n平均後", unknown_avg, "#c53030")]
    for i, (nm, dat, col) in enumerate(groups_plot):
        ax.scatter(np.full(len(dat), i) + np.random.RandomState(0).uniform(-.08, .08, len(dat)),
                  dat, alpha=.65, s=30, color=col)
        ax.plot([i - .22, i + .22], [np.mean(dat)] * 2, color="k", lw=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups_plot], fontsize=10)
    ax.set_ylabel("新しさ（1 − 一番近い記憶との似方）"); ax.set_ylim(0, 1.0); ax.grid(alpha=.3, axis="y")
    ax.set_title("物体ファイルで数コマぶん平均すると、新しさの物差しは強くなるか", fontsize=12)
    fig.tight_layout()
    p = OUT + "/図_1枚 vs 平均.png"; fig.savefig(p, dpi=110); print("図:", p)


if __name__ == "__main__":
    main()
