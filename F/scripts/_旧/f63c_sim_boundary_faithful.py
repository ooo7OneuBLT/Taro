# -*- coding: utf-8 -*-
"""f63b_sim_boundary_probe.py の追加調査（2026-09-04）。

f63b で fwd(前後距離)を0.29mに固定して横オフセットだけ振ったところ、offset>=10cm
以降つねに「seed(視野中心2x2マス)が2物体の間の床の上に乗る」という別の現象が
支配的になり、原文(f63_multi_object_probe.py)の「12cm成功・18cm失敗」を再現できな
かった。原文は横オフセットと同時にfwd(前後距離)も0.30m→0.28mへ変えている
（2物体を離すのと同時に少しだけ近づけている）ため、この2変数が連動して境界を
作っている可能性がある。本スクリプトは原文と同じfwdの連動（線形補間）を保った
まま、横オフセットだけを0.5cm刻みで振り、境界を再現する。

既存の object_detector.py, f63_multi_object_probe.py, f63b_sim_boundary_probe.py は
読むだけで変更しない。

    .venv/Scripts/python.exe F/scripts/f63c_sim_boundary_faithful.py
出力: F/logs/F2-63_複数物体/f63c_sim_XXXcm.png
      F/logs/F2-63_複数物体/f63c_sim_report.json
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("taro_core/src/brain"))
sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod
from senses.object_detector import patch_features, detect, PATCH_GRID

OUT = "F/logs/F2-63_複数物体"


def raw_sim(patch_feats, n):
    v = patch_feats / (np.linalg.norm(patch_feats, axis=1, keepdims=True) + 1e-9)
    g = v.reshape(n, n, -1)
    c = n // 2
    seed = g[max(c - 1, 0):c + 1, max(c - 1, 0):c + 1].reshape(-1, g.shape[-1]).mean(axis=0)
    seed = seed / (np.linalg.norm(seed) + 1e-9)
    sim = (v @ seed).reshape(n, n)
    edge = np.concatenate([sim[0], sim[-1], sim[:, 0], sim[:, -1]])
    lo, hi = float(edge.mean()), float(sim.max())
    t = lo + (hi - lo) * (1.0 - 0.55)
    return sim, seed, lo, hi, t


def patch_labels(seg, n=PATCH_GRID, img_size=224):
    ps = img_size // n
    lab = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            block = seg[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps, 0]
            vals, counts = np.unique(block, return_counts=True)
            lab[i, j] = int(vals[np.argmax(counts)])
    return lab


def classify(gid, m, id5, id6):
    if gid == -1:
        return "空"
    if gid == id5:
        return "toy5"
    if gid == id6:
        return "toy6"
    bid = m.geom_bodyid[gid]
    return "床" if m.body(bid).name == "world" else "他"


def fwd_of(off_m):
    """原文と同じ連動：offset=0.12m→fwd=0.30m、offset=0.18m→fwd=0.28mを線形補間・外挿。"""
    return 0.30 + (off_m - 0.12) / (0.18 - 0.12) * (0.28 - 0.30)


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)
    for a in qadr.values():
        d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
    mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]; right = Rc[:, 0]
    ren = mujoco.Renderer(m, height=224, width=224)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    ang5 = sc["world"]["parent_labeling"]["present_angles"]["toy5"]
    ang6 = sc["world"]["parent_labeling"]["present_angles"]["toy6"]

    def place(slot, pos, ang):
        a = qadr[slot]; d.qpos[a:a + 3] = pos; d.qpos[a + 3:a + 7] = present_quat(u, ang["yaw"], ang["tilt"])

    def reset_all():
        for a in qadr.values():
            d.qpos[a:a + 3] = [5.0, 3.0, -2.0]

    reset_all(); place("toy5", eye + fwd * 0.30 - right * 0.12, ang5); place("toy6", eye + fwd * 0.30 + right * 0.12, ang6)
    mujoco.mj_forward(m, d)
    ren.enable_segmentation_rendering(); ren.update_scene(d, camera="eye_left"); seg0 = ren.render()
    id5 = [g for g in np.unique(seg0[..., 0]) if g >= 0 and m.body(m.geom_bodyid[g]).name == "test_object5"][0]
    id6 = [g for g in np.unique(seg0[..., 0]) if g >= 0 and m.body(m.geom_bodyid[g]).name == "test_object6"][0]
    ren.disable_segmentation_rendering()

    offsets = [round(x, 4) for x in np.arange(0.11, 0.1901, 0.005)]
    results = []
    for off in offsets:
        fw = fwd_of(off)
        reset_all(); place("toy5", eye + fwd * fw - right * off, ang5); place("toy6", eye + fwd * fw + right * off, ang6)
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        img = ren.render().copy()
        ren.enable_segmentation_rendering(); ren.update_scene(d, camera="eye_left"); seg = ren.render()
        ren.disable_segmentation_rendering()

        p, n = patch_features(model, img)
        sim, seed, lo, hi, t = raw_sim(p, n)
        dets = detect(p, n, thresh=0.55)
        lab = patch_labels(seg, n)
        cls = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                cls[i, j] = classify(lab[i, j], m, id5, id6)
        c = n // 2
        seed_cls = [cls[i, j] for i in range(max(c - 1, 0), c + 1) for j in range(max(c - 1, 0), c + 1)]
        row = c
        row_cls = [cls[row, j] for j in range(n)]
        row_sim = [round(float(sim[row, j]), 3) for j in range(n)]
        row_above_t = [bool(sim[row, j] >= t) for j in range(n)]

        # 橋渡し（床/空パッチが閾値を超えてtoy5とtoy6を連結しているか）を判定
        above = np.array(row_above_t)
        bridged = False
        # 中心行でtoy5とtoy6のパッチ位置を探す
        idx5 = [j for j, cl in enumerate(row_cls) if cl == "toy5"]
        idx6 = [j for j, cl in enumerate(row_cls) if cl == "toy6"]
        if idx5 and idx6:
            lo_j, hi_j = max(idx5), min(idx6)
            if lo_j < hi_j:
                between = above[lo_j + 1:hi_j]
                bridged = bool(between.all()) if len(between) > 0 else True

        results.append({
            "offset_cm": round(off * 100, 2), "fwd_m": round(fw, 4),
            "n_det": len(dets),
            "dets": [{"area": round(dd["area"], 4)} for dd in dets],
            "t": round(t, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "seed_cls": seed_cls, "row_cls": row_cls, "row_sim": row_sim,
            "row_above_t": row_above_t, "bridged_between_objects": bridged,
        })
        print("%5.1fcm (fwd=%.3fm): 検出%d件 t=%.3f  seed=%s  橋渡し=%s" % (
            off * 100, fw, len(dets), t, seed_cls, bridged))
        print("   行クラス:", row_cls)
        print("   行sim   :", row_sim)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(img); axes[0].set_title("入力 横%.1fcm fwd%.3fm" % (off * 100, fw)); axes[0].axis("off")
        im = axes[1].imshow(sim, cmap="viridis", vmin=lo, vmax=hi)
        axes[1].set_title("simマップ(生)"); plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].axhline(row, color="red", lw=0.7)
        mask = (sim >= t).astype(np.uint8)
        axes[2].imshow(mask, cmap="gray"); axes[2].set_title("二値化後(検出%d件)" % len(dets)); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(OUT + "/f63c_sim_%05.1fcm.png" % (off * 100), dpi=110); plt.close(fig)

    with io.open(OUT + "/f63c_sim_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    ren.close(); env.close()
    print("完了:", OUT)


if __name__ == "__main__":
    main()
