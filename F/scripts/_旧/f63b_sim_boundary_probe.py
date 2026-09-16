# -*- coding: utf-8 -*-
"""figure_mask（object_detector.py）が出す「二値化する前の生のsim値」を、
2物体の間隔を細かく振って可視化・数値化する（2026-09-04）。

F2-63(f63_multi_object_probe.py)の「12cm離す→2件検出成功／18cm離す→1件に誤合体」を
1cm刻みで再現し、失敗の原因を mujoco のセグメンテーション描画（ピクセルごとの
正解ラベル＝どの物体/床/背景か）を使って機械的に判定する。目視・推測に頼らない。

既存の object_detector.py, f63_multi_object_probe.py は読むだけで変更しない。
figure_mask の計算式はここに複製している（生のsim値を取り出すため。式は
object_detector.py 47-59行と一致させてある）。

    .venv/Scripts/python.exe F/scripts/f63b_sim_boundary_probe.py
出力: F/logs/F2-63_複数物体/f63b_sim_XXcm.png （各間隔のヒートマップ）
      F/logs/F2-63_複数物体/f63b_sim_report.json （生の数値）
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
    """object_detector.py の figure_mask (47-59行) と同じ式。二値化する前の
    sim・seed・lo/hi/tを返す（複製。ロジックは変更していない）。"""
    v = patch_feats / (np.linalg.norm(patch_feats, axis=1, keepdims=True) + 1e-9)
    g = v.reshape(n, n, -1)
    c = n // 2
    seed = g[max(c - 1, 0):c + 1, max(c - 1, 0):c + 1].reshape(-1, g.shape[-1]).mean(axis=0)
    seed = seed / (np.linalg.norm(seed) + 1e-9)
    sim = (v @ seed).reshape(n, n)
    edge = np.concatenate([sim[0], sim[-1], sim[:, 0], sim[:, -1]])
    lo, hi = float(edge.mean()), float(sim.max())
    thresh = 0.55
    t = lo + (hi - lo) * (1.0 - thresh)
    return sim, seed, lo, hi, t


def patch_labels(seg, n=PATCH_GRID, img_size=224):
    """セグメンテーション画像(224x224, ch0=geom id)からパッチごとの多数決geom idを返す。"""
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
        return "背景(空)"
    if gid == id5:
        return "toy5(コップ)"
    if gid == id6:
        return "toy6"
    bid = m.geom_bodyid[gid]
    name = m.body(bid).name
    if name == "world":
        return "床"
    return "他:" + name


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

    # geom id を先に特定（toy5/toy6の識別用。オフセットが変わっても同じgeomなので1回でよい）
    reset_all(); place("toy5", eye + fwd * 0.29 - right * 0.12, ang5); place("toy6", eye + fwd * 0.29 + right * 0.12, ang6)
    mujoco.mj_forward(m, d)
    ren.enable_segmentation_rendering()
    ren.update_scene(d, camera="eye_left")
    seg0 = ren.render()
    gids5 = [g for g in np.unique(seg0[..., 0]) if g >= 0 and m.body(m.geom_bodyid[g]).name == "test_object5"]
    gids6 = [g for g in np.unique(seg0[..., 0]) if g >= 0 and m.body(m.geom_bodyid[g]).name == "test_object6"]
    id5 = gids5[0] if gids5 else -999
    id6 = gids6[0] if gids6 else -999
    ren.disable_segmentation_rendering()
    print("geom id: toy5=%d toy6=%d" % (id5, id6))

    # まずオリジナル(F2-63)の12cm/18cmを厳密に再現（fwd も原文どおり0.30/0.28で確認）
    print("\n--- 原文再現（fwdも原文どおり） ---")
    for off, fw in [(0.12, 0.30), (0.18, 0.28)]:
        reset_all(); place("toy5", eye + fwd * fw - right * off, ang5); place("toy6", eye + fwd * fw + right * off, ang6)
        mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
        img = ren.render().copy()
        p, n = patch_features(model, img)
        dets = detect(p, n, thresh=0.55)
        print("  offset=%.0fcm fwd=%.2fm: 検出%d件 面積比=%s" % (off * 100, fw, len(dets), [round(x["area"], 3) for x in dets]))

    # 本番の掃引：fwdは0.29で固定し、横方向の間隔だけを1cm刻みで振る（変数を1つに絞る）
    print("\n--- 掃引（fwd=0.29m固定、横オフセットのみ変化） ---")
    offsets = [round(x, 3) for x in np.arange(0.09, 0.201, 0.01)]
    results = []
    for off in offsets:
        reset_all()
        place("toy5", eye + fwd * 0.29 - right * off, ang5)
        place("toy6", eye + fwd * 0.29 + right * off, ang6)
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera="eye_left")
        img = ren.render().copy()
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera="eye_left")
        seg = ren.render()
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

        results.append({
            "offset_cm": round(off * 100, 1),
            "n_det": len(dets),
            "dets": [{"pos": [round(v, 1) for v in dd["pos"]], "area": round(dd["area"], 4)} for dd in dets],
            "lo": round(lo, 4), "hi": round(hi, 4), "t": round(t, 4),
            "seed_cls": seed_cls,
            "row_cls": row_cls, "row_sim": row_sim, "row_above_t": row_above_t,
        })
        print("%5.1fcm: 検出%d件  t=%.3f lo=%.3f hi=%.3f  seed周辺=%s" % (off * 100, len(dets), t, lo, hi, seed_cls))
        print("        中心行クラス :", row_cls)
        print("        中心行sim   :", row_sim)
        print("        閾値超え     :", ["○" if b else "" for b in row_above_t])

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].imshow(img); axes[0].set_title("入力 (横%.0fcm)" % (off * 100)); axes[0].axis("off")
        im = axes[1].imshow(sim, cmap="viridis", vmin=lo, vmax=hi)
        axes[1].set_title("simマップ(生・二値化前)")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].axhline(row, color="red", lw=0.7)
        for k in range(n):
            axes[1].text(k, row, "%.2f" % sim[row, k], color="white", fontsize=5, ha="center", va="center")
        mask = (sim >= t).astype(np.uint8)
        axes[2].imshow(mask, cmap="gray"); axes[2].set_title("二値化後 t=%.3f (検出%d件)" % (t, len(dets))); axes[2].axis("off")
        plt.tight_layout()
        fp = OUT + "/f63b_sim_%04.1fcm.png" % (off * 100)
        plt.savefig(fp, dpi=110); plt.close(fig)

    # 単体提示（1つだけ）の場合のseed・sim分布（比較用）
    reset_all(); place("toy5", eye + fwd * 0.29, ang5)
    mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
    img1 = ren.render().copy()
    p1, n1 = patch_features(model, img1)
    sim1, seed1, lo1, hi1, t1 = raw_sim(p1, n1)
    dets1 = detect(p1, n1, thresh=0.55)
    c = n1 // 2
    print("\n--- 単体提示（比較用） ---")
    print("検出%d件 t=%.3f lo=%.3f hi=%.3f max=%.3f 面積比=%s" % (
        len(dets1), t1, lo1, hi1, sim1.max(), [round(x["area"], 3) for x in dets1]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(img1); axes[0].set_title("単体提示・入力"); axes[0].axis("off")
    im = axes[1].imshow(sim1, cmap="viridis", vmin=lo1, vmax=hi1)
    axes[1].set_title("単体提示・simマップ(生)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(OUT + "/f63b_sim_単体.png", dpi=110); plt.close(fig)

    with io.open(OUT + "/f63b_sim_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "single": {"lo": lo1, "hi": hi1, "t": t1, "n_det": len(dets1),
                       "dets": [{"pos": list(dd["pos"]), "area": dd["area"]} for dd in dets1]},
            "sweep": results,
        }, f, ensure_ascii=False, indent=2)

    ren.close(); env.close()
    print("\n完了:", OUT)


if __name__ == "__main__":
    main()
