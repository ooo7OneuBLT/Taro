# -*- coding: utf-8 -*-
"""「見た目の予測誤差」案の下見（2026-09-04）：本体に組み込む前の安いテスト。

F2-58：1枚だけの新しさ（記憶との遠さ）は既知0.288・未知0.411で分布が重なった（弱い）。
F2-62：3コマを単純平均しても、既知0.495→0.456・未知0.606→0.581と両方下がるだけで
差は広がらなかった（効果なし）。

今回の案（2026-09-04夜・設計）：平均ではなく、直近の数枚をバッファに生のまま持ち、
新しい1枚が来たら「バッファの中で似ている物に重みをつけた加重平均」を予測とし、
実際とのズレ（1-コサイン類似度）を驚きとする。学習された重みは使わない。

やること：1個体につき3角度（-40, 0, +40）の連番を「見続けた列」とみなし、
最初の2枚をバッファに入れ、3枚目を予測→実際とのズレを測る。これを
既知8語（個体単位）・未知6件（積み木・車）で比べ、分布が分かれるかを見る。

ユーザーの懸念：角度を変えるだけでも見た目はけっこう変わるはずで、
「知っている物を別角度から見た」ときの驚きが、「知らない物」の驚きと
大差なくなるのでは。→この懸念どおりなら、既知の驚きも高止まりして
分布が重なるはず。それ自体を確かめる。

    .venv/Scripts/python.exe F/scripts/f67_appearance_prediction_probe.py
出力: F/logs/F2-67c_見た目予測プローブ/図_予測誤差の分かれ方.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
import f49_test as T
import f58b_novelty_control as U
from senses.object_detector import patch_features, detect

OUT = "F/logs/F2-67c_見た目予測プローブ"


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a / na) @ (b / nb))


def appearance_vec(model, img):
    """1枚の画像から見た目ベクトル（検出された塊のパッチ平均）を1本返す。無検出ならNone。"""
    p, n = patch_features(model, img)
    dets = detect(p, n, thresh=0.55)
    if not dets:
        return None
    return np.asarray(dets[0]["appearance"], dtype=np.float64)


def predict_and_surprisal(vecs_in_order):
    """先頭から最後の1枚を除いた分をバッファとし、最後の1枚を予測して驚きを返す。
    バッファが空（1枚しかない）なら驚き0とする。"""
    if len(vecs_in_order) < 2:
        return None
    buf = vecs_in_order[:-1]
    target = vecs_in_order[-1]
    obs = target / (np.linalg.norm(target) + 1e-9)
    sims = np.array([_cos(obs, b) for b in buf])
    weights = np.exp(sims) / np.sum(np.exp(sims))
    predicted = np.average(np.stack(buf, axis=0), axis=0, weights=weights)
    pred = predicted / (np.linalg.norm(predicted) + 1e-9)
    return 1.0 - float(pred @ obs)


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    # ① 既知8語：個体ごとに3角度をまとめ、角度順（-40, 0, +40）を「見続けた列」とみなす
    stim = T.render_stimuli()
    groups = {}
    for word, idx, unseen, yaw, img, name in stim:
        groups.setdefault((word, idx), []).append((yaw, img))
    known = []
    for (word, idx), items in groups.items():
        imgs = [im for _, im in sorted(items, key=lambda t: t[0])]
        vecs = [v for v in (appearance_vec(model, im) for im in imgs) if v is not None]
        s = predict_and_surprisal(vecs)
        if s is not None:
            known.append(s)
    print("既知8語（個体単位・%d件）：予測誤差 平均%.3f±%.3f"
          % (len(known), np.mean(known), np.std(known)))

    # ② 語彙に無い物：積み木・車（各3角度、-30/0/+30）
    xml = "MIMo/mimoEnv/assets/f58_unknown.xml"
    U.build_world(xml)
    r1 = "run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE
    sc = json.load(io.open(r1, encoding="utf-8"))
    sc["name"] = "座位_12ヶ月_F2-67c_見た目予測プローブ_%s" % G.DATE
    sc["note"] = "見た目の予測誤差の下見。学習には使わない。"
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
    unknown = []
    for k, (word, name) in enumerate(U.UNKNOWN):
        slot = G.SLOT_KEYS[k]
        vecs = []
        for off in U.YAW:
            for a in qadr.values():
                d.qpos[a:a + 3] = [5.0, 3.0, -2.0]
            mujoco.mj_forward(m, d)
            eye = np.array(d.cam_xpos[cid]); fwd = -np.array(d.cam_xmat[cid]).reshape(3, 3)[:, 2]
            goal = eye + fwd * dist; goal[2] = max(goal[2], 0.05)
            a = qadr[slot]; d.qpos[a:a + 3] = goal
            d.qpos[a + 3:a + 7] = present_quat(u, off, -20)
            mujoco.mj_forward(m, d); ren.update_scene(d, camera="eye_left")
            v = appearance_vec(model, ren.render().copy())
            if v is not None:
                vecs.append(v)
        s = predict_and_surprisal(vecs)
        if s is not None:
            unknown.append(s)
    ren.close(); env.close()
    print("語彙に無い物（%d件）：予測誤差 平均%.3f±%.3f"
          % (len(unknown), np.mean(unknown), np.std(unknown)))

    gap = float(min(unknown) - max(known)) if known and unknown else float("nan")
    print("分布の隙間（正なら重ならない・負なら重なる）：%.3f" % gap)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    groups_plot = [("既知\n(個体単位)", known, "#2b6cb0"), ("未知\n(積み木・車)", unknown, "#c53030")]
    for i, (nm, dat, col) in enumerate(groups_plot):
        ax.scatter(np.full(len(dat), i) + np.random.RandomState(0).uniform(-.08, .08, len(dat)),
                   dat, alpha=.7, s=40, color=col)
        ax.plot([i - .22, i + .22], [np.mean(dat)] * 2, color="k", lw=2)
    ax.set_xticks(range(2)); ax.set_xticklabels([g[0] for g in groups_plot], fontsize=11)
    ax.set_ylabel("見た目の予測誤差（1 − 予測と実際の似方）"); ax.set_ylim(0, 1.0); ax.grid(alpha=.3, axis="y")
    ax.set_title("角度違いから予測する方式は、既知/未知を分けられるか", fontsize=12)
    fig.tight_layout()
    p = OUT + "/図_予測誤差の分かれ方.png"; fig.savefig(p, dpi=110); print("図:", p)

    with io.open(OUT + "/結果.json", "w", encoding="utf-8") as f:
        json.dump({"known": known, "unknown": unknown, "gap": gap}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
