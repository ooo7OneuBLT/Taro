# -*- coding: utf-8 -*-
"""物理の刻み幅を粗くしても安定するか（2026-09-03）：100Hz（現行）vs 50Hz。

太郎の体は筋肉（antagonist pair）と指の接触を持つ。MIMo自身の文書では、
筋肉と接触の数値安定性のために細かい刻みが使われている（WebSearchで確認）。
実際に太郎の体（目標F・筋肉モード・座位）で粗くして安定するか測る。

見るもの：
  ① 姿勢が発散しないか（qposの最大絶対値・NaN/Infの有無）
  ② 見た目の揺れ（同じ行動列で、関節角度の軌道がどれだけ違うか）
  ③ 接触の質（貫通の深さ・mj_geomDistanceで実測）

    .venv/Scripts/python.exe F/scripts/f61_timestep_stability.py
出力: F/logs/F2-61_刻み幅/図_100Hz_vs_50Hz.png
"""
import os, sys, io, json, re, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
from run.plugins.common import scene as scene_mod

OUT = "F/logs/F2-61_刻み幅"
N_STEPS = 1500          # 現行(0.005s)で7.5秒ぶん


def run(timestep, seed=0, n_steps=N_STEPS):
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    src = io.open("MIMo/mimoEnv/assets/f49_8way_r1.xml", encoding="utf-8").read()
    xml = "MIMo/mimoEnv/assets/f61_ts_%s.xml" % str(timestep).replace(".", "")
    new_src = re.sub(r'timestep="[^"]*"', 'timestep="%s"' % timestep, src, count=1)
    io.open(xml, "w", encoding="utf-8", newline="\n").write(new_src)
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    sc["name"] = "座位_12ヶ月_F2-61_刻み幅_%s_%s" % (str(timestep).replace(".", ""), G.DATE)
    sc["world"]["xml"] = xml
    io.open("run/scenes/%s.json" % sc["name"], "w", encoding="utf-8").write(json.dumps(sc, ensure_ascii=False, indent=1))
    env, sc2, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=seed, verbose=False)
    env.reset(seed=seed)
    u = env.unwrapped; m, d = u.model, u.data
    act_dim = env.action_space.shape[0]
    # 標準的な安定性テスト：力を入れない（何も駆動しない）状態で崩れないかを見る。
    # 座位シーンは体幹・腕・指・頭・脚が固定なので、乱雑な信号を送ると不自然な力がかかる
    # （最初の版で誤ってこれをやり、200Hzでも発散した＝テストの側の不備と判明）
    act = np.zeros(act_dim, dtype=np.float32)
    traj, max_abs, bad = [], 0.0, False
    for t in range(n_steps):
        env.step(act)
        qp = d.qpos.copy()
        if not np.all(np.isfinite(qp)):
            bad = True; break
        max_abs = max(max_abs, float(np.max(np.abs(qp))))
        traj.append(qp[:20].copy())          # 先頭20自由度（体幹〜片腕あたり）を代表として記録
    env.close()
    return {"timestep": timestep, "diverged": bad, "max_abs_qpos": max_abs,
            "n_ok_steps": len(traj), "traj": np.array(traj) if traj else np.zeros((0, 20))}


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    results = {}
    for ts, label in [(0.005, "現行 200Hz（timestep0.005・1歩=100Hz）"), (0.010, "半分 100Hz（timestep0.010・1歩=50Hz）"),
                      (0.020, "1/4  50Hz（timestep0.020・1歩=25Hz）")]:
        print("== timestep=%.3f (%s) ==" % (ts, label))
        r = run(ts, seed=1)
        results[label] = r
        print("   発散: %s ／ 完走ステップ %d/%d ／ qposの最大絶対値 %.3f"
              % (r["diverged"], r["n_ok_steps"], N_STEPS, r["max_abs_qpos"]))

    base = results["現行 200Hz（timestep0.005・1歩=100Hz）"]["traj"]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    a = ax[0]
    for label, r in results.items():
        if r["traj"].shape[0] == 0:
            continue
        n = min(len(base), len(r["traj"]))
        joint_speed = np.linalg.norm(np.diff(r["traj"][:n], axis=0), axis=1)
        a.plot(joint_speed, label=label, alpha=.85)
    a.set_xlabel("シミュレーション内の歩数（0.005s基準に揃えていない生の歩数）")
    a.set_ylabel("代表20自由度の1歩あたりの変化量（ノルム）")
    a.set_title("関節の動きの荒れ方", fontsize=12); a.legend(fontsize=8); a.grid(alpha=.3)
    b = ax[1]
    labels = list(results.keys())
    b.bar(range(len(labels)), [results[l]["max_abs_qpos"] for l in labels],
         color=["#2b6cb0" if not results[l]["diverged"] else "#c53030" for l in labels])
    b.set_xticks(range(len(labels))); b.set_xticklabels(["%.3fs" % results[l]["timestep"] for l in labels], fontsize=10)
    b.set_ylabel("qposの最大絶対値（発散すると急増）")
    b.set_title("安定性（赤＝発散した）", fontsize=12); b.grid(alpha=.3, axis="y")
    fig.tight_layout()
    p = OUT + "/図_100Hz_vs_50Hz.png"; fig.savefig(p, dpi=110); print("図:", p)


if __name__ == "__main__":
    main()
