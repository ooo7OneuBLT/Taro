"""制御頻度プローブ（測定器）：**学習を一切せず**に「上位の命令間隔Kを短くすると
動きが小さくなるのか」を筋レベル・関節レベルで直接測る。

【なぜこれを作ったか、2026-07-25】
階層化Hz運動制御（[E/docs/階層化Hz運動制御設計.md]）の段階0。設計の前提が2つあり、
どちらも**学習を回さずに確かめられる**と気づいた：

  前提1「K=10だと動きが小さくなる」
    過去のK=10失敗（C研究日誌 350-356行、経験量を揃えても届かず）の原因として
    「RLの探索ノイズが0.1秒ごとに独立サンプルされて高周波化し、**力積が打ち消し合う**」
    という説が出た（arXiv:2010.04304＝DOFの多い体ほど高頻度制御が不利、と整合）。
    これが正しいなら**学習ゼロ・色付きノイズだけでも現れるはず**。
  前提2「仰向けなら姿勢保持は要らない」
    文献調査で否定された（仰臥位は姿勢筋の活動需要が最も低い[Tier1]／同時硬直性収縮は
    脳性麻痺の早期マーカー[Tier1]）。MuscleModelの**受動的筋力 F_P(L)**（活性化ゼロでも
    働く）だけで姿勢が保たれるかを実測で裏どりする。

【測るもの】★筋レベルを入れたのがこのプローブの要点
  関節だけ見ると「動きが小さい」は分かっても**なぜか**が分からない。
  「打ち消し合い」仮説を直接見るには筋レベルが要る（ユーザー指摘、2026-07-25）。

  筋レベル（MuscleModel、180次元＝拮抗筋2本/関節）
    - 同時活性化 mean(min(a[i], a[i+90]))  ← ★打ち消し合いの直接指標
    - 拮抗筋ペアの時系列相関 mean(corr(a[i], a[i+90]))
    - 総筋力 vs 正味の力 |f[i] - f[i+90]|  ← 出した力のうち何割が相殺されたか
  関節レベル
    - |関節角速度| の平均  ← ★「動きの量」の主指標
    - 関節角の可動範囲 (max-min)
    - mean|jerk|（既存の指標と接続するため）
  姿勢（前提2の確認）
    - 頭の傾き（垂直からの角度）／手が体から離れている距離

⚠️**拮抗筋ペアの並びは (i, i+90)**。MIMoの`muscle.py`が
`muscle_lengths() = concatenate([lce_1, lce_2])` と定義しており、前半90が筋1・後半90が筋2。
（(2i, 2i+1) ではない。実装前にソースで確認済み）

⚠️**公平性**：Kを変えると1tickあたりのsim時間が変わるので、**総物理ステップ数を固定**して
比較する（N_TICK = TOTAL_STEPS // K）。過去の予測幅スイープも「経験量そろえ」の軸を
持っていた（C研究日誌 353行）。

【使い方】
    python E/scripts/e_ctrl_freq_probe.py            # 4条件を順に回して比較表を出す
    E_COND=K10 python ...                             # 1条件だけ
    E_STEPS=12000 python ...                          # 物理ステップ数を変える
    E_NOVIDEO=1 python ...                            # 録画しない（速い）
    ★E_VIEW=1 E_REALTIME=1 E_COND=K100 python ...    # Viewerで等倍速ライブ再生
      （キー操作: . 速く / , 遅く / 0 等倍 / M 最速）
"""
import os
import sys
import csv
import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "taro_core"))
import paths; paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)
sys.path.insert(0, os.path.join(paths.SRC, "brain"))

import numpy as np
import mujoco
from d_supine_env import SupineMimoEnv
from mimoActuation.muscle import MuscleModel
from spinal_cord.cpg import ColoredNoiseGenerator

AGE = 0.0
BETA = float(os.environ.get("E_BETA", "0.7"))
SEED = int(os.environ.get("E_SEED", "0"))
# 総物理ステップ数を固定＝Kが違っても「同じだけ体を動かした」状態で比べる
TOTAL_STEPS = int(os.environ.get("E_STEPS", "6000"))
NOVIDEO = os.environ.get("E_NOVIDEO", "0") == "1"

LOG_DIR = os.path.join(paths.CORE_ROOT, os.pardir, "E", "logs", "E", "ctrl_freq_probe")

# 条件：(名前, 筋活性化の中心, ゆらぎ幅, K, 説明)
# ⚠️ACT_CENTERは「全筋に一定のベースライン活性化」＝構造的に共収縮に近い。
#   passiveでこれを0にすると受動的筋力(F_P)だけの素の身体が見える。
_CONDS = [
    ("passive", 0.0, 0.0, 100, "完全脱力（受動的筋力だけ）＝姿勢保持が要るかの確認"),
    ("tonic",   0.3, 0.0, 100, "一定活性化のみ（ゆらぎなし）＝固まるかの確認"),
    ("K100",    0.3, 0.3, 100, "現状の1秒ホールド（比較の基準）"),
    ("K10",     0.3, 0.3,  10, "★本命：0.1秒ごとに命令を出し直すと動きが縮むか"),
    # 2026-07-25 追加：Viewerで「両方ともすぐうつぶせになる」と分かったので振幅を振る。
    # 実際の新生児は寝返りできない（4-6ヶ月から）ので、ノイズだけでうつぶせになるのは非人間的。
    ("K10_amp15", 0.3, 0.15, 10, "振幅を半分に：うつぶせ化は振幅のせいか"),
    ("K100_amp15", 0.3, 0.15, 100, "同上のK100版（振幅の効果をKと分離する）"),
]


def _head_tilt_deg(model, data):
    """頭の傾き（頭のz軸が世界のz軸となす角、度）。仰向けで0に近いほど正面を向いている。"""
    try:
        xmat = data.body("head").xmat.reshape(3, 3)
        z_axis = xmat[:, 2]
        return float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))
    except Exception:
        return float("nan")


def _trunk_rotation_deg(data, R0):
    """★体幹が初期姿勢（仰向け）からどれだけ回転したか（度）。
    0度＝仰向けのまま、180度＝完全にうつぶせ。

    【なぜ追加したか、2026-07-25】最初の版は「頭のz軸と世界z軸の角度」しか測っておらず、
    **体全体がひっくり返ったことを捉えられなかった**。Viewerで見たユーザーが
    「姿勢はどっちも激しすぎてすぐうつぶせになってる」と指摘して発覚。
    ＝数値上は「姿勢は変わらない(84.7°vs86.6°)」と報告してしまっていた。
    [[feedback-watch-dont-just-measure]]の典型例（目視が測定器の穴を見つけた）。"""
    try:
        R = data.body("upper_body").xmat.reshape(3, 3)
        rel = R0.T @ R                       # 初期姿勢からの相対回転
        cos = (np.trace(rel) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    except Exception:
        return float("nan")


def _hand_dist(model, data):
    """両手が体幹からどれだけ離れているか（m）の平均。脱力すると床に落ちて広がる想定。"""
    try:
        trunk = data.body("upper_body").xpos
        rh = np.linalg.norm(data.body("right_hand").xpos - trunk)
        lh = np.linalg.norm(data.body("left_hand").xpos - trunk)
        return float((rh + lh) / 2.0)
    except Exception:
        return float("nan")


def _pairwise_corr(a_hist, n_joint):
    """拮抗筋ペア (i, i+n_joint) の時系列相関の平均。
    正に大きい＝両方が同時に強くなる（＝打ち消し合っている疑い）。"""
    a = np.asarray(a_hist)          # (T, 2*n_joint)
    if a.shape[0] < 3:
        return float("nan")
    cors = []
    for i in range(n_joint):
        x, y = a[:, i], a[:, i + n_joint]
        sx, sy = x.std(), y.std()
        if sx < 1e-8 or sy < 1e-8:
            continue                # 変化のない筋は相関が定義できない（tonic条件など）
        cors.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(cors)) if cors else float("nan")


def run_condition(name, act_center, act_amp, K, desc):
    n_tick = max(1, TOTAL_STEPS // K)
    print(f"\n=== {name} ===  {desc}")
    print(f"  中心={act_center} ゆらぎ={act_amp} K={K}  → {n_tick}tick × K{K} = {n_tick*K}物理step")

    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=AGE)
    m, d = env.unwrapped.model, env.unwrapped.data
    n_act = env.action_space.shape[0]          # 180
    n_joint = n_act // 2                       # 90（拮抗筋ペアは i と i+n_joint）
    act_model = getattr(env.unwrapped, "actuation_model", None)

    gen = ColoredNoiseGenerator(n_act, seed=SEED)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    dofs = [m.jnt_dofadr[i] for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

    renderer = cam = None
    frames = []
    if not NOVIDEO:
        renderer = mujoco.Renderer(m, height=480, width=480)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.2; cam.azimuth = 90; cam.elevation = -35

    obs, _ = env.reset(seed=SEED)
    # 初期姿勢（＝仰向け）の体幹の向きを基準に取る。以後ここからの回転量を測る。
    R0 = env.unwrapped.data.body("upper_body").xmat.reshape(3, 3).copy()
    jerks, qvels, tilts, hands, trunk_rots = [], [], [], [], []
    qpos_hist = []
    a_hist, coact, f_total, f_net = [], [], [], []
    prev_qacc = None
    _warned = {"muscle": False}   # 測定失敗は握りつぶさず1度だけ出す
    if act_model is None:
        print("  ⚠️actuation_model が取れない＝筋レベルの測定は全てnanになる")

    for tick in range(n_tick):
        noise = gen.sample(BETA) if act_amp > 0 else 0.0
        act = np.clip(act_center + act_amp * noise, 0.0, 1.0).astype(np.float32)
        for k in range(K):
            obs, r, te, tr, info = env.step(act)

            # --- 関節レベル ---
            qacc = d.qacc[dofs].copy()
            if prev_qacc is not None:
                jerks.append(np.abs((qacc - prev_qacc) / dt_env).mean())
            prev_qacc = qacc
            qvels.append(float(np.abs(d.qvel[dofs]).mean()))
            qpos_hist.append(d.qpos[[m.jnt_qposadr[i] for i in range(m.njnt)
                                     if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]].copy())

            # --- 筋レベル（★打ち消し合いの直接測定） ---
            # ⚠️muscle_activations / muscle_forces は @property（()を付けると TypeError）。
            #   最初の実装で () を付けたうえ except:pass で握りつぶし、全部nanになった。
            #   → 例外は握りつぶさず1度だけ表示する（[[feedback-bug-to-checklist]]）。
            if act_model is not None:
                try:
                    a = np.asarray(act_model.muscle_activations, dtype=np.float64)
                    f = np.abs(np.asarray(act_model.muscle_forces, dtype=np.float64))
                    a_hist.append(a.copy())
                    # 同時活性化＝両方の筋が同時に引いている量（引き分けて無駄になる分）
                    coact.append(float(np.minimum(a[:n_joint], a[n_joint:]).mean()))
                    f_total.append(float(f.mean()))
                    # 正味＝拮抗筋の差（実際に関節を動かせる分）
                    f_net.append(float(np.abs(f[:n_joint] - f[n_joint:]).mean()))
                except Exception as e:
                    if not _warned["muscle"]:
                        print(f"  ⚠️筋レベルの測定に失敗: {type(e).__name__}: {e}")
                        _warned["muscle"] = True

            # --- 姿勢（前提2の確認） ---
            tilts.append(_head_tilt_deg(m, d))
            hands.append(_hand_dist(m, d))
            trunk_rots.append(_trunk_rotation_deg(d, R0))   # ★うつぶせ化の検出

            if renderer is not None and k % 4 == 0:
                cam.lookat = d.body("upper_body").xpos.copy()
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())
            if te or tr:
                obs, _ = env.reset(); prev_qacc = None
                break

    qpos_arr = np.asarray(qpos_hist)
    rom = float((qpos_arr.max(axis=0) - qpos_arr.min(axis=0)).mean()) if len(qpos_arr) else float("nan")
    res = {
        "cond": name, "K": K, "act_center": act_center, "act_amp": act_amp,
        "n_step": len(qvels),
        # 関節レベル
        "qvel_mean": float(np.mean(qvels)) if qvels else float("nan"),
        "rom_mean_rad": rom,
        "jerk_mean": float(np.mean(jerks)) if jerks else float("nan"),
        # 筋レベル
        "coactivation": float(np.mean(coact)) if coact else float("nan"),
        "antag_corr": _pairwise_corr(a_hist, n_joint) if a_hist else float("nan"),
        "force_mean": float(np.mean(f_total)) if f_total else float("nan"),
        "force_net": float(np.mean(f_net)) if f_net else float("nan"),
        # 姿勢
        "head_tilt_deg": float(np.nanmean(tilts)) if tilts else float("nan"),
        "hand_dist_m": float(np.nanmean(hands)) if hands else float("nan"),
        # ★うつぶせ化：0度=仰向けのまま、180度=完全にうつぶせ
        "trunk_rot_mean_deg": float(np.nanmean(trunk_rots)) if trunk_rots else float("nan"),
        "trunk_rot_max_deg": float(np.nanmax(trunk_rots)) if trunk_rots else float("nan"),
        # 90度を超えた時間の割合＝「仰向けでなくなっていた」割合
        "prone_frac": (float(np.mean(np.asarray(trunk_rots) > 90.0)) if trunk_rots else float("nan")),
    }
    # 出した力のうち実際に関節を動かせた割合（低いほど打ち消し合っている）
    res["force_efficiency"] = (res["force_net"] / res["force_mean"]
                               if res["force_mean"] and res["force_mean"] > 1e-12 else float("nan"))

    print(f"  |qvel|={res['qvel_mean']:.4f}  可動域={res['rom_mean_rad']:.4f}rad  "
          f"jerk={res['jerk_mean']:.1f}")
    print(f"  同時活性化={res['coactivation']:.4f}  拮抗筋相関={res['antag_corr']:+.3f}  "
          f"力の効率={res['force_efficiency']:.3f}")
    print(f"  頭の傾き={res['head_tilt_deg']:.1f}度  手-体幹={res['hand_dist_m']:.3f}m")
    print(f"  ★体幹の回転={res['trunk_rot_mean_deg']:.1f}度(最大{res['trunk_rot_max_deg']:.1f})  "
          f"仰向けでない時間={res['prone_frac']*100:.1f}%")

    if frames:
        os.makedirs(LOG_DIR, exist_ok=True)
        out_mp4 = os.path.join(LOG_DIR, f"ctrl_freq_{name}.mp4")
        try:
            import cv2
            fps = (1.0 / dt_env) / 4
            hh, ww, _ = frames[0].shape
            vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"  RECORDED {out_mp4} ({len(frames)}フレーム, {fps:.0f}fps)")
        except Exception as e:
            print("  録画失敗", type(e).__name__, e)

    env.close()
    return res


def view_condition(name, act_center, act_amp, K, desc):
    """★Viewerで等倍速ライブ再生する（[[feedback-both-view-videos]]＝確認は原則Viewer）。
    共通Viewer `taro_core/tools/motor_viewer.py` をそのまま使う（Viewerを書き直さない、
    [[feedback-core-target-neutral-naming]]の精神）。学習した脳は使わないので、
    共通Viewerが要求する brain / policy_fn は色付きノイズを出すだけのダミーで満たす。
    """
    sys.path.insert(0, os.path.join(paths.CORE_ROOT, "tools"))
    import torch
    from motor_viewer import run_viewer

    env = SupineMimoEnv(actuation_model=MuscleModel, vision_params=None, age=AGE)
    n_act = env.action_space.shape[0]
    gen = ColoredNoiseGenerator(n_act, seed=SEED)

    class _NoiseOnly:
        """共通Viewerのインターフェースを満たすだけの入れ物（脳ではない）。"""
        def init_motor_hidden(self):
            return None

    def policy_fn(obs, prev_a, hidden, *, recompute, frac):
        # recompute=True＝命令の境界。ここでだけ新しい筋活性化を決める＝K物理stepに1回。
        if not recompute:
            return prev_a, hidden
        noise = gen.sample(BETA) if act_amp > 0 else 0.0
        a = np.clip(act_center + act_amp * noise, 0.0, 1.0).astype(np.float32)
        return torch.as_tensor(a), hidden

    def rescale_action(a, action_space):
        # 筋活性化は既に[0,1]＝環境のctrlそのもの。変換不要。
        return np.asarray(a, dtype=np.float32)

    banner = f"{name}: {desc}  (中心={act_center} ゆらぎ={act_amp} K={K})"
    run_viewer(env, _NoiseOnly(), policy_fn, rescale_action, K=K, n_act=n_act, banner=banner)
    env.close()


def main():
    only = os.environ.get("E_COND", "")
    conds = [c for c in _CONDS if not only or c[0] == only]
    if not conds:
        print(f"条件 '{only}' は無い。選べるのは: {[c[0] for c in _CONDS]}")
        return

    # ★Viewerモード：E_VIEW=1。E_COND で条件を選ぶ（既定は最初の条件）。
    if os.environ.get("E_VIEW", "0") == "1":
        view_condition(*conds[0])
        return

    print(f"制御頻度プローブ  age={AGE} β={BETA} seed={SEED} 総物理step={TOTAL_STEPS}")
    print("★学習は一切しない。色付きノイズだけで体を動かし、筋と関節を測る。")
    results = [run_condition(*c) for c in conds]

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOG_DIR, f"ctrl_freq_probe_{stamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n" + "=" * 92)
    print(f"{'条件':<12}{'|qvel|':>9}{'可動域':>8}{'jerk':>9}{'同時活性':>9}{'力効率':>8}"
          f"{'体幹回転':>9}{'うつ伏せ%':>10}")
    print("-" * 92)
    for r in results:
        print(f"{r['cond']:<12}{r['qvel_mean']:>9.4f}{r['rom_mean_rad']:>8.3f}{r['jerk_mean']:>9.1f}"
              f"{r['coactivation']:>9.4f}{r['force_efficiency']:>8.3f}"
              f"{r['trunk_rot_mean_deg']:>9.1f}{r['prone_frac']*100:>10.1f}")
    print("=" * 92)

    # 判定の読み方を毎回出す（後から結果だけ見て誤読しないため）
    by = {r["cond"]: r for r in results}
    if "K100" in by and "K10" in by:
        ratio = by["K10"]["qvel_mean"] / by["K100"]["qvel_mean"] if by["K100"]["qvel_mean"] else float("nan")
        print(f"\n★K10 / K100 の動きの量 = {ratio:.3f}")
        print("  <1.0 なら『Kを短くすると動きが縮む』＝打ち消し合い仮説を支持。")
        print("  そのとき同時活性化・拮抗筋相関が上がり、力効率が下がっていれば原因まで確定。")
        print("  ≈1.0 なら仮説は否定＝過去のK=10失敗の原因は別（学習側を疑う）。")
    if "passive" in by:
        p = by["passive"]
        print(f"\n★passive（完全脱力）: 体幹回転={p['trunk_rot_mean_deg']:.1f}度  "
              f"手-体幹={p['hand_dist_m']:.3f}m")
        print("  受動的筋力だけで仰向けが保たれていれば、姿勢保持機構は不要という文献の結論を裏づける。")
    # ★うつぶせ化の読み方（2026-07-25、Viewerでの目視から追加）
    print("\n★体幹の回転（0度=仰向けのまま、180度=完全にうつぶせ）")
    print("  実際の新生児は寝返りできない（4-6ヶ月から）ので、ノイズだけでうつぶせになるのは")
    print("  明確に非人間的＝振幅が大きすぎるか、身体の物理が軽すぎる疑い。")
    print("  amp15（振幅半分）でうつぶせ%が下がれば『振幅が原因』、下がらなければ別の原因。")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
