"""
★egomotion割引の門番：自己運動で崩れた視覚を、体の感覚で立て直せるか。

【なぜ・2026-07-16】昨日 d_vision_selfmove.py で、太郎が動くと「近づく/遠ざかる」が
98%→54%(チャンス)に崩壊した。原因は egomotion（自己運動）＝自分が動くと視界全体がズレ、
「相手が近づいた」のか「自分が寄った」のか画像だけでは分離できない。人間はこれを
前庭覚(頭の動き)・固有感覚(手足の配置)・遠心性コピー(自分の運動命令)で割り引いている。
太郎の観測にも全部ある。＝**判定器に渡す材料を「画像だけ」→「画像＋体の感覚」に増やせば
自己運動下でも読めるようになるか**を、使い捨て判定器で安く確かめる。

【設計】自己運動条件で、入力の中身だけ変えて比較：
  ① 画像だけ            … 床（昨日54%相当）を同じ運動レジームで再計測
  ② 体の感覚だけ(画像なし)… カンニング検査。相手の動きと自分の運動は無関係→~50%になるべき
  ③ 画像＋体の感覚       … 本命。②の情報で自己運動を割り引き、①より上がれば「情報は在る」
  ④ ③のシャッフル対照    … 時間順序を本当に使っているか（下がるべき）
  静止条件の画像だけ      … 天井（~98%）
体の感覚 = 前庭覚(6) + 固有感覚(621) + 遠心性コピー(action 90)。

【正直さチェック】各コマに養育者の赤が写っている割合も数える。自己運動で赤が消えていたら
「読めないのは egomotion ではなく相手が画面外に出ただけ」＝別問題。切り分ける。

使い方: python d_vision_egomotion.py [n_clip]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import torch
import torch.nn as nn
import cv2
import mujoco

torch.set_num_threads(4)

import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register

from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params

register(id="CarerEgo-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 32
FACE_X = 0.37
K = 4
# ★距離レンジは「相手が必ず写る帯」(z≈0.32〜0.45)の内側に収める。較正でz≤0.28は狭い上向き
#   視野から外れて0pxになると判明。この帯内なら近いほど大きく写り looming が読める。
Z_MID_LO, Z_MID_HI = 0.37, 0.40
SPAN_LO, SPAN_HI = 0.03, 0.05
SUBSTEPS = 25
SELF_SCALE = 0.60   # 自己運動の強さ
CARER_SIZE = 0.07   # 養育者カプセルの半径(m)。この帯で z0.32:144px→z0.44:26px と明瞭な looming
# 動かす関節＝腕・手のみ(14-71)。頭[5-7]・目[8-13]・胴[3-4]・腰[0-2]・脚[72-89]は動かさない
#   →視線は相手に残したまま、手が視界に割り込む"手の割り込み型egomotion"を作る
#   （頭運動型は太郎の狭い上向き視野では相手が即消えるため幾何的に不可＝別途 gaze安定化が要る）
MOVABLE = list(range(14, 72))
RED_MIN = 3         # 各コマにこの数以上の赤画素が要る（相手が写っているクリップだけ採用）


def _red_pixels(img):
    """養育者(赤カプセル)の写っている画素数。背景は青系(R<G<B)なので、絶対値でなく
    "Rが G・Bの平均よりどれだけ高いか"という相対的な赤みで判定する。

    【なぜ・2026-07-16】旧基準(R>100 & G<80 & B<80)は画面端で暗くなった養育者
    (実測RGB=100,32,32)を「赤でない」と誤判定していた。肉眼には明らかに赤いのに、
    絶対値のRしきい値100をわずかに割っただけで弾かれた＝視界外率が過大評価されていた。
    """
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    redness = r - (g + b) // 2
    return int(np.count_nonzero(redness > 25))


def _third_person(renderer, data, size=240):
    """太郎を外から見た第三者視点を描く（自由カメラ）。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.20, 0.0, 0.20]
    cam.distance = 1.1
    cam.azimuth = 90.0
    cam.elevation = -18.0
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def collect(self_move, n_clip, scale=SELF_SCALE, size=CARER_SIZE, movable=MOVABLE,
            save_examples=0):
    """近づく/遠ざかるクリップを集め、各コマの 画像/固有感覚/前庭/遠心性コピー を記録。

    self_move=True のとき、各コマ(25サブステップ)につき1つの運動命令を引いて保持
    （＝遠心性コピーが1コマ1本にきれいに対応する）。運動は相手の動きと完全に独立。
    ★相手が全コマに写っている(各コマ赤>=RED_MIN)クリップだけ採用＝相手が画面外に出た
    「見失い」を除外し、純粋に egomotion による見かけの乱れだけを測る。
    save_examples>0 なら最初の数クリップの[第三者視点+一人称視界]を動画用に貯めて返す。
    """
    env = gym.make("CarerEgo-v0", vision_params=lean_vision_params(RES),
                   touch_params=infant_touch_params(2.0), hand_size=size,
                   render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    na = u.action_space.shape[0]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    # ★第三者視点レンダラは必要なときだけ生成（常時生成すると目の描画とGL衝突し赤が消える）
    tp_ren = mujoco.Renderer(u.model, 240, 240) if save_examples > 0 else None
    imgs, props, vests, acts, Y = [], [], [], [], []
    examples = []
    attempts = 0
    max_attempts = n_clip * 8
    while len(Y) < n_clip and attempts < max_attempts:
        attempts += 1
        mid = float(rng.uniform(Z_MID_LO, Z_MID_HI)); span = float(rng.uniform(SPAN_LO, SPAN_HI))
        approach = bool(rng.integers(0, 2))
        zs = (np.linspace(mid + span, mid - span, K) if approach
              else np.linspace(mid - span, mid + span, K))
        f_img, f_prop, f_vest, f_act = [], [], [], []
        reds, big_eye, big_tp = [], [], []
        want_ex = len(examples) < save_examples
        # ★自己運動は「こちらが命令する」＝ランダムなバタつきでなく、1クリップ内で一方向へ
        #   なめらかに漂う運動(コヒーレントなドリフト)。向き・振幅はラベル(approach)を見ずに抽選
        #   ＝相手の遠近と自己運動は独立(体の感覚だけでは当たらない＝50%対照が保たれる)。
        #   ctrl経由で動かすので固有感覚・遠心性コピーは物理的に正常なまま。
        drift = np.zeros(na, dtype=np.float32)
        if self_move:
            d = rng.standard_normal(len(movable)); d /= (np.linalg.norm(d) + 1e-9)
            drift[movable] = d.astype(np.float32)                 # 腕(movable)を一方向へ
        for t in range(K):
            ctrl = (scale * (t + 1) / K * drift).astype(np.float32)  # 0→振幅scale へ滑らかに増加
            for _ in range(SUBSTEPS):
                u.set_hand_target([FACE_X, 0.0, zs[t]])
                obs, r, te, tr, info = env.step(ctrl)
            eye = obs["eye_left"]
            f_img.append(eye.astype(np.float32).reshape(-1) / 255.0)
            f_prop.append(np.asarray(obs["observation"], dtype=np.float32))
            f_vest.append(np.asarray(obs["vestibular"], dtype=np.float32))
            f_act.append(np.asarray(ctrl, dtype=np.float32))       # 遠心性コピー
            reds.append(_red_pixels(eye))
            if want_ex:
                big_eye.append(cv2.resize(eye, (240, 240), interpolation=cv2.INTER_NEAREST))
                big_tp.append(_third_person(tp_ren, u.data))
        if min(reds) < RED_MIN:                # 相手が消えたコマがある→このクリップは捨てる
            continue
        imgs.append(np.stack(f_img)); props.append(np.stack(f_prop))
        vests.append(np.stack(f_vest)); acts.append(np.stack(f_act)); Y.append(int(approach))
        if want_ex:
            examples.append((approach, big_tp, big_eye))
        if len(Y) % 50 == 0:
            print(f"  採用 {len(Y)}/{n_clip}（試行{attempts}）", flush=True)
    if tp_ren is not None:
        tp_ren.close()
    env.close()
    accept_rate = len(Y) / max(attempts, 1)
    return (dict(img=np.stack(imgs), prop=np.stack(props), vest=np.stack(vests),
                 act=np.stack(acts), y=np.asarray(Y)), accept_rate, examples)


def _standardize(a, mean, std):
    return (a - mean) / std


def build_X(data, keys, tr_idx, shuffle=False):
    """指定モダリティを連結して (n, K*dim) の特徴行列に。画像以外は train統計で標準化。"""
    n = len(data["y"])
    mats = []
    for k in keys:
        a = data[k].astype(np.float32)                 # (n, K, d)
        if k != "img":                                 # 画像は既に/255。他はz標準化
            flat = a[tr_idx].reshape(-1, a.shape[-1])
            mean = flat.mean(0, keepdims=True); std = flat.std(0, keepdims=True) + 1e-6
            a = _standardize(a, mean, std)
        mats.append(a)
    X = np.concatenate(mats, axis=2)                   # (n, K, sum_d)
    if shuffle:
        X = X.copy(); rsh = np.random.default_rng(7)
        for i in range(n):
            rsh.shuffle(X[i])                          # K方向(時間)をバラす
    return X.reshape(n, -1)


def train_eval(X, y, tr, te, tag, steps=4000):
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(X[tr], dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32).unsqueeze(1)
    for _ in range(steps):
        idx = torch.randperm(len(Xtr))[:128]
        loss = lossf(net(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (net(torch.tensor(X[te], dtype=torch.float32)).squeeze(1) > 0).numpy().astype(int)
    acc = float((pred == y[te]).mean())
    print(f"[{tag:34s}] 正解率 {acc*100:5.1f}%", flush=True)
    return acc


def split(n):
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    return perm[:int(n * 0.8)], perm[int(n * 0.8):]


def run_combo(data, keys, tr, te, tag, shuffle=False):
    X = build_X(data, keys, tr, shuffle=shuffle)
    return train_eval(X, data["y"], tr, te, tag)


def save_dual_video(examples, out, fps=10):
    """[第三者視点 | 一人称視界] を横並びにした動画を保存。各コマにラベル。"""
    frames = []
    for approach, tps, eyes in examples:
        label = "chikaduku(近づく)" if approach else "toozakaru(遠ざかる)"
        for j, (tp, eye) in enumerate(zip(tps, eyes)):
            canvas = np.zeros((240, 480 + 8, 3), dtype=np.uint8)
            canvas[:, :240] = tp
            canvas[:, 248:] = eye
            cv2.putText(canvas, "soto kara (3rd person)", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(canvas, "Taro no me (eye)", (254, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(canvas, f"{label}  {j+1}/{K}", (6, 232),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            for _ in range(max(1, fps // 3)):
                frames.append(canvas)
        for _ in range(4):
            frames.append(np.zeros_like(frames[-1]))          # クリップ間の区切り
    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    n_clip = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print("=== ★egomotion割引の門番：体の感覚で自己運動を割り引けるか ===")
    print(f"自己運動条件で入力の中身だけ変えて比較。相手が全コマ写るクリップを{n_clip}本ずつ採用。\n")

    print("--- 収集①：静止（天井の基準）---")
    st, acc_st, ex_st = collect(False, n_clip, save_examples=6)
    print(f"  → クリップ採用率: {acc_st*100:.0f}%")
    print("--- 収集②：自己運動（腕を動かす）---")
    mv, acc_mv, ex_mv = collect(True, n_clip, save_examples=6)
    print(f"  → クリップ採用率: {acc_mv*100:.0f}%（低いほど自己運動で相手が画面外に出やすい）")

    # 両視点の動画を保存（静止・自己運動それぞれ）
    vdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video")
    out_st = os.path.join(vdir, "ego_static_dual.mp4")
    out_mv = os.path.join(vdir, "ego_selfmove_dual.mp4")
    save_dual_video(ex_st, out_st)
    save_dual_video(ex_mv, out_mv)

    tr_s, te_s = split(len(st["y"]))         # ★実際に採れた本数で分割（フィルタで目標数に届かない場合に対応）
    tr_m, te_m = split(len(mv["y"]))
    print(f"\n実際に採れた本数: 静止{len(st['y'])} / 自己運動{len(mv['y'])}")
    print("\n=== 判定 ===")
    print("[天井] 静止・画像だけ")
    c_top = run_combo(st, ["img"], tr_s, te_s, "静止：画像だけ")

    print("\n[本題] 自己運動下で入力を変える")
    a_img = run_combo(mv, ["img"], tr_m, te_m, "自己運動：画像だけ（床）")
    a_body = run_combo(mv, ["vest", "prop", "act"], tr_m, te_m, "自己運動：体の感覚だけ(画像なし)")
    a_all = run_combo(mv, ["img", "vest", "prop", "act"], tr_m, te_m, "自己運動：画像＋体の感覚")
    a_all_sh = run_combo(mv, ["img", "vest", "prop", "act"], tr_m, te_m, "自己運動：画像＋体の感覚(順番バラバラ)", shuffle=True)

    print("\n=== まとめ ===")
    print(f"  天井（静止・画像だけ）          : {c_top*100:5.1f}%")
    print(f"  床  （自己運動・画像だけ）        : {a_img*100:5.1f}%")
    print(f"  検査（自己運動・体の感覚だけ）    : {a_body*100:5.1f}%  ←~50%であるべき(カンニング無し)")
    print(f"  本命（自己運動・画像＋体の感覚）  : {a_all*100:5.1f}%")
    print(f"  本命のシャッフル                : {a_all_sh*100:5.1f}%  ←下がるべき(時間順序を使用)")
    print(f"\n  クリップ採用率: 静止{acc_st*100:.0f}% / 自己運動{acc_mv*100:.0f}%"
          " ←自己運動で激減なら相手が画面外に出やすい(全コマ写るものだけ採用済み)")
    print(f"  両視点の動画: {os.path.normpath(out_st)}")
    print(f"            : {os.path.normpath(out_mv)}")

    print("\n=== 解釈 ===")
    if a_body > 0.62:
        print("⚠ 体の感覚だけで当たっている＝カンニング(相手の動きが体の感覚に漏れている)。設計見直し。")
    elif a_all - a_img > 0.15 and a_all - a_all_sh > 0.12:
        print("★体の感覚を足すと、自己運動下でも読めるようになった＝egomotionは体の感覚で割り引ける。")
        print("  → 情報は在ると確定。本物の脳(前庭・固有感覚・遠心性コピーを持つ)につなぐ意味が出た。")
    elif a_all - a_img > 0.05:
        print("△ 体の感覚で多少改善するが不十分。運動の強さ・入力の選び方を調整して再検証。")
    else:
        print("→ 体の感覚を足しても改善せず。判定器では割り引けない＝もっと深い統合が要る可能性。")


if __name__ == "__main__":
    main()
