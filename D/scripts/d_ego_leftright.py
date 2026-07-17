"""
★②本番：親=左右に動く（信号）＋ 太郎=頭を左右に振る（egomotion）。
体の感覚（前庭・固有感覚・遠心性コピー）を足せば、自己運動下でも親の左右移動を読めるか。

【土台の安定化】各クリップの頭で 太郎を初期姿勢へリセット＋養育者を定位置にワープ
（太郎の沈み・養育者の降下という揺らぎを断ち、測るのを純粋に egomotion だけにする）。

【設計】親を y 方向に左右へ動かす（左/右がラベル）。自己運動は頭yaw（振り向き）で、
向き・振り幅は左右ラベルと"独立"に抽選＝体の感覚だけでは当てられない(50%対照が成立)。

【比較する5条件】
  ①静止・画像だけ            … 天井（左右は静止なら読める）
  ②自己運動・画像だけ        … 床（頭yawで相手の左右が埋もれる）
  ③自己運動・体の感覚だけ    … カンニング検査。~50%であるべき
  ④自己運動・画像＋体の感覚  … 本命。②より上がれば「体の感覚で割り引ける」
  ⑤本命のシャッフル          … 時間順序を使っているか

【動画】静止/自己運動 × 親が左/右 の4パターンを両視点(第三者+一人称)で保存＋絶対パス表示。

使い方: python d_ego_leftright.py [n_clip]
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "C"))
import paths; paths.setup_brain_path(); sys.path.insert(0, paths.MIMO_DIR)

import numpy as np
import torch
import cv2
import mujoco
import gymnasium as gym
import mimoEnv  # noqa
from gymnasium.envs.registration import register
from d1_carer_vision_env import CarerVisionEnv, lean_vision_params
from d_supine_env import infant_touch_params
import d_vision_egomotion as E   # build_X / train_eval / split / run_combo / _red_pixels を再利用
import d_ego_demo as D           # third_person / HOME_Z を再利用

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 絵文字・日本語をWindowsで安全に出力
torch.set_num_threads(4)
register(id="EgoLR-v0", entry_point="d1_carer_vision_env:CarerVisionEnv", max_episode_steps=100000)

RES = 64            # 解像度（32→64：エッジが滑らかで左右移動の分解能が上がる）
FOVY = 120          # 視野角(度)。本家60は狭すぎ頭yawで即視界外→広げて確定(検証済み)
CARER_SIZE = 0.10   # 養育者サイズ。fovy120では0.10でも飽和せず中央~480px(検証済み)。0.14+は腕と衝突
FACE_X = 0.37       # 養育者の前後位置（固定）
Z_FIX = 0.45        # 養育者の高さ（大きく写る＋第三者で目も見える）
Y_SPAN = 0.04       # 左右の移動幅。頭yawより小さくして「弱い頭yawでも反転が起きる」を成立させる
CENTER_MARGIN = 0.03  # 出発位置をこの範囲でランダムにずらす（始点位置での位置漏れを断つ）
K = 4
SUB = 25
HEAD_YAW = 5        # act:head_swivel（左右の振り向き）
YAW_LO, YAW_HI = 0.70, 0.90   # 頭yawの振り幅。yaw0.8前後で「反転あり＆視界外0」を実測確認。
                              # 強すぎ(1.0+)は自分の腕が視界に写り込む＆視界外→生存バイアス再発、
                              # なので上限を抑え、親の移動をそれより小さくして反転を成立させる。
# 赤判定を相対赤みに直した結果、最悪ケース(mode C)でも視界外0%（検証済み）＝生存バイアスは
# 原理的に起きない。RED_MINは念のための下限（端でも十分写る）。
RED_MIN = 40        # 各コマにこの数以上の赤（新赤判定で中央~480px。端の最小でも125）
VDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "video"))


def place(u, y, z=Z_FIX):
    """太郎を初期姿勢へ、養育者を (FACE_X, y, z) へワープ（土台の安定化）。"""
    u.data.qpos[:] = u.init_position
    u.data.qvel[:] = 0.0
    for nm, val in (("carer_x", FACE_X - 0.10), ("carer_y", y), ("carer_z", z - D.HOME_Z)):
        a = u.model.joint(nm).qposadr[0]
        u.data.qpos[a] = val
    mujoco.mj_forward(u.model, u.data)


def collect(self_move, n_clip):
    """親の左右移動クリップを集め、各コマの 画像/固有感覚/前庭/遠心性コピー を記録。"""
    env = gym.make("EgoLR-v0", vision_params=lean_vision_params(RES, fovy=FOVY),
                   touch_params=infant_touch_params(2.0), hand_size=CARER_SIZE, render_mode="rgb_array")
    env.reset(seed=0); u = env.unwrapped; na = u.action_space.shape[0]
    rng = np.random.default_rng(0); torch.manual_seed(0)
    imgs, props, vests, acts, Y = [], [], [], [], []
    attempts = 0; max_attempts = n_clip * 10
    while len(Y) < n_clip and attempts < max_attempts:
        attempts += 1
        right = bool(rng.integers(0, 2))
        # ★出発位置をランダムにずらす＝「動きの向き」だけがラベルになる（始点位置での位置漏れを断つ）
        center = float(rng.uniform(-CENTER_MARGIN, CENTER_MARGIN))
        ys = center + (np.linspace(-Y_SPAN, Y_SPAN, K) if right else np.linspace(Y_SPAN, -Y_SPAN, K))
        # 自己運動：頭yaw。向き・振り幅は左右ラベルと独立に抽選（体の感覚だけでは当てられない）
        yaw_dir = float(rng.choice([-1.0, 1.0])); yaw_amp = float(rng.uniform(YAW_LO, YAW_HI))
        place(u, ys[0])
        for _ in range(15):
            u.set_hand_target([FACE_X, ys[0], Z_FIX]); env.step(np.zeros(na, np.float32))
        f_img, f_prop, f_vest, f_act, reds = [], [], [], [], []
        for t in range(K):
            ctrl = np.zeros(na, np.float32)
            if self_move:
                ctrl[HEAD_YAW] = yaw_amp * (t + 1) / K * yaw_dir
            for _ in range(SUB):
                u.set_hand_target([FACE_X, ys[t], Z_FIX]); obs, *_ = env.step(ctrl)
            eye = obs["eye_left"]
            f_img.append(eye.astype(np.float32).reshape(-1) / 255.0)
            f_prop.append(np.asarray(obs["observation"], np.float32))
            f_vest.append(np.asarray(obs["vestibular"], np.float32))
            f_act.append(ctrl.copy())
            reds.append(E._red_pixels(eye))
        if min(reds) < RED_MIN:
            continue
        imgs.append(np.stack(f_img)); props.append(np.stack(f_prop))
        vests.append(np.stack(f_vest)); acts.append(np.stack(f_act)); Y.append(int(right))
        if len(Y) % 50 == 0:
            print(f"  採用 {len(Y)}/{n_clip}（試行{attempts}）", flush=True)
    env.close()
    return (dict(img=np.stack(imgs), prop=np.stack(props), vest=np.stack(vests),
                 act=np.stack(acts), y=np.asarray(Y)), len(Y) / max(attempts, 1))


def record_case(self_move, right, label):
    """1クリップを両視点(第三者+一人称)でなめらかに撮り、mp4/gif/pngで保存。絶対パスを返す。"""
    env = gym.make("EgoLR-v0", vision_params=lean_vision_params(RES, fovy=FOVY),
                   touch_params=infant_touch_params(2.0), hand_size=CARER_SIZE, render_mode="rgb_array")
    env.reset(seed=0); u = env.unwrapped; na = u.action_space.shape[0]
    ren = mujoco.Renderer(u.model, 240, 240)
    ys = (np.linspace(-Y_SPAN, Y_SPAN, K) if right else np.linspace(Y_SPAN, -Y_SPAN, K))
    yaw_amp, yaw_dir = 0.8, 1.0    # 動画用：反転が見える＆腕が写り込まない強さ
    place(u, ys[0])
    for _ in range(15):
        u.set_hand_target([FACE_X, ys[0], Z_FIX]); env.step(np.zeros(na, np.float32))
    frames = []
    for t in range(K):
        ctrl = np.zeros(na, np.float32)
        if self_move:
            ctrl[HEAD_YAW] = yaw_amp * (t + 1) / K * yaw_dir
        for s in range(SUB):
            u.set_hand_target([FACE_X, ys[t], Z_FIX]); obs, *_ = env.step(ctrl)
            if s % 4 == 0:                                   # コマ内も記録＝なめらか
                eye = cv2.resize(obs["eye_left"], (240, 240), interpolation=cv2.INTER_NEAREST)
                tp = D.third_person(ren, u.data)
                c = np.zeros((240, 488, 3), np.uint8); c[:, :240] = tp; c[:, 248:] = eye
                cv2.putText(c, "3rd person", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(c, "Taro eye", (254, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(c, label, (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                frames.append(c)
    ren.close(); env.close()
    os.makedirs(VDIR, exist_ok=True)
    tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    mp4 = os.path.join(VDIR, f"lr_{tag}.mp4"); gif = os.path.join(VDIR, f"lr_{tag}.gif")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15, (488, 240))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    try:
        import imageio
        imageio.mimsave(gif, frames, duration=0.06)
    except Exception:
        from PIL import Image
        Image.fromarray(frames[0]).save(gif, save_all=True,
            append_images=[Image.fromarray(f) for f in frames[1:]], duration=60, loop=0)
    return os.path.abspath(mp4), os.path.abspath(gif)


def main():
    n_clip = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print("=== ②本番：親=左右 ＋ 太郎=頭yaw。体の感覚でegomotionを割り引けるか ===")
    print(f"解像度{RES} 親高さ{Z_FIX} 左右幅±{Y_SPAN} 頭yaw{YAW_LO}-{YAW_HI}。{n_clip}クリップずつ。\n")

    print("--- 収集①：静止（天井の基準）---")
    st, acc_st = collect(False, n_clip)
    print(f"  → 採用率 {acc_st*100:.0f}%  採れた {len(st['y'])}本")
    print("--- 収集②：自己運動（頭yaw）---")
    mv, acc_mv = collect(True, n_clip)
    print(f"  → 採用率 {acc_mv*100:.0f}%  採れた {len(mv['y'])}本")

    tr_s, te_s = E.split(len(st["y"])); tr_m, te_m = E.split(len(mv["y"]))
    print("\n=== 判定 ===")
    c_top = E.run_combo(st, ["img"], tr_s, te_s, "①静止：画像だけ(天井)")
    a_img = E.run_combo(mv, ["img"], tr_m, te_m, "②自己運動：画像だけ(床)")
    a_body = E.run_combo(mv, ["vest", "prop", "act"], tr_m, te_m, "③自己運動：体の感覚だけ")
    a_all = E.run_combo(mv, ["img", "vest", "prop", "act"], tr_m, te_m, "④自己運動：画像＋体の感覚")
    a_sh = E.run_combo(mv, ["img", "vest", "prop", "act"], tr_m, te_m, "⑤④のシャッフル", shuffle=True)

    print("\n=== まとめ ===")
    print(f"  ①天井（静止・画像だけ）      : {c_top*100:5.1f}%")
    print(f"  ②床  （自己運動・画像だけ）    : {a_img*100:5.1f}%")
    print(f"  ③検査（自己運動・体の感覚だけ）: {a_body*100:5.1f}%  ←~50%であるべき")
    print(f"  ④本命（自己運動・画像＋体感覚）: {a_all*100:5.1f}%")
    print(f"  ⑤④のシャッフル              : {a_sh*100:5.1f}%  ←下がるべき")

    print("\n=== 解釈 ===")
    if a_body > 0.62:
        print("⚠ 体の感覚だけで当たる＝カンニング(左右が体の感覚に漏れている)。設計見直し。")
    elif a_all - a_img > 0.12 and a_all - a_sh > 0.10:
        print("★体の感覚を足すと自己運動下でも読める＝egomotionは体の感覚で割り引ける。情報は在る。")
    elif a_all - a_img > 0.05:
        print("△ 体の感覚で多少改善するが不十分。振り幅や入力を調整して再検証。")
    else:
        print("→ 体の感覚を足しても改善せず。判定器では割り引けない可能性。")

    print("\n=== すべての場合の動画（両視点・絶対パス）===")
    cases = [(False, False, "STILL LEFT"), (False, True, "STILL RIGHT"),
             (True, False, "SELFMOVE LEFT"), (True, True, "SELFMOVE RIGHT")]
    for sm, rt, lab in cases:
        mp4, gif = record_case(sm, rt, lab)
        print(f"  【{lab}】")
        print(f"     mp4: {mp4}")
        print(f"     gif: {gif}", flush=True)


if __name__ == "__main__":
    main()
