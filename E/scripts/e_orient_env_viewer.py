"""視線誘導反射の環境テストの前に、環境そのものをViewerで目視確認する。

【確認ポイント】
  1. おもちゃ（test_object1）がどこにあるか。太郎の視界に入る位置か
  2. 柵（E_FENCE）があるか。視界を邪魔していないか
  3. おもちゃを「小さく激しく」揺らしたとき、どう見えるか
  4. 太郎の姿勢（新生児・仰向け・生理的屈曲）が想定どおりか

操作: 左ドラッグ=視点回転／右ドラッグ=平行移動／スクロール=ズーム

★編集モード（E_EDIT=1）
  おもちゃをマウスでつかんで動かし、置きたい位置を探せる。
    ダブルクリックでおもちゃを選択 → Ctrl+右ドラッグ で移動
    （Ctrl+左ドラッグ で回転）
  重力と吊り紐を打ち消すので、離した場所にとどまる。
  ターミナルに「目からの距離」「肩からの距離」「腕の何%か」が出続けるので、
  それを見ながら良い位置を探す。決まったら数値を控えて TOY_DISTANCE 等に反映する。

環境変数:
  E_EDIT=1        ★編集モード（ドラッグでおもちゃを動かせる。揺れは止まる）
  E_SHAKE=0       おもちゃを揺らさない（既定は揺らす）
  E_SHAKE_AMP     振幅[m]（既定 0.015 = 1.5cm）
  E_SHAKE_HZ      周波数[Hz]（既定 2.5）
  E_FLEXION=1     ★生理的屈曲をON（既定OFF。OFFだと膝が伸びきる）
  E_VOR=0         前庭動眼反射をOFF（既定ON。ONだと脱力時に目が下を向く）
  E_TOY_OBJ=0     おもちゃを消す
  E_FENCE=0       柵を消す

使い方:
    E_FLEXION=1 E_VOR=0 E_EDIT=1 .venv/Scripts/python.exe E/scripts/e_orient_env_viewer.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco
import mujoco.viewer

EDIT = os.environ.get("E_EDIT", "0") == "1"
SHAKE = os.environ.get("E_SHAKE", "1") == "1" and not EDIT
SHAKE_AMP = float(os.environ.get("E_SHAKE_AMP", "0.015"))   # 1.5cm
SHAKE_HZ = float(os.environ.get("E_SHAKE_HZ", "2.5"))
ARM_REACH = 0.158   # 実測（上腕7.9 + 前腕7.9 cm）


def main():
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=True)
    env = ToySupineEnv(actuation_model=MuscleModel,
                       vision_params=infant_vision_params(),
                       age=0.0, toy=True, orient=False, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)

    if EDIT:
        # ★吊り紐を無効化する。_apply_tether() は env.step の中で毎step
        #   おもちゃを支点へ引き戻す力を xfrc_applied に書くので、
        #   これを止めないとドラッグしても元の位置に戻ってしまう
        #   （＝「Ctrl+右ドラッグで動かない」の原因）。
        #   _anchor が None なら _apply_tether は即 return する。
        env.unwrapped._anchor = None

    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    toy_bid = m.body("test_object1").id
    toy_jid = m.body_jntadr[toy_bid]
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])
    toy_center = d.qpos[toy_qadr:toy_qadr + 3].copy()
    toy_mass = float(m.body_mass[toy_bid])

    eye_bid = m.body("left_eye").id
    sh_bid = m.body("right_upper_arm").id
    hand_bid = m.body("right_hand").id

    P = lambda bid: d.xpos[bid].copy()

    print(flush=True)
    print("=" * 64, flush=True)
    print("  モード         :", "★編集（ドラッグで動かせる）" if EDIT else "観察", flush=True)
    print("  おもちゃ初期位置:", np.round(toy_center, 3), "[m]", flush=True)
    print("  生理的屈曲     :", "ON" if kw.get("flexion") else "OFF（E_FLEXION=1でON）", flush=True)
    print("  VOR            :", "OFF" if os.environ.get("E_VOR") == "0" else "ON（E_VOR=0でOFF）",
          flush=True)
    print("  腕の長さ       : %.1f cm" % (ARM_REACH * 100), flush=True)
    print("=" * 64, flush=True)
    print(flush=True)
    if EDIT:
        print("【編集モードの操作】", flush=True)
        print("  1. おもちゃを【ダブルクリック】して選択", flush=True)
        print("  2. 【Ctrl + 右ドラッグ】で移動（Ctrl + 左ドラッグ で回転）", flush=True)
        print("  3. 重力と吊り紐は打ち消してあるので、離した場所にとどまる", flush=True)
        print("  4. 下に出る距離を見ながら、良い位置を探す", flush=True)
        print("  5. 決まったら数値を教えてください（コードに反映します）", flush=True)
    else:
        print("【確認してほしいこと】", flush=True)
        print("  1. おもちゃが太郎の視界に入る位置にあるか", flush=True)
        print("  2. 柵が視界を邪魔していないか", flush=True)
        print("  3. 揺れ方が『小さく激しく』になっているか", flush=True)
        print("  4. 太郎の姿勢（仰向け・生理的屈曲）が想定どおりか", flush=True)
    print(flush=True)
    print("  視点: 左ドラッグ=回転 / 右ドラッグ=平行移動 / スクロール=ズーム", flush=True)
    print(flush=True)

    dt = float(m.opt.timestep) * int(env.unwrapped.frame_skip)
    t = 0.0
    last_print = -1.0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            if EDIT:
                # 重力を打ち消して、離した場所にとどまるようにする
                d.xfrc_applied[toy_bid, 2] = toy_mass * 9.81
                # 減衰（ドラッグを離したあとすぐ静止させる）。
                # ⚠️強すぎるとドラッグ自体が効かなくなるので控えめに。
                d.qvel[toy_dof:toy_dof + 6] *= 0.97
            elif SHAKE:
                off = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * t)
                d.qpos[toy_qadr:toy_qadr + 3] = toy_center + np.array([0.0, off, 0.0])
                d.qvel[toy_dof:toy_dof + 6] = 0.0

            env.step(zero)          # 脱力（反射も方策も無し＝環境だけを見る）
            t += dt

            if EDIT and t - last_print > 0.4:
                last_print = t
                toy, eye, sh, hd = P(toy_bid), P(eye_bid), P(sh_bid), P(hand_bid)
                de = np.linalg.norm(toy - eye) * 100
                ds = np.linalg.norm(toy - sh) * 100
                dh = np.linalg.norm(toy - hd) * 100
                print(f"  toy={np.round(toy,3)}  目→{de:5.1f}cm  "
                      f"肩→{ds:5.1f}cm ({ds/ (ARM_REACH*100) *100:3.0f}% of arm)  "
                      f"手→{dh:5.1f}cm", flush=True)
            viewer.sync()
    env.close()

    if EDIT:
        toy = P(toy_bid)
        print(flush=True)
        print("=" * 64, flush=True)
        print("  最終位置:", np.round(toy, 4), flush=True)
        print("  この数値を教えてください。コードに反映します。", flush=True)
        print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
