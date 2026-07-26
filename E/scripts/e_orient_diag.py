"""視線誘導反射が発火しない原因の切り分け。

【疑い】
  A. 視覚が更新されていない（VISION_MIN_DTのキャッシュ）
  B. qpos の直接書き換えが視覚に反映されていない（mj_forwardが要る）
  C. 反射の残差計算が常にゼロを返している

【診断】
  1. env.step() で get_vision_obs が呼ばれているか（画像を取得できるか）
  2. qposを書き換えて、視覚が実際に変わるか（2枚の画像を比べる）
  3. reflex._direction を手動で呼んで h_dir が出るか
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"), _HERE]:
    if p not in sys.path: sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco


def main():
    from e_toy_env import ToySupineEnv
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env

    kw = body_kwargs_from_env(0.0, verbose=False)
    env = ToySupineEnv(actuation_model=MuscleModel, vision_params=None,
                       age=0.0, toy=True, orient=True, **kw)
    m, d = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    n_act = env.action_space.shape[0]
    zero = np.zeros(n_act, dtype=np.float32)

    toy_bid = m.body("test_object1").id
    toy_jid = m.body_jntadr[toy_bid]
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dof = int(m.jnt_dofadr[toy_jid])

    # まず1step進めて安定
    for _ in range(50):
        env.step(zero)

    # === 診断1: 現時点で get_vision_obs が動くか ===
    print("=== 診断1: get_vision_obs の動作 ===")
    img1 = env.unwrapped.get_vision_obs()
    print(f"  返り値のキー: {list(img1.keys()) if isinstance(img1, dict) else type(img1)}")
    if isinstance(img1, dict):
        for k, v in img1.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")

    # === 診断2: おもちゃを大きく動かして、視覚が変わるか ===
    print("\n=== 診断2: おもちゃを右に10cm動かして視覚が変わるか ===")
    toy_center = d.qpos[toy_qadr:toy_qadr+3].copy()
    # 元の位置で描画
    d.qpos[toy_qadr:toy_qadr+3] = toy_center
    d.qvel[toy_dof:toy_dof+6] = 0
    mujoco.mj_forward(m, d)     # ★ここが重要：qpos書き換え後は forward が要る
    # キャッシュを飛ばすため time を進める
    env.unwrapped._vision_t = None
    img_before = env.unwrapped.get_vision_obs()
    eye_before = img_before["eye_left"].copy() if "eye_left" in img_before else None

    d.qpos[toy_qadr:toy_qadr+3] = toy_center + np.array([0, -0.10, 0])
    d.qvel[toy_dof:toy_dof+6] = 0
    mujoco.mj_forward(m, d)
    env.unwrapped._vision_t = None
    img_after = env.unwrapped.get_vision_obs()
    eye_after = img_after["eye_left"].copy() if "eye_left" in img_after else None

    if eye_before is not None and eye_after is not None:
        diff = np.abs(eye_after.astype(float) - eye_before.astype(float))
        print(f"  画像の差: mean={diff.mean():.4f}, max={diff.max():.2f}, "
              f"変化画素% ={100*(diff > 5).mean():.1f}")
        if diff.max() < 1:
            print("  ⚠️ 画像がほぼ変わっていない ＝ おもちゃの位置が視覚に反映されていない")
        else:
            print("  ○ 画像が変化している ＝ 視覚は正しく更新されている")

    # === 診断3: reflex._direction を直接呼ぶ ===
    print("\n=== 診断3: 反射の方向計算を直接呼ぶ ===")
    reflex = env.unwrapped._orienting
    if eye_before is not None and eye_after is not None:
        # 前フレームをセットしてから新フレームで方向を計算
        reflex.prev_eye = eye_before
        h_dir, v_dir = reflex._direction(eye_after)
        print(f"  h_dir={h_dir:.4f}, v_dir={v_dir:.4f}  "
              f"（★おもちゃが右に動いたので h_dir が負の大きい値になるべき）")
        if abs(h_dir) < 0.01 and abs(v_dir) < 0.01:
            print("  ⚠️ 方向計算がゼロを返す ＝ 残差法が働いていない")
        else:
            print("  ○ 方向計算が値を返している ＝ 反射のロジックは動く")

    # === 診断4: step() の中で orienting.update が呼ばれているか ===
    print("\n=== 診断4: step() で orienting.h_dir が更新されるか ===")
    reflex.reset()  # h_dir をゼロに戻す
    # おもちゃを毎ステップ動かしながら 30ステップ進める
    for k in range(30):
        d.qpos[toy_qadr:toy_qadr+3] = toy_center + np.array([0, -0.05, 0]) * (1 + 0.3 * np.sin(k * 0.5))
        d.qvel[toy_dof:toy_dof+6] = 0
        env.step(zero)
    print(f"  30step後: h_dir={reflex.h_dir:.4f}, v_dir={reflex.v_dir:.4f}")
    if abs(reflex.h_dir) < 0.001 and abs(reflex.v_dir) < 0.001:
        print("  ⚠️ step()を経由すると発火しない ＝ update が呼ばれていない or キャッシュされ続けている")
    else:
        print("  ○ step()経由でも反射が発火 ＝ 元テストの qpos 書き換えタイミングが問題")

    env.close()


if __name__ == "__main__":
    main()
