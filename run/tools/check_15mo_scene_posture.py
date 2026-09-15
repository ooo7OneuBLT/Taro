# -*- coding: utf-8 -*-
"""15ヶ月シーン（座位_15ヶ月_2おもちゃ_F2-2_1個提示_fovea_2026-08-23）の座位が
成立するかを確かめる（設計_F2-2_見た物の名前を言う.md 第4部 実装作業①）。

run/tools/check_12mo_scene_posture.py の15ヶ月版（流用）。

依頼：F2-2の準備（本走行なし）。
やること：12ヶ月版の関節角度を15ヶ月の体に流用して数十ステップ動かし、
座位が崩れないかを第三者視点の画像で目視確認する。

【この道具がやらないこと】
    ・学習は回さない（run/main.pyは使わない）
    ・崩れた場合の姿勢作り直しはしない（ユーザーの判断が要るため、ここでは
      崩れたかどうかを実測して画像に残すところまで）

使い方::

    .venv/Scripts/python.exe run/tools/check_15mo_scene_posture.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)
_SCENE_TOOLS = os.path.join(_R, "run", "scene_tools")
if _SCENE_TOOLS not in sys.path:
    sys.path.insert(0, _SCENE_TOOLS)

import numpy as np                                          # noqa: E402
import scene_io                                               # noqa: E402

SCENE_NAME = "run/scenes/_旧/座位_15ヶ月_2おもちゃ_F2-2_1個提示_fovea_2026-08-23.json"
OUT_DIR = os.path.join(_R, "F", "logs", "15ヶ月シーン準備_2026-08-23")
N_STEPS = 60  # 「数十ステップ」


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    scene = scene_io.load(SCENE_NAME)
    print(f"[scene] {scene['name']}  age_months={scene['body']['age_months']}"
          f"  fovea_camera={scene['body'].get('fovea_camera')}"
          f"  develop_from_age={scene['body'].get('develop_from_age')}")

    env, hands = scene_io.build(scene, orient=False, vor=False, seed=0, verbose=False)
    scene_io.verify(scene, env, strict=False, verbose=True)  # 指紋なし→素通り想定

    scene_io.reset_to_scene(env, scene, hands=hands, seed=0)

    u = env.unwrapped
    m, d = u.model, u.data

    def hip_pos():
        # body_support.pin_root=true。骨盤(hip)の位置で「座位が崩れて倒れた/沈んだ」
        # を見る簡便な代理指標（詳細な体幹角度は目視画像で補う）。
        bid = m.body("hip").id if _has_body(m, "hip") else None
        if bid is None:
            return None
        return d.xpos[bid].copy()

    def _has_body(model, name):
        try:
            model.body(name)
            return True
        except KeyError:
            return False

    p0 = hip_pos()
    scene_io.capture_eye(env, os.path.join(OUT_DIR, "step000_body.png"), cam="body")

    positions = [p0]
    for i in range(1, N_STEPS + 1):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        if i in (10, 20, 30, 45, 60):
            scene_io.capture_eye(
                env, os.path.join(OUT_DIR, f"step{i:03d}_body.png"), cam="body")
        positions.append(hip_pos())

    print("[結果] hip位置の推移（xyz, m）")
    for i, p in zip([0] + list(range(1, N_STEPS + 1)), positions):
        if i in (0, 10, 20, 30, 45, 60) and p is not None:
            print(f"  step{i:03d}: {p}")

    if p0 is not None and positions[-1] is not None:
        drift = float(np.linalg.norm(positions[-1] - p0))
        print(f"[結果] hip位置の総移動量(0→{N_STEPS}): {drift:.4f} m")
        # Tier3：崩れ判定のしきい値に文献根拠は無い。ここでは「大きくずれたら
        # 目視で必ず確認する」ための粗いフラグとして0.05m(5cm)を暫定採用
        # （check_12mo_scene_posture.pyと同じしきい値）。
        if drift > 0.05:
            print("[警告] hip位置が5cm以上動いた＝崩れている可能性。画像で確認すること。")
        else:
            print("[所見] hip位置の移動は小さい（5cm未満）。ただし数値だけで断定せず"
                  "画像を必ず目視すること（落とし穴チェックリスト項70）。")

    print(f"[画像保存先] {OUT_DIR}")


if __name__ == "__main__":
    main()
