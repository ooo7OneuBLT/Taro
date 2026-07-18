"""taro-C5：MIMoの成長モジュールで年齢を変え、体重・筋力（最大トルク）がどう変わるかを確認。
新生児にすると本当に「軽く・弱く」なるか＝根拠ある弱さになるかの数値チェック。"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths
sys.path.insert(0, paths.MIMO_DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from mimoGrowth.growth import adjust_mimo_to_age
from mimoGrowth.scene import delete_growth_scene


def mimo_stats(model):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mimo_location")
    def is_desc(bid):
        while bid != 0:
            if bid == root:
                return True
            bid = model.body_parentid[bid]
        return False
    mass = sum(model.body_mass[i] for i in range(model.nbody) if is_desc(i))
    gear = np.abs(model.actuator_gear[:, 0])
    return mass, gear, model.njnt, model.nu


print(f"{'月齢':>5} | {'体重kg':>7} | {'関節数':>5} {'筋数':>5} | {'膝の最大トルク':>12} {'合計トルク':>9}")
print("-" * 62)
for age in [0, 1, 3, 6, 18]:
    scene = adjust_mimo_to_age(age, paths.SCENE, create_log=False)
    try:
        m = mujoco.MjModel.from_xml_path(scene)
        mass, gear, njnt, nu = mimo_stats(m)
        knee = 0.0
        for i in range(nu):
            if "left_knee" in m.actuator(i).name:
                knee = abs(m.actuator_gear[i, 0]); break
        print(f"{age:>4}ヶ | {mass:>7.2f} | {njnt:>5} {nu:>5} | {knee:>12.2f} {gear.sum():>9.1f}")
    finally:
        delete_growth_scene(scene)

print("\n参考：前回測ったデフォルト(18ヶ月相当)＝体重11.3kg・膝11.08Nm。")
print("関節数・筋数が全年齢で同じなら＝構造不変＝感覚の次元も不変＝学習済みモデルは継続学習でOK。")
