# -*- coding: utf-8 -*-
"""F2-64: 「視野に物が無い」場合のMobileSAMの挙動を調べる下準備（2026-09-04）。

ユーザーの指摘：人間の目が開いている限り、床・壁は必ず映る。「本当に何も
見えない」のは目を閉じたときだけ。だから正しいテストは「画面が真っ黒」では
なく「床・壁は映っているが、つかめる物が1つも無い」場面にすること。

本スクリプトはMobileSAMを呼ばない（別venvのため）。太郎の実際のカメラ
（f63cと同じセットアップ・レンダラー）で、
  1) 物を1つも置かない場面（床・壁だけ）
  2) 比較用に物を1つ置いた場面
の生画像を224x224で書き出すだけ。MobileSAMへの投入は別スクリプトで行う。

既存の object_detector.py, f63c_sim_boundary_faithful.py は読むだけで変更しない。

    .venv/Scripts/python.exe F/scripts/f64_empty_view_probe.py
出力: F/logs/F2-64_視野に物が無い場合/raw_empty.png
      F/logs/F2-64_視野に物が無い場合/raw_with_object.png
"""
import os, sys, io, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("taro_core/src/brain"))
sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, mujoco
from PIL import Image
import f_gen_f49 as G
from f_present_angle_probe import present_quat
from run.plugins.common import scene as scene_mod

OUT = "F/logs/F2-64_視野に物が無い場合"


def main():
    os.makedirs(OUT, exist_ok=True)
    spec = json.load(io.open("F/experiments/F2-49_r1_%s.json" % G.DATE, encoding="utf-8"))
    sc = json.load(io.open("run/scenes/座位_12ヶ月_F2-49_実物8択_r1_個体1_%s.json" % G.DATE, encoding="utf-8"))
    env, sc, _ = scene_mod.build(sc["name"], taro=spec["taro"], seed=0, verbose=False); env.reset(seed=0)
    u = env.unwrapped; m, d = u.model, u.data
    cid = int(m.camera("eye_left").id)
    qadr = {k: v["qadr"] for k, v in dict(getattr(u, "_present_slots", {})).items()}
    qadr["toy1"] = int(u._toy_qadr)

    def reset_all():
        for a in qadr.values():
            d.qpos[a:a + 3] = [5.0, 3.0, -2.0]

    reset_all(); mujoco.mj_forward(m, d)
    eye = np.array(d.cam_xpos[cid]); Rc = np.array(d.cam_xmat[cid]).reshape(3, 3)
    fwd = -Rc[:, 2]; right = Rc[:, 0]
    ren = mujoco.Renderer(m, height=224, width=224)

    # 1) 物を1つも置かない場面
    reset_all(); mujoco.mj_forward(m, d)
    ren.update_scene(d, camera="eye_left")
    img_empty = ren.render().copy()
    Image.fromarray(img_empty).save(OUT + "/raw_empty.png")
    print("空の場面を保存:", OUT + "/raw_empty.png", img_empty.shape)

    # 2) 比較用：物を1つ置いた場面（f63cと同じ距離0.30m）
    ang5 = sc["world"]["parent_labeling"]["present_angles"]["toy5"]
    a = qadr["toy5"]
    reset_all()
    d.qpos[a:a + 3] = eye + fwd * 0.30
    d.qpos[a + 3:a + 7] = present_quat(u, ang5["yaw"], ang5["tilt"])
    mujoco.mj_forward(m, d)
    ren.update_scene(d, camera="eye_left")
    img_obj = ren.render().copy()
    Image.fromarray(img_obj).save(OUT + "/raw_with_object.png")
    print("物あり場面を保存:", OUT + "/raw_with_object.png", img_obj.shape)

    ren.close(); env.close()
    print("完了:", OUT)


if __name__ == "__main__":
    main()
