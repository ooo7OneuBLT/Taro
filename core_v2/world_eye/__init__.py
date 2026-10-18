"""目だけの場面（2026-10-18、ユーザー決定「目だけの場面」）。

言葉の段階（7〜10）で、MIMo の体の代わりに使う小さな MuJoCo の世界。体は無く、頭の位置に目（カメラ）が1つあり、
左右（pan）と上下（tilt）に向きを変える。目の動き方は、今の目を向ける反射の決まり（撃てる間隔・1回で動く角度）を
そのまま「動き方の式」（運動学）で書いたもの。筋肉や体の揺れは無い。

MIMo は消さない（run/world/ はそのまま）。体が要る段階が来たら、そちらへ戻す。

置き場所：
* scene.py    … 場面の XML を作る（部屋・床・壁・机・親の場所・物・目）。MIMo の場面と同じ寸法・同じ置き方
* objects.py  … 物の形の一覧（ボール・コップ・妨害物・くつ・おさら・…・段階6の6つの物）
* eye.py      … 目の動き方（撃てる間隔 0.25 秒、方向 (h, v) → 角度 h×30度・v×30度、主系列の速さ）
* env.py      … 環境本体。run_v2 が使う形（reset・step・obs["eye_left"]・info["parent_utterance"]・
                 unwrapped._orienting.set_map_target・_gaze_angle_to）をそのまま持つ。物を見せる・隠す・動かす・
                 親が持つ、をはっきりした呼び出しで行い、そのたびに前後の画像を記録のフォルダへ残す
* parent.py   … 親（run/world/parent.py の ParentLabeling・WordSchedule をそのまま使い、物の動かし方だけこの世界の
                 呼び出しへつなぐ）
* build.py    … run_v2.build_env と同じ引数で、この世界を作る（同じ場面の JSON を読む）
* switch.py   … 道具に --world eye|mimo を足すための切り替え
* eye_backend.py … 目の入口（eyein）の DINOv2 を、PC（GPU）・クラウド（重みが取れたら）で同じように作る。
                 重みが無いところで世界の配線だけ確かめるための「差し替え品」（DINOv2 ではない）もここ
"""

import os as _os
import sys as _sys

# 画面の無い所で描く方法は、mujoco を読む前に決める。Linux で指定が無ければ、GPU の口（/dev/dri）があれば egl、
# 無ければ osmesa（クラウド。CPU で描く、1枚 約16ミリ秒）。Windows の PC は指定しない（mujoco の既定）
if _sys.platform.startswith("linux") and not _os.environ.get("MUJOCO_GL"):
    _os.environ["MUJOCO_GL"] = "egl" if _os.path.exists("/dev/dri") else "osmesa"
# クラウド（osmesa）では、mujoco で1枚描いたあとに torch._dynamo を初めて読むと落ちる（Segmentation fault。
# torch.optim.Adam を作るときに読まれる。2026-10-18、段階9で見つけた）。描く前に読んでおく
if _os.environ.get("MUJOCO_GL") == "osmesa":
    import torch._dynamo  # noqa: F401,E402

from .env import EyeWorldEnv  # noqa: F401,E402
from .build import build_env, close_env  # noqa: F401,E402
