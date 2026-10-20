"""目の出口（出口（筋・内臓））。

段階2・工程2：見えた方へ目を向ける反射（orienteye）から方向 (h, v) が来た回だけ、
環境への指令として返す。run_v2 がそれを環境の中の目を動かす仕組み
（`OrientingReflexV2.set_map_target(h, v)`、h・v は [-1,1]、右・上が正）へ渡す。
方向がゼロ（＝その回は撃たない）なら何も返さない。

gaze→eyeout（ゆっくり追う道）・cb→eyeout は段階2では使わない。
"""

import torch

from .base import ENV, Box


class Eyeout(Box):
    BOX_ID = "eyeout"
    PURE = True   # 出すものは届いた値だけで決まる（同じ値が届き続ける回は計算しない、brain.tick）

    def step(self, inbox):
        out = self.passthrough()
        cmd = inbox.get("orienteye")
        if cmd is not None and float(torch.linalg.vector_norm(cmd)) > 0.0:
            out[ENV] = (float(cmd[0]), float(cmd[1]))
        return out
