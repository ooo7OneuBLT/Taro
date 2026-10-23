"""段階6（スミスとユー）の場面を目だけの場面で回せるか（2026-10-18、smith_yu --world eye が
run_v2.directions の KeyError 'test_object3' で止まった不具合の確かめ）。

脳は回さない（目の入口は DINOv2 と scipy が要り、クラウドでは作れない）。環境と、道具が毎回呼ぶ記録の部分
（gaze_angles・directions）と、画面に出た物を見ているかの角度だけを確かめる。

予想：場面に無い test_object3 の列は記録に出ず、視線の列（x_gaze・y_gaze）とボール・コップの列は出る。
画面が出ているあいだは、左右の物（sy_obj）の角度が数字で取れる。
"""

import os

os.environ.setdefault("TARO_EYE_LOG", "off")

from core_v2 import run_v2  # noqa: E402
from core_v2.tools import smith_yu  # noqa: E402
from core_v2.world_eye import build  # noqa: E402


class _NoBrain:
    def reset(self):
        pass


def test_smith_yu_scene_runs_in_eye_world(tmp_path):
    sy = smith_yu._schedule_tool().make(0, gap_s=smith_yu.GAP_S, test_iti_s=smith_yu.GAP_S)
    world_extra = {"pairs": {"schedule": sy["pairs_schedule"]}, "word_test": {"schedule": sy["word_schedule"]}}
    env = build.build_env(smith_yu.SCENE, None, seed=0, world_extra=world_extra, log_dir=str(tmp_path))
    s0 = sy["slides"][0]
    seen = []

    def on_tick(row):
        if s0["t0"] <= row["t"] < s0["t1"]:
            u = env.unwrapped
            seen.append([u._gaze_angle_to(f"sy_obj{k}") for k in (s0["left"], s0["right"])])

    try:
        rows = run_v2.run_trial(_NoBrain(), env, seed=0, duration_s=s0["t0"] + 1.0, use_brain=False, on_tick=on_tick)
    finally:
        build.close_env(env)
    r = rows[-1]
    assert "x_gaze" in r and "y_gaze" in r
    assert "x_distractor" not in r          # 場面に無い物は書かない
    assert seen and all(a is not None for a in seen[-1])
