"""段階9：目を動かす指令の写しで、目が動いた分を先に知る（eyein.ego_shift）。"""

import numpy as np
import pytest
import torch

from core_v2.boxes.eyein import Eyein, command_shift, image_shift_near
from core_v2.formats import DTYPE


def _tex(seed=0):
    g = np.random.default_rng(seed)
    return g.random((224, 224)).astype(np.float32) * 255.0     # 224×224 のまだら


def test_command_shift_direction():
    dx, dy = command_shift(1.0, 0.0)
    assert abs(dx + 112.0) < 1e-6 and dy == 0.0          # 右へ視野の半分回る → 端の物が真ん中へ（中身は左へ112画素）
    dx, dy = command_shift(0.0, 0.5)
    assert dx == 0.0 and dy > 0                           # 上へ回る → 中身は下へ


def test_near_search_finds_shift_around_prediction():
    a = _tex()
    b = np.roll(a, (8, -60), axis=(0, 1))
    dx, dy, peak = image_shift_near(a, b, (-55.0, 5.0), 16.0)
    assert (dx, dy) == (-60.0, 8.0) and peak > 0.3


def _eye(ego="efference"):
    e = Eyein(ego=ego)
    return e


def test_efference_used_on_next_image_and_corrected_by_image():
    e = _eye()
    a = _tex(1)
    e.ego_shift(a)
    e.take_efference(torch.tensor([0.5, 0.0], dtype=DTYPE))
    pred = command_shift(0.5, 0.0)
    b = np.roll(a, int(round(pred[0])) - 6, axis=1)        # 体が少し短く跳んだ（6画素）
    dx, dy = e.ego_shift(b)
    assert e.last_ego_mode == "pred+image" and (dx, dy) == (round(pred[0]) - 6, 0.0)
    assert e.last_pred_px == pytest.approx(pred, abs=1e-3)
    dx, dy = e.ego_shift(b)                                # 写しは1回で使い切る
    assert (dx, dy) == (0.0, 0.0)


def test_falls_back_to_prediction_when_image_unreliable():
    e = _eye()
    e.ego_shift(_tex(2))
    e.take_efference(torch.tensor([-0.8, 0.3], dtype=DTYPE))
    dx, dy = e.ego_shift(_tex(3))                           # まったく別の画像（相関の山が無い）
    assert e.last_ego_mode == "pred" and (dx, dy) == pytest.approx(command_shift(-0.8, 0.3), abs=1e-3)


def test_image_mode_keeps_old_rule():
    e = _eye("image")
    a = _tex(4)
    e.ego_shift(a)
    e.take_efference(torch.tensor([0.5, 0.0], dtype=DTYPE))
    b = np.roll(a, 20, axis=1)
    assert e.ego_shift(b) == (20.0, 0.0) and e.last_ego_mode == "image"


def test_step_takes_copy_after_this_image():
    """写しが届いた回の画像は動く前。写しはその次の画像で使う。"""
    e = _eye()
    seen = []
    e.see = lambda img: (seen.append(e.ego_shift(img)), ([], torch.zeros(28, 28, dtype=DTYPE)))[1]
    e._ready = True
    a = _tex(5)
    cmd = torch.tensor([0.25, 0.0], dtype=DTYPE)
    e.step({"env": a, "orienteye": cmd})
    assert e.last_pred_px == (0.0, 0.0) and e._eff_pending[0] < 0      # この回の画像には使わず控える
    e.step({"env": np.roll(a, -30, axis=1), "orienteye": torch.zeros(2, dtype=DTYPE)})
    assert e.last_pred_px == pytest.approx(command_shift(0.25, 0.0), abs=1e-3) and seen[-1] == (-30.0, 0.0)
