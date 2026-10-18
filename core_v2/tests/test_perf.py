"""負荷の決まり（core_v2/perf.py）：core_v2 を import しただけで torch のスレッドが1本になる。"""
import os

import torch

import core_v2  # noqa: F401


def test_one_thread():
    want = int(os.environ.get("TARO_THREADS", "1"))
    assert torch.get_num_threads() == want
