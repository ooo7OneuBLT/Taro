"""PC で太郎を回すときの負荷の決まり（core_v2 を import すると1回だけ効く。core_v2/__init__.py から呼ぶ）。

【なぜ1か所にまとめたか・2026-10-18】太郎の計算は小さい物ばかりなので、torch の計算スレッドは1本の方が速い
（voice_bed で 4本：1日8.3秒 → 1本：6.2秒）。この1行は voice_bed・noun_bed など一部の道具にしか入っておらず、
G/scripts/pass_test.py には無かった。そのため torch が物理コアの数だけ補助スレッドを立て、仕事の無いまま
起きる・眠るをくり返して CPU（特に E コア）を埋め、PC 全体が重くなった（毎秒70万回の切り替え）。
道具ごとに書くと抜けるので、core_v2 を使う入口すべてに効くようにここへ集める。

やること（どれも「速さ」と「PC の使いやすさ」だけを変え、太郎の計算の中身は変えない）：
1. 下で動く計算の部品（OpenMP・MKL・OpenBLAS など）のスレッド数も絞る。torch.set_num_threads は torch 自身の
   スレッドしか絞らないため（旧太郎 B/perf_setup.py で、24コア全部が薄く busy になる現象として観測済み）。
   これは numpy・torch を import する前に環境変数で決める必要がある
2. torch の計算スレッドを1本に（set_num_threads・set_num_interop_threads）
3. Windows では優先度を「通常以下」に下げ、ブラウザなどほかの作業を先に通す（旧太郎 B/perf_setup.py と同じ）

本数を変えたいときは環境変数 TARO_THREADS（既定 1）。
"""

from __future__ import annotations

import os
import sys

_done = False

_BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def threads() -> int:
    try:
        return max(1, int(os.environ.get("TARO_THREADS", "1")))
    except ValueError:
        return 1


def setup() -> None:
    global _done
    if _done:
        return
    _done = True
    n = threads()
    for v in _BLAS_VARS:
        os.environ.setdefault(v, str(n))
    # numpy が先に読み込まれていたら、環境変数は効かない。threadpoolctl があればそれで後から絞る
    if "numpy" in sys.modules:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(n)
        except Exception:  # noqa: BLE001  入っていなければそのまま
            pass
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)   # 並列の計算がもう始まっていると変えられない（そのときはそのまま）
        except RuntimeError:
            pass
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import psutil
            psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:  # noqa: BLE001
            pass
