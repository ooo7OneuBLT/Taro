# -*- coding: utf-8 -*-
"""【転送のみ】中身は `taro_core/src/brain/midbrain/orienting_v1.py` へ移した（2026-09-13・段D）。

【なぜ移したか】これは**視線誘導反射の旧版（v1）の実装そのもの**で、太郎の脳。
新版（v2）は 2026-08-30 に `taro_core/src/brain/midbrain/orienting.py` へ移っていたのに、
旧版だけが目標Eのフォルダに残っていた。`E_ORIENT_V=1` で今も復活できる
（ただし 363 のシーンすべてが v2 を使っており、既定も v2）。

新しく書くコードは `from brain.midbrain.orienting_v1 import ...` を直接使うこと。
設計：doc/設計_太郎をCoreで完結させる_2026-09-13.md（段D）
"""
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from brain.midbrain.orienting_v1 import *                      # noqa: F401,F403,E402
import brain.midbrain.orienting_v1 as _m                       # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
