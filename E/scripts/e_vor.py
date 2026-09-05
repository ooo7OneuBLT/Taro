# -*- coding: utf-8 -*-
"""【転送のみ】中身は taro_core/src/brain/brainstem/vor.py へ移した（2026-08-30の整理）。

「太郎の身体はCoreで完結」の方針による。新しく書くコードは移動先を直接 import
すること。整理の全体像は `doc/脳の地図.md`。
"""
import os as _os
import sys as _sys
# 単体実行のプローブは core のパスを通していないことがあるので、4フォルダ全部を足す
# （移した先は senses/（retina）や body/ も import するため brain だけでは足りない）
_CORE = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir, "taro_core", "src"))
for _sub in ("wrapper", "senses", "brain", "body"):
    _p = _os.path.join(_CORE, _sub)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from brainstem.vor import *          # noqa: F401,F403,E402
import brainstem.vor as _m           # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
