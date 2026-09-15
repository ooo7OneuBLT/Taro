# -*- coding: utf-8 -*-
"""【転送のみ】中身は `run/world/scene.py` へ移した（2026-09-13・段D）。

【なぜ移したか】これは**環境を組み立てる関数**であってプラグイン（測る道具）ではない。
`run/plugins/` に置かれていたのは置き場の誤り。

新しく書くコードは `from run.world.scene import ...` を直接使うこと。
設計：doc/設計/設計_太郎をCoreで完結させる_2026-09-13.md（段D）
"""
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir, _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from run.world.scene import *                      # noqa: F401,F403,E402
import run.world.scene as _m                       # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
