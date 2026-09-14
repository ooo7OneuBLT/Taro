# -*- coding: utf-8 -*-
"""【転送のみ】中身は `run/world/body_config.py` へ移した（2026-09-13・段D）。

【なぜ移したか】体型（月齢ごとの体の大きさ・筋の強さ）の設定。
太郎の身体が置かれる世界の側の情報で、目標Eだけのものではない
（`run/world/toy_env.py` と `run/viewer_tools/e_viewer.py` も使う）。

新しく書くコードは `from run.world.body_config import ...` を直接使うこと。
設計：doc/設計_太郎をCoreで完結させる_2026-09-13.md（段D）
"""
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from run.world.body_config import *                      # noqa: F401,F403,E402
import run.world.body_config as _m                       # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
