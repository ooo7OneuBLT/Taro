# -*- coding: utf-8 -*-
"""【転送のみ】このファイルの中身は principles/homeostatic_scaling.py へ移した（2026-08-30の整理）。

既存の `from homeostatic_scaling import ...` を壊さないために、ここに転送だけを残している。
新しく書くコードは移動先を直接 import すること。
整理の全体像は `doc/脳の地図.md`。
"""
from principles.homeostatic_scaling import *          # noqa: F401,F403
from principles import homeostatic_scaling as _m   # noqa: E402

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
