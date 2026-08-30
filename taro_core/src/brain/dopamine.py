# -*- coding: utf-8 -*-
"""【転送のみ】このファイルの中身は neuromodulator/dopamine.py へ移した（2026-08-30の整理）。

既存の `from dopamine import ...` を壊さないために、ここに転送だけを残している。
新しく書くコードは移動先を直接 import すること。
整理の全体像は `doc/脳の地図.md`。
"""
from neuromodulator.dopamine import *          # noqa: F401,F403
from neuromodulator import dopamine as _m   # noqa: E402

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
