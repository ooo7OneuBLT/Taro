# -*- coding: utf-8 -*-
"""【転送のみ】このファイルの中身は drives/imitation_reward.py へ移した（2026-08-30の整理）。

既存の `from imitation_reward import ...` を壊さないために、ここに転送だけを残している。
新しく書くコードは移動先を直接 import すること。
整理の全体像は `doc/脳の地図.md`。
"""
from drives.imitation_reward import *          # noqa: F401,F403
from drives import imitation_reward as _m   # noqa: E402

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
