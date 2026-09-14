# -*- coding: utf-8 -*-
"""【転送のみ】中身は `run/world/hand_in_view.py` へ移した（2026-09-13・段D）。

【なぜ移したか】ここは目標Eのフォルダだが、このファイルは目標F からも
実行基盤（run/）からも使われていた＝**実行基盤が目標フォルダに依存する**逆向きの形。
世界の側（環境・物・親）は目標をまたぐものなので `run/world/` へ置いた。

【名前が変わった理由】同じ名前のモジュールが2箇所にあると、どちらが読まれるかが
sys.path の順で変わる（二重import）。衝突しないよう移設先では `e_` を外してある。

新しく書くコードは `from run.world.hand_in_view import ...` を直接使うこと。
設計：doc/設計_太郎をCoreで完結させる_2026-09-13.md（段D）
"""
import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from run.world.hand_in_view import *            # noqa: F401,F403,E402
import run.world.hand_in_view as _m             # noqa: E402
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
