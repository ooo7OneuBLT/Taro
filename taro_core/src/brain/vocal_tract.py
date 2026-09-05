# -*- coding: utf-8 -*-
"""【転送のみ】中身は body/vocal_tract.py へ移した（2026-08-30の整理）。

声道は脳ではなく器官。「太郎の身体はCoreで完結」の方針による。
"""
# 注意：`from vocal_tract import *` と書くと、この転送ファイル自身が
#   「vocal_tract」を名乗っている場面（brainだけを path に足した古いプローブ）で
#   自分を import して空になる。importlib でファイルを直接読む方式にする。
import os as _os
import importlib.util as _ilu
_real = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, "body", "vocal_tract.py"))
_spec = _ilu.spec_from_file_location("_vocal_tract_body", _real)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
