# -*- coding: utf-8 -*-
"""反射3本と声道を taro_core へ移す（2026-08-30・「太郎の身体はCoreで完結」）。

【なぜ】反射（輻輳・定位・前庭動眼）は太郎の神経系そのものなのに、実験置き場の
E/scripts に居座っていた。声道は器官なのに brain/ にあった。
ユーザー指示：「太郎の身体、脳に関わるものはすべてcoreのなかで完結するようにして」。

【どこへ】人間の部位に合わせる（feedback-organ-level-naming）：
  e_vergence.py     → brain/brainstem/vergence.py   輻輳の中枢は脳幹
  e_vor.py          → brain/brainstem/vor.py        前庭動眼反射（前庭神経核）
  e_orienting_v2.py → brain/midbrain/orienting.py   定位反射（上丘＝中脳）
  brain/vocal_tract.py → body/vocal_tract.py        声道は脳ではなく器官

【安全策】reorganize_brain.py と同じ転送ファイル方式。元の場所に転送だけ残し、
既存の import（`import e_vergence` 等）は1つも壊さない。E/scripts 側の転送は
単体実行のスクリプトからも読めるよう、sys.path の追加を転送ファイル自身が行う。

    .venv/Scripts/python.exe taro_core/tools/move_reflexes_to_core.py
"""
import os
import io
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\claude\AI\Taro"
SRC = os.path.join(ROOT, "taro_core", "src")

# (元の絶対パス, 移す先の絶対パス, 転送 import に使うモジュール名)
PLAN = [
    (os.path.join(ROOT, "E", "scripts", "e_vergence.py"),
     os.path.join(SRC, "brain", "brainstem", "vergence.py"), "brainstem.vergence"),
    (os.path.join(ROOT, "E", "scripts", "e_vor.py"),
     os.path.join(SRC, "brain", "brainstem", "vor.py"), "brainstem.vor"),
    (os.path.join(ROOT, "E", "scripts", "e_orienting_v2.py"),
     os.path.join(SRC, "brain", "midbrain", "orienting.py"), "midbrain.orienting"),
    (os.path.join(SRC, "brain", "vocal_tract.py"),
     os.path.join(SRC, "body", "vocal_tract.py"), "vocal_tract"),
]

# E/scripts 用の転送（brainのパスを自分で通す。単体実行スクリプト対策）
SHIM_E = '''# -*- coding: utf-8 -*-
"""【転送のみ】中身は taro_core/src/brain/{rel} へ移した（2026-08-30の整理）。

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
from {mod} import *          # noqa: F401,F403,E402
import {mod} as _m           # noqa: E402
globals().update({{k: v for k, v in vars(_m).items() if not k.startswith("__")}})
'''

# brain 内用の転送（vocal_tract。body/ は sys.path で brain より先に来るので
#   実際にはこの転送は読まれないが、brain だけを path に足す古いスクリプト対策で残す）
SHIM_B = '''# -*- coding: utf-8 -*-
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
'''


def main():
    for src, dst, mod in PLAN:
        if not os.path.exists(src):
            print("飛ばす（元が無い）:", src)
            continue
        if os.path.exists(dst):
            print("飛ばす（先に既にある）:", dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        if "E\\scripts" in src or "E/scripts" in src:
            rel = os.path.relpath(dst, os.path.join(SRC, "brain")).replace("\\", "/")
            shim = SHIM_E.format(rel=rel, mod=mod)
        else:
            shim = SHIM_B
        io.open(src, "w", encoding="utf-8").write(shim)
        print("移した: %s → %s" % (os.path.relpath(src, ROOT), os.path.relpath(dst, ROOT)))


if __name__ == "__main__":
    main()
