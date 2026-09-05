# -*- coding: utf-8 -*-
"""【転送のみ】このファイルの中身は cerebral_cortex/parietal_lobe/intraparietal_sulcus.py へ移した（2026-09-04）。

「これ何？」「同じ物」の追跡は、人間では頭頂間溝（intraparietal sulcus, IPS）が
担うとされる働きに近いため、大脳皮質・頭頂葉の下へ移動した。
既存の `from brain.object_files import ...` を壊さないために、ここに転送だけを残している。
新しく書くコードは移動先を直接 import すること。
整理の全体像は `doc/脳の地図.md`。
"""
from cerebral_cortex.parietal_lobe.intraparietal_sulcus import *          # noqa: F401,F403
from cerebral_cortex.parietal_lobe import intraparietal_sulcus as _m      # noqa: E402

globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
