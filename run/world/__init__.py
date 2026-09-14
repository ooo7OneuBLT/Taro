# -*- coding: utf-8 -*-
"""世界の側：太郎の身体が置かれている環境・物・親（2026-09-13・段D）。

太郎（taro_core）でも、測る道具（run/plugins）でもないもの。
元は `E/scripts/`（目標Eのフォルダ）にあったが、目標F からも実行基盤からも
使われていて「実行基盤が目標フォルダに依存する」逆向きの形になっていた。

元の場所には「転送のみ」のシムを残してあるので、`from e_toy_env import ...`
と書いた既存のスクリプトはそのまま動く。新しく書くコードはここを直接使う。
設計：doc/設計_太郎をCoreで完結させる_2026-09-13.md（段D）
"""
import os as _os
import sys as _sys

# 【段D・2026-09-13・やり残し】ここに移したファイルは、まだ E/scripts に残っている
#   ものを素の名前で読んでいる（`e_body_config`＝体型の設定、`e_vor`/`e_orienting`/
#   `e_orienting_v2`/`e_vergence`＝反射。ただし反射3本は taro_core への転送シム）。
#   それらも整理できるまでの間、ここで道を通しておく。
#   ⇒ 片付いたらこのブロックごと消す。台帳「6. 置き場の誤り」参照。
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       _os.pardir, _os.pardir))
#   実際に読んでいる素の名前（機械で全部数えた）：
#     D/scripts   … d_supine_env（MIMoの土台の環境）・d_c5_motor_quality
#     E/scripts   … e_body_config（体型）・e_vor/e_orienting/e_orienting_v2/e_vergence（反射。
#                   ただし3本は taro_core への転送シム）
#     taro_core   … paths（MIMoの置き場）
#     taro_core/src/* … development・infant_body・semicircular_canals・spinal_cord
_CORE = _os.path.join(_ROOT, "taro_core")
_PATHS = [_ROOT, _CORE,
          _os.path.join(_ROOT, "E", "scripts"),
          _os.path.join(_ROOT, "D", "scripts"),
          _os.path.join(_ROOT, "C", "scripts")]
_PATHS += [_os.path.join(_CORE, "src", _s)
           for _s in ("wrapper", "senses", "brain", "body")]
for _p in _PATHS:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
