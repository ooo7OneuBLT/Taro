"""場所の一元管理（案内板）。

全スクリプトはハードコードした絶対パスを書かず、ここから読む。
自分の居場所（__file__）を基準に計算するので、プロジェクトごと別の場所へ
移動しても動く。MIMo の置き場が変わったら、このファイルの MIMO_DIR だけ直す。

配置：`Taro/taro_core/`（太郎の脳＋感覚＋内臓＋2体環境の中立な置き場）。
以前は `Taro/C/paths.py` にあったが、脳は「目標Cのもの」でなく「太郎そのもの」なので
中立化した（移行記録：doc/移行記録_taro_core化_2026-07-17.md）。
"""
import os
import sys

CORE_ROOT = os.path.dirname(os.path.abspath(__file__))          # = Taro/taro_core
SRC = os.path.join(CORE_ROOT, "src")                            # 脳＋感覚＋内臓
TESTS = os.path.join(CORE_ROOT, "tests")
# 実験ログは各目標フォルダ側（C/logs, D/logs 等）に置く。ここは純粋なコード置き場。
# LOGS は後方互換のため残すが、現状どのスクリプトからも参照されていない。
LOGS = os.path.join(CORE_ROOT, "logs")

# MIMo は全目標共有の外部依存。目標フォルダの外（プロジェクト直下）に置く。
# taro_core も Taro 直下なので、os.pardir で Taro/MIMo に届く（Cの時と同じ）。
MIMO_DIR = os.path.abspath(os.path.join(CORE_ROOT, os.pardir, "MIMo"))  # = Taro/MIMo
SCENE = os.path.join(MIMO_DIR, "mimoEnv", "assets", "benchmarkv2_scene.xml")


def setup_brain_path():
    """脳モジュール（brain/senses/wrapper）と tests を import 可能にする。

    src/ はフラット構成（__init__.py 無し）なので、各サブフォルダを直接 sys.path に
    載せて素の名前で import できるようにする。body は hybrid_env.py が自分で載せる。
    """
    for sub in ("wrapper", "senses", "brain"):
        p = os.path.join(SRC, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if TESTS not in sys.path:
        sys.path.insert(0, TESTS)
