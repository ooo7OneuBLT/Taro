"""場所の一元管理（案内板）。

全スクリプトはハードコードした絶対パスを書かず、ここから読む。
自分の居場所（__file__）を基準に計算するので、プロジェクトごと別の場所へ
移動しても動く。MIMo の置き場が変わったら、このファイルの MIMO_DIR だけ直す。
"""
import os
import sys

C_ROOT = os.path.dirname(os.path.abspath(__file__))              # = Taro/C
SRC = os.path.join(C_ROOT, "src")                               # 脳＋glue
TESTS = os.path.join(C_ROOT, "tests")
LOGS = os.path.join(C_ROOT, "logs")

# MIMo は C・D 共有の外部依存。目標フォルダの外（プロジェクト直下）に置く。
MIMO_DIR = os.path.abspath(os.path.join(C_ROOT, os.pardir, "MIMo"))  # = Taro/MIMo
SCENE = os.path.join(MIMO_DIR, "mimoEnv", "assets", "benchmarkv2_scene.xml")


def setup_brain_path():
    """脳モジュール（brain/senses/wrapper）と tests を import 可能にする。"""
    for sub in ("wrapper", "senses", "brain"):
        p = os.path.join(SRC, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    if TESTS not in sys.path:
        sys.path.insert(0, TESTS)
