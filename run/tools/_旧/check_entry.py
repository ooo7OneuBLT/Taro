"""入口以外から実験していないかを機械的に調べる。

【なぜ要るか、2026-07-30】この日1日で、同じ型の事故を6つ踏んだ。

    1  基準の汚染          四肢の筋力補正の基準を、体を作る途中で測っていた（項76）
    2  保存で消える状態    保存されるのは重みだけで、経験バッファは消えていた（項75）
    3  次元の希釈          目標が621次元まるごと＝一度直したバグと同じ構造（項77）
    4  駆動モードの分裂  学習は関節モード、測定は筋肉モードで別の体だった
    5  nan の誤読          測れなかったのを「動かせない」と表示した（項78）
    6  この検査の最初の版が、自分の説明文を違反として検出した（偽陽性12件）

どれも「人が気をつける」では防げなかった。機械が見つける形にする。
注意：ただし6が示すとおり、**偽陽性だらけの検査は使われなくなる**。
  疑わしいものを全部挙げるのではなく、確実なものだけを挙げる。

【何を見るか】
  実験・測定のコードが、環境を自分で組み立てていないか。
  組み立ててよいのは run/plugins/common/scene.py だけ。

【何を見ないか（意図的に外している）】
  ・E/scripts など各目標の scripts    古い方式。削除せず残す方針（設計 §6）
  ・taro_core/src/body, brain          体の定義＝駆動モードを**受け取って使う**側
  ・関数定義の行（def ...）            引数として受け取るのは違反ではない

使い方:
    .venv/Scripts/python.exe run/tools/check_entry.py
    → 違反があれば一覧を出し、終了コード 1
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

EXEMPT_DIRS = (
    os.path.join("E", "scripts"),
    os.path.join("C", "scripts"),
    os.path.join("D", "scripts"),
    os.path.join("A", "scripts"),
    os.path.join("B", "scripts"),
    os.path.join("taro_core", "src", "body"),
    os.path.join("taro_core", "src", "brain"),
    os.path.join("taro_core", "src", "senses"),
    os.path.join("taro_core", "src", "wrapper"),
    os.path.join("taro_core", "tests"),
    os.path.join("taro_core", "tools"),
    ".venv", "MIMo", "__pycache__", ".git", "node_modules",
)
# 環境を組み立ててよい唯一の場所＋この検査自身
EXEMPT_FILES = (
    os.path.join("run", "plugins", "common", "scene.py"),
    os.path.join("run", "tools", "check_entry.py"),
)

RULES = (
    (re.compile(r"\b(SupineMimoEnv|ToySupineEnv|gym\.make)\s*\("),
     "環境を直接組み立てている",
     "run/plugins/common/scene.py を通す（実験ファイルの scene 欄で指定）"),
    # 注意：`actuation_model=None`（受け取る側の既定値）は違反ではない。
    #   何かを**渡している**呼び出しだけを見る。
    (re.compile(r"actuation_model\s*=\s*(?!None\b)[A-Za-z_]"),
     "駆動モードを直接渡している",
     "シーンの既定（筋肉モード）に従う。関節モードは実験ファイルの "
     "taro.actuation: \"joint\" で明示する（逸脱リスト 逸脱5）"),
)


def _exempt_dir(rel):
    sep = os.sep
    padded = f"{sep}{rel}"
    return any(rel.startswith(d) or f"{sep}{d}{sep}" in padded for d in EXEMPT_DIRS)


def scan():
    bad = []
    checked = 0
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", ".venv", "MIMo")]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), _ROOT)
            if _exempt_dir(rel) or rel in EXEMPT_FILES:
                continue
            checked += 1
            try:
                text = open(os.path.join(dirpath, fn), encoding="utf-8",
                            errors="replace").read()
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                st = line.strip()
                if not st or st.startswith("#"):
                    continue
                if st.startswith("def ") or st.startswith("async def "):
                    continue          # 受け取る側の定義は違反ではない
                for pat, what, how in RULES:
                    if pat.search(line):
                        bad.append((rel, i, what, how, st[:88]))
    return checked, bad


def main():
    checked, bad = scan()
    print("=" * 78)
    print(" 入口の検査 ── 実験は run/main.py を通っているか")
    print("=" * 78)
    print(f"  調べたファイル {checked} 本")
    print("  除外: 各目標の scripts（古い方式）／taro_core（体と脳の定義）")
    if not bad:
        print("\n  ✓ 違反なし")
        return 0
    print(f"\n  違反 {len(bad)} 件")
    for rel, i, what, how, src in bad:
        print(f"\n    {rel}:{i}  {what}")
        print(f"      > {src}")
        print(f"      {how}")
    print("\n  設計: E/docs/実行基盤_設計.md")
    return 1


if __name__ == "__main__":
    sys.exit(main())
