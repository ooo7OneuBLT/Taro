"""共有テーマファイルに、別プロジェクト側の言葉が混入していないかを調べる。

【なぜ要るか、2026-08-14】run\\viewer_tools\\e_viewer_qt_theme.py は
太郎本体Viewer（taro_core外だが太郎プロジェクト側のファイル）と、
別リポジトリのPySide6ビューアが共有するテーマ機構である。
依存の向きは 別プロジェクト -> Taro の一方向だけが許されており（太郎側が正、
向こうが一方向で読みに行く）、太郎側のファイルに向こうの都合
（用語・パス）が紛れ込むと、この依存の向きが暗黙に逆転してしまう
（太郎を単体公開するときの事故にもなる）。

【何を見るか】
    対象ファイル（TARGET_FILES）の中身に、禁止語（FORBIDDEN_WORDS）が
    1語でも含まれていないかを機械的に調べる。

【何を見ないか（意図的に外している）】
    このファイル自身・run/tools配下の他の検査スクリプト。
    対象は「太郎側から共有されるテーマ機構」に限定する
    （将来チェック対象を増やす場合はTARGET_FILESに追記する）。

使い方:
    .venv/Scripts/python.exe run/tools/check_qt_theme_neutral.py
    -> 違反があれば標準エラーに一覧を出し、終了コード 1
       違反が無ければ "[ok] 中立命名チェック OK" を出し、終了コード 0
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# 将来、共有テーマ機構が複数ファイルに増えたらここへ追記する。
TARGET_FILES = (
    os.path.join(_ROOT, "run", "viewer_tools", "e_viewer_qt_theme.py"),
)

# 別プロジェクト側を指す言葉。太郎側のファイルに書いてはならない。
# 【なぜこの語のリストか】実際に混入した例をそのまま使う。将来、別の
# 向こう固有の語が見つかったらここへ追記する（本文が広く使う一般語
# 「テーマ」「設定」等は含めない＝偽陽性を避ける）。
FORBIDDEN_WORDS = (
    "組織",
    "organization",
    "部署",
    "稼働ログ",
    "想定外ログ",
    "トークン記録",
    "稼働中",
    "作業台帳",
)


def check_file(path: str) -> list[tuple[str, int, str]]:
    """1ファイルを調べ、(語, 行番号, 行の中身) のリストを返す（空なら違反なし）。"""
    violations = []
    if not os.path.isfile(path):
        # 対象ファイルが存在しないこと自体は、この検査の役割外（別の問題）。
        return violations
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for lineno, line in enumerate(lines, start=1):
        for word in FORBIDDEN_WORDS:
            if word in line:
                violations.append((word, lineno, line.rstrip("\n")))
    return violations


def main() -> int:
    all_violations = []
    for path in TARGET_FILES:
        for word, lineno, line in check_file(path):
            all_violations.append((path, word, lineno, line))

    if all_violations:
        print("[ng] 中立命名チェックで違反が見つかりました:", file=sys.stderr)
        for path, word, lineno, line in all_violations:
            rel = os.path.relpath(path, _ROOT)
            print(f"  {rel}:{lineno}: 禁止語 '{word}' -> {line}", file=sys.stderr)
        return 1

    print("[ok] 中立命名チェック OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
