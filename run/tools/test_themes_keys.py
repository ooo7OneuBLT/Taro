"""e_viewer_qt_theme.THEMES の各テーマが、キーの集合が完全一致しているかを調べる。

【なぜ要るか、2026-08-14】THEMES辞書にテーマ（例: "windows11_dark"）を
追加するとき、キーを1つ足し忘れると、片方のテーマだけ色がおかしくなる
という目で見て気づきにくい不具合になる（作業記録（非公開）
2026-08-14_別リポジトリのビューアPySide6化詳細設計_統合版.md 6-1節）。
この検査は、キー集合を機械的に突き合わせることでそれを防ぐ。

使い方:
    .venv/Scripts/python.exe run/tools/test_themes_keys.py
    -> キー集合が一致していれば "[ok] ..." を出して終了コード 0
       不一致があれば AssertionError を投げて終了コード 1
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_VIEWER_TOOLS = os.path.join(_ROOT, "run", "viewer_tools")
if _VIEWER_TOOLS not in sys.path:
    sys.path.insert(0, _VIEWER_TOOLS)

from e_viewer_qt_theme import THEMES  # noqa: E402


def main() -> int:
    names = list(THEMES.keys())
    assert len(names) >= 1, "THEMESが空です"

    base_name = names[0]
    base_keys = set(THEMES[base_name].keys())

    for name in names[1:]:
        keys = set(THEMES[name].keys())
        missing = base_keys - keys
        extra = keys - base_keys
        assert not missing and not extra, (
            f"テーマ '{name}' のキー集合が '{base_name}' と一致しません。"
            f" 不足: {sorted(missing)} / 余分: {sorted(extra)}"
        )

    print(f"[ok] test_themes_keys: {names} のキー集合が全て一致（{len(base_keys)}キー）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
