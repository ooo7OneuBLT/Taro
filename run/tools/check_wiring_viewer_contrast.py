"""TaroMap（配線可視化アプリ）のテーマ配色が、コントラスト比の基準を満たすかを検証する。

`run/viewer_tools/wiring_viewer/contrast_check.py` の relative_luminance() /
contrast_ratio() をそのまま使う（式の二重実装をしない）。

【何を測るか】
    `theme.THEMES` の全テーマについて、role_* 系の色（バッジの文字色として使う）を
    対応する card 色（バッジの背景）に対して測る。
    文字用しきい値 4.5:1（role_innate等、実際に文字として使う色）。
    図形用しきい値 3:1（accent色。ボタン等の地色・線として使う想定で、
        文字ほど厳しくない基準で足りる）。

使い方（既存の run/tools/check_*.py 群と同じ流儀）:
    .venv\\Scripts\\python.exe run\\tools\\check_wiring_viewer_contrast.py
    -> pytestからも `pytest run/tools/check_wiring_viewer_contrast.py` で呼べる。
       失敗があれば非0終了コード。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.viewer_tools.wiring_viewer.contrast_check import (  # noqa: E402
    _hex_to_rgb,
    contrast_ratio,
)
from run.viewer_tools.wiring_viewer.theme import THEMES  # noqa: E402

# 文字として使う役（バッジの文字色）。厳しい基準の4.5:1を要求する。
_TEXT_ROLES = ("role_innate", "role_emergent", "role_tool",
               "role_unclassified", "role_deviation", "role_provisional")
_TEXT_THRESHOLD = 4.5

# 図形（アイコン・枠線・地色）として使う役。3:1で足りる。
_GRAPHIC_ROLES = ("accent",)
_GRAPHIC_THRESHOLD = 3.0


def test_all_themes_exist():
    assert "windows11" in THEMES
    assert "windows11_dark" in THEMES
    assert len(THEMES) >= 2


def test_text_roles_meet_threshold():
    """role_* の文字色が、対応するcard背景に対して4.5:1以上あるか。"""
    failures = []
    for theme_name, colors in THEMES.items():
        card = _hex_to_rgb(colors["card"])
        for role in _TEXT_ROLES:
            if role not in colors:
                continue
            rgb = _hex_to_rgb(colors[role])
            ratio = contrast_ratio(rgb, card)
            if ratio < _TEXT_THRESHOLD:
                failures.append(
                    f"[{theme_name}] {role}={colors[role]} vs card={colors['card']} "
                    f"ratio={ratio:.2f} < {_TEXT_THRESHOLD}")
    assert not failures, "コントラスト不足（文字用4.5:1未達）:\n" + "\n".join(failures)


def test_graphic_roles_meet_threshold():
    """accent色が、bg背景に対して3:1以上あるか（ボタン等の図形用途）。"""
    failures = []
    for theme_name, colors in THEMES.items():
        bg = _hex_to_rgb(colors["bg"])
        for role in _GRAPHIC_ROLES:
            if role not in colors:
                continue
            rgb = _hex_to_rgb(colors[role])
            ratio = contrast_ratio(rgb, bg)
            if ratio < _GRAPHIC_THRESHOLD:
                failures.append(
                    f"[{theme_name}] {role}={colors[role]} vs bg={colors['bg']} "
                    f"ratio={ratio:.2f} < {_GRAPHIC_THRESHOLD}")
    assert not failures, "コントラスト不足（図形用3:1未達）:\n" + "\n".join(failures)


def test_ink_on_card_readable():
    """本文の文字色(ink)がcard背景に対して4.5:1以上あるか（最重要の基本組み合わせ）。"""
    failures = []
    for theme_name, colors in THEMES.items():
        ratio = contrast_ratio(_hex_to_rgb(colors["ink"]), _hex_to_rgb(colors["card"]))
        if ratio < _TEXT_THRESHOLD:
            failures.append(f"[{theme_name}] ink vs card ratio={ratio:.2f}")
    assert not failures, "本文色のコントラスト不足:\n" + "\n".join(failures)


def _run_all() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"[ok] {t.__name__}")
        except AssertionError as exc:
            failed.append(t.__name__)
            print(f"[ng] {t.__name__}: {exc}", file=sys.stderr)
    if failed:
        print(f"\n{len(failed)}件失敗: {failed}", file=sys.stderr)
        return 1
    print(f"\nすべて合格（{len(tests)}件）")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
