"""色のコントラスト比を測る（Qt非依存の純粋関数）。

WCAG 2.0 の定義通りに実装する。文字色は 4.5:1 以上、図形（アイコン・枠線など
文字を含まないもの）は 3:1 以上を推奨値とする。

`if __name__ == "__main__":` で `wiring_viewer.theme.THEMES` の role_* 系の色を
対応する背景色（card）に対して測り、しきい値を満たすか報告する自己テストを行う。
"""


def relative_luminance(r: float, g: float, b: float) -> float:
    """WCAG 2.0 の相対輝度。r,g,b は 0-255 のsRGB値。"""
    def _chan(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = _chan(r), _chan(g), _chan(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def contrast_ratio(rgb1, rgb2) -> float:
    """WCAG 2.0 のコントラスト比。(L1+0.05)/(L2+0.05)、L1が明るい方。"""
    l1 = relative_luminance(*rgb1)
    l2 = relative_luminance(*rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# role_provisional（自動判定・参考表示）は文字用途にも使うことがあるため、
#   文字用のしきい値(4.5)で判定する。図形専用にしか使わない役なら3.0でも足りるが、
#   より厳しい基準を満たしておけばどちらの用途でも安全という判断。
_TEXT_ROLES = ("role_innate", "role_emergent", "role_tool",
               "role_unclassified", "role_deviation", "role_provisional")
_TEXT_THRESHOLD = 4.5
_GRAPHIC_THRESHOLD = 3.0


def _self_test():
    # テーマの定義は同じ wiring_viewer パッケージの中の theme.py にある
    #   （run/viewer_tools/e_viewer_qt_theme.py という同名の既存ファイルが
    #   別プロジェクト用に既にあったため、衝突を避けてここに置いた。
    #   main_window.py のコメント・作業記録に経緯を記載）。
    from run.viewer_tools.wiring_viewer.theme import THEMES

    all_ok = True
    for theme_name, colors in THEMES.items():
        card = _hex_to_rgb(colors["card"])
        print(f"[{theme_name}] card={colors['card']}")
        for role in _TEXT_ROLES:
            if role not in colors:
                continue
            rgb = _hex_to_rgb(colors[role])
            ratio = contrast_ratio(rgb, card)
            ok = ratio >= _TEXT_THRESHOLD
            all_ok = all_ok and ok
            mark = "OK" if ok else "NG"
            print(f"  {role:22s} {colors[role]}  vs card  "
                  f"ratio={ratio:.2f}  (>=4.5?) {mark}")
    print()
    if all_ok:
        print("すべての role_* 色が文字用しきい値(4.5:1)を満たしている。")
    else:
        print("しきい値を満たさない色がある。THEMES を調整すること。")
    return all_ok


if __name__ == "__main__":
    import sys
    ok = _self_test()
    sys.exit(0 if ok else 1)
