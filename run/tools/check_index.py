# -*- coding: utf-8 -*-
"""落とし穴チェックリストの索引が本体に追いついているかを調べる（2026-09-12）。

【なぜ作ったか】ユーザー指示「ノウハウは索引だけ絶対読むようにして、
  その索引も随時更新するようにして」。
  実測：索引の「最終同期」は 2026-08-14 のままで、それ以降に本体へ足した
  **32件が索引に1件も載っていなかった**。つまり索引を読んでも最近の教訓に当たらない。
  実害も出た（2026-09-12 に同型のミスを索引が止められなかった）。

  「次からは索引も更新する」と書いても守られないのは既に証明済みなので、
  **数えて差を出す機械**にする。本体に足して索引に足し忘れたら、ここが教える。

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_index
    .venv/Scripts/python.exe -m run.tools.check_index --quiet   # 差があるときだけ出す

【どう照合しているか】索引の各行には「見出し（Grep用）」が入っている。
  本体の見出し行のうち、**索引のどの検索文字列にも当たらないもの**を漏れとみなす。
  完全一致ではなく含むかどうかで見る（索引は短い文字列で本体をGrepする作りなので）。
"""
import argparse
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))
本体 = os.path.join(_ROOT, "doc", "検証の落とし穴チェックリスト.md")
索引 = os.path.join(_ROOT, "doc", "検証の落とし穴チェックリスト_索引.md")


# 索引に載せる必要が**無い**見出し（文書の骨組み・日付のまとめ見出し・小見出し）
_除外 = (
    r"^#{2,3} 段階",              # 「段階0：問題が起きたとき」など本体の区分け
    r"^#{2,3} 使い方",
    r"^#{2,3} \**ルール",         # 各項の中の小見出し
    r"^#{2,3} \**やること",
    r"^#{2,3} \**これは",
    r"^#{2,3} 実際に",
    r"に追加[：:]",               # 「2026-07-21 に追加：…」＝下にある項のまとめ見出し
    # すでに索引にある項の**中**の追記・補足（独立した落とし穴ではない）
    r"^#{2,3} (さらに|追記|この型は|なお)",
    r"に実際にやった",
)
# 索引に載せるべき見出し＝「項N」か「項（日付）」か、日付を含む具体的な事例
_項目 = r"項\s*[0-9（(]|\d{4}-\d{2}-\d{2}|【\d{4}"


def _は項目(見出し):
    if any(re.search(p, 見出し) for p in _除外):
        return False
    return bool(re.search(_項目, 見出し))


def _見出し(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            h = line.rstrip()
            if re.match(r"^#{2,3} ", h) and _は項目(h):
                out.append((i, h))
    return out


def _索引の検索文字列(path):
    """索引の表の1列目（|...|...|...| の最初の欄）を拾う。"""
    keys = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] and cells[0] != "見出し":
                keys.append(cells[0])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="漏れがあるときだけ出力する")
    a = ap.parse_args()

    for p in (本体, 索引):
        if not os.path.isfile(p):
            print("見つからない: %s" % p)
            return 2

    本体見出し = _見出し(本体)
    keys = _索引の検索文字列(索引)
    if not keys:
        print("索引から検索文字列を1つも拾えなかった（表の形が変わった？）: %s" % 索引)
        return 2

    漏れ = []
    for ln, h in 本体見出し:
        # 索引のどれかの検索文字列が、この見出し行に含まれていれば載っている
        if not any(k and k in h for k in keys):
            漏れ.append((ln, h))

    if not a.quiet or 漏れ:
        print("本体 %s" % os.path.relpath(本体, _ROOT))
        print("  見出し %d 件 / 索引の項目 %d 件" % (len(本体見出し), len(keys)))
    if not 漏れ:
        if not a.quiet:
            print("  索引は本体に追いついている（漏れ 0 件）")
        return 0

    print("")
    print("  索引に無い見出しが %d 件あります。索引に1行ずつ足してください：" % len(漏れ))
    for ln, h in 漏れ:
        print("    %s:%d  %s" % (os.path.relpath(本体, _ROOT), ln, h[:90]))
    print("")
    print("  索引: %s" % os.path.relpath(索引, _ROOT))
    print("  形式: |見出し（Grep用）|要約（20〜35字）|読むとき|")
    print("  置く節は「その落とし穴が**防げる場面**」（段階0〜5のどれか）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
