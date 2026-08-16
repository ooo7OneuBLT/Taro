"""TaroMap（⑤区分一覧ページ）用：`doc/人間模倣からの逸脱リスト.md` の全文検索。

【なぜ「表としてパースしない」か、設計判断・案C/案D】
この文書は4000行を超えるが、統一された表形式では書かれていない（見出しの階層も
`#`〜`####`が混在し、追記が時系列でぶら下がる運用）。これを機械的に「表」として
解釈しようとすると、誤読（違う項目を同じ項目だと取り違える等）のリスクが高い。
⇒ ここでは**全文検索＋直近の見出しの位置**だけを返す、割り切った実装にする。
  該当が無ければ空リストを返し、呼び出し側（page_classification.py）が
  「未記載の可能性がある」という中立な表示を出すための材料にする。
  ここで「該当なし＝許容されている」等の安全側の判定は絶対にしない
  （空リストをそのまま返すだけで、意味づけは呼び出し側に委ねる）。
"""
from __future__ import annotations

import os
import re

_DEFAULT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir,
    "doc", "人間模倣からの逸脱リスト.md",
))

_HEADING_RE = re.compile(r"^#+\s*(.*)$")


def search_deviations(keywords, path: str | None = None) -> list[dict]:
    """逸脱リスト.md を全文検索する。

    Args:
        keywords: 文字列1つ、または文字列のイテラブル（複数キーワードのOR検索）。
        path: 検索対象ファイルのパス。省略時は既定の
            `doc/人間模倣からの逸脱リスト.md` を使う。

    Returns:
        [{"heading": str, "line_no": int, "matched_line": str, "keyword": str}, ...]
        該当が無ければ空リスト（安全側の代替表示は一切しない。中立表示は
        呼び出し側の責任）。ファイルが読めない場合も同様に空リストを返す
        （例外で落とすと呼び出し側のUIが構築できなくなるため、失敗は
        「該当なし」と見分けがつかない形でしか呼び出し側に返せない点に注意。
        ここでは静かに握りつぶさず、標準エラーに一言出す）。
    """
    if isinstance(keywords, str):
        keyword_list = [keywords]
    else:
        keyword_list = [k for k in keywords if k]
    keyword_list = [k for k in keyword_list if k.strip()]
    if not keyword_list:
        return []

    target_path = path or _DEFAULT_PATH
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:      # noqa: BLE001
        import sys
        print(f"注意[search_deviations] 逸脱リストを読めません: {e}", file=sys.stderr)
        return []

    hits: list[dict] = []
    current_heading = "（見出しの前）"
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        m = _HEADING_RE.match(line)
        if m:
            heading_text = m.group(1).strip()
            if heading_text:
                current_heading = heading_text
            continue
        for kw in keyword_list:
            if kw in line:
                hits.append({
                    "heading": current_heading,
                    "line_no": line_no,
                    "matched_line": line.strip(),
                    "keyword": kw,
                })
                break     # 1行で複数キーワードに当たっても重複登録しない
    return hits


if __name__ == "__main__":
    # 自己テスト：実在するはずのキーワードと、存在しないはずのキーワードで確認する。
    # 【なぜencode/decodeで一段挟むか】落とし穴チェックリスト項35：Windows既定コンソール
    #   （cp932）は逸脱リスト.md内の絵文字等を出力できずUnicodeEncodeErrorで落ちる。
    #   自己テストの表示だけの都合なので、表示できない文字は落として続行する。
    def _safe(s):
        return s.encode("cp932", errors="replace").decode("cp932")

    for kw in ("触覚", "これは存在しないはずのダミーキーワードXYZ123"):
        result = search_deviations(kw)
        print(_safe(f"keyword={kw!r}: {len(result)}件"))
        for row in result[:3]:
            print(_safe(f"    line {row['line_no']}: [{row['heading']}] "
                        f"{row['matched_line'][:60]}"))
