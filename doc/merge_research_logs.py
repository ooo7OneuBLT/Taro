"""目標A〜Eの研究日誌を1つのファイルにまとめる（読む用のコピーを作る）。

【使い方】
    python doc/merge_research_logs.py
出力： doc/研究日誌まとめ_A-E.md

【方針】
- **元ファイルは一切変更しない**。これは読むためのコピーを作るだけ。
- **更新はユーザーの指示があったときだけ**実行する（自動では走らせない）。
  日誌は各目標フォルダのものが**正**で、このまとめは常に古くなりうる。
- 見出しレベルだけ1段下げる（`#` → `##`）。全体に1つの `#` を置いて目次を機能させるため。
  注意：レベル6を超える見出しは下げない（Markdownの上限）。**文言は一切変えない**。

【含めないもの】
- `A/docs/ユーザー研究日誌.md` … ユーザー自身が書いた日誌で性質が違う（日付＋「やったこと」形式）。
  必要になったら SOURCES に足す。
"""
import datetime
import io
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))

# (目標, パス, 一行説明) — 順序がそのまま出力の順序になる
SOURCES = [
    ("A", "A/docs/研究日誌.md",  "最初の実装。感覚運動の土台"),
    ("B", "B/docs/研究日誌.md",  "初語（泣く・飲む・発話）"),
    ("C", "C/docs/研究日誌.md",  "MIMo統合・自己モデル"),
    ("D", "D/docs/D研究日誌.md", "他者理解"),
    ("E", "E/docs/研究日誌.md",  "運動発達"),
]

OUT = os.path.join(_HERE, "研究日誌まとめ_A-E.md")


def demote(text, by=1):
    """見出しレベルを by 段下げる。6を超えるものは下げない。文言は変えない。"""
    out = []
    in_code = False
    for line in text.split("\n"):
        # コードブロック内の `#` はコメントなので触らない
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if not in_code:
            m = re.match(r"^(#{1,6})(\s)", line)
            if m and len(m.group(1)) + by <= 6:
                line = "#" * (len(m.group(1)) + by) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out)


def main():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []
    parts.append("# 研究日誌まとめ（目標A〜E）\n")
    parts.append(
        f"注意**これは読むためのコピーです。**`{os.path.basename(__file__)}` で自動生成しました"
        f"（生成日時 {stamp}）。\n\n"
        "- **編集しないでください。** 直しても次の生成で消えます。\n"
        "- **正しいのは各目標フォルダの元ファイル**です。このまとめは生成した時点のもので、"
        "以後の追記は反映されていません。\n"
        "- 更新するとき： `python doc/merge_research_logs.py`\n"
        "- 見出しレベルは1段下げてあります（目次を機能させるため）。**文言は元のままです**。\n"
    )

    # 目次
    parts.append("\n## 目次\n")
    rows = ["| 目標 | 内容 | 元ファイル | 行数 |", "|---|---|---|---|"]
    bodies = []
    for goal, rel, desc in SOURCES:
        path = os.path.join(_ROOT, rel)
        if not os.path.exists(path):
            rows.append(f"| {goal} | {desc} | 注意**見つかりません**: `{rel}` | — |")
            bodies.append((goal, rel, desc, None))
            continue
        text = io.open(path, encoding="utf-8").read()
        n = text.count("\n") + 1
        anchor = f"目標{goal}"
        rows.append(f"| [{goal}](#{anchor}) | {desc} | `{rel}` | {n:,} |")
        bodies.append((goal, rel, desc, text))
    parts.append("\n".join(rows) + "\n")

    total = sum((t.count("\n") + 1) for _, _, _, t in bodies if t)
    parts.append(f"\n合計 **{total:,}行**。\n")

    # 本体
    for goal, rel, desc, text in bodies:
        parts.append("\n\n---\n")
        parts.append(f"\n## 目標{goal}\n")
        parts.append(f"> {desc} ／ 元ファイル: `{rel}`\n")
        if text is None:
            parts.append("\n注意**元ファイルが見つかりませんでした。**\n")
            continue
        parts.append("\n" + demote(text, by=1) + "\n")

    io.open(OUT, "w", encoding="utf-8").write("".join(parts))
    print(f"生成: {OUT}")
    print(f"  {len(SOURCES)}本の日誌 / 合計 {total:,}行 / "
          f"{os.path.getsize(OUT)/1024:.0f} KB")
    for goal, rel, _, text in bodies:
        mark = "OK" if text else "MISSING"
        print(f"  [{mark}] 目標{goal}: {rel}")


if __name__ == "__main__":
    main()
