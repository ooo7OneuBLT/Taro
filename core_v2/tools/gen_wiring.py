"""構造図（drawio「やさしい版」）から配線表 core_v2/wiring_table.py を書き出す。

使い方（リポジトリ直下で）:
    python -m core_v2.tools.gen_wiring <drawioのパス>

入力の図は `doc/drawio/脳の構造_試作.drawio`（ページ「やさしい版」）か、
その写しを含む `core_v2_段階2_設計図.drawio`。どちらでも同じ表になる。

* 係＝ id が `n_` で始まる UserObject のうち、中に別の `n_*` を持たないもの（41個）
* 矢印＝ source と target がどちらも係の edge（「やさしい版」で117本）
* 図に手で足した矢印（wiring.ADDED_EDGES）は、図に描いてあっても表からは外す。
  wiring.py の側で `added=True` の印付きで足すため（どこが図の書き漏れの修正かを残す）

手で書き足さないこと。図を直したらこのスクリプトを再実行する。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "wiring_table.py"
WIRING = Path(__file__).resolve().parents[1] / "wiring.py"


def _clean(label: str) -> str:
    text = re.sub(r"<[^>]+>", "", html.unescape(label or ""))
    return re.sub(r"\s+", " ", text).strip()


def _find_page(root: ET.Element) -> ET.Element:
    for diagram in root.iter("diagram"):
        if any(uo.get("id") == "n_brain" for uo in diagram.iter("UserObject")):
            return diagram
    raise SystemExit("図に n_brain（やさしい版）が見つからない")


def parse(path: Path):
    page = _find_page(ET.parse(path).getroot())
    objs = {}  # id -> (label, parent)
    for uo in page.iter("UserObject"):
        oid = uo.get("id") or ""
        cell = uo.find("mxCell")
        if oid.startswith("n_") and cell is not None and cell.get("edge") != "1":
            objs[oid] = (_clean(uo.get("label")), cell.get("parent"))
    parents = {p for _, p in objs.values()}
    leaves = [oid for oid in objs if oid not in parents]

    boxes = []
    for oid in leaves:
        label, parent = objs[oid]
        group = objs.get(parent, ("", None))[0]
        boxes.append((oid[2:], label, group))

    leaf_set = set(leaves)
    edges = []
    for cell in page.iter("mxCell"):
        if cell.get("edge") != "1":
            continue
        src, dst = cell.get("source"), cell.get("target")
        if src not in leaf_set or dst not in leaf_set:
            raise SystemExit(f"係でない所につながる矢印がある: {src} -> {dst}")
        edges.append((src[2:], dst[2:]))
    return boxes, edges


def _added_edges():
    """wiring.py の ADDED_EDGES を読む（torch を読み込まずに済むよう、文字として解析する）。"""
    tree = ast.parse(WIRING.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ADDED_EDGES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("wiring.py に ADDED_EDGES が無い")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("drawio", type=Path)
    args = ap.parse_args(argv)

    boxes, edges = parse(args.drawio)
    added = {(e[0], e[1]) for e in _added_edges()}
    edges = [e for e in edges if e not in added]
    if len(set(edges)) != len(edges):
        raise SystemExit("同じ向きの矢印が2本ある（係の受け口が区別できない）")

    digest = hashlib.sha256(args.drawio.read_bytes()).hexdigest()[:16]
    lines = [
        '"""配線表（自動生成。手で書き足さない）。',
        "",
        "生成: python -m core_v2.tools.gen_wiring <drawio>",
        f"元の図: {args.drawio.name}（sha256 先頭16桁 {digest}）",
        f"係 {len(boxes)} 個・矢印 {len(edges)} 本（手で足した矢印は wiring.py 側）",
        '"""',
        "",
        "# (係id, 名前, まとまり)",
        "BOXES = [",
        *[f"    ({b[0]!r}, {b[1]!r}, {b[2]!r})," for b in boxes],
        "]",
        "",
        "# (出す係, 受ける係)。図での並び順",
        "EDGES = [",
        *[f"    ({s!r}, {d!r})," for s, d in edges],
        "]",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT}: 係 {len(boxes)}・矢印 {len(edges)}")


if __name__ == "__main__":
    main()
