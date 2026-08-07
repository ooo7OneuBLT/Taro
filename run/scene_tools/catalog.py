# -*- coding: utf-8 -*-
"""シーン一覧（`E/docs/シーン一覧.md`）を自動生成する道具。

【なぜ要るか、2026-08-05】既存の `E/docs/シーン一覧.md` は手書きで、
シーンJSON側に新しいシーンを足しても一覧側の更新を忘れる問題があった。
実際に11個目のシーン（`リーチング_リクライニング0度_自己接触版`）が
既存の手書き一覧に反映されないまま「全10個」と書かれ続けていた。

このファイルは `run/scenes/` 配下の全シーンJSONと `E/experiments/` 配下の
全実験ファイルから、その場で表を作り直す。「目的の要約」「流用時の注意」
といった人が書いた自由文の説明は機械的に導出できないので、無理に生成しない
（各シーンJSONの `note` フィールドをそのまま注記として載せる）。

呼び出され方は2通り：
    ① 単体実行   python run/scene_tools/catalog.py
    ② e_scene.save() の末尾から自動で呼ばれる（保存のたびに一覧を作り直す）

出力先は変更せず、従来どおり `E/docs/シーン一覧.md`。
"""
import os
import sys
import glob
import json

sys.stdout.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import e_scene  # noqa: E402

SCENE_DIR = e_scene.SCENE_DIR
EXPERIMENTS_DIR = os.path.join(_ROOT, "E", "experiments")
OUT_PATH = os.path.join(_ROOT, "E", "docs", "シーン一覧.md")

GROUP_JP = {"arm": "腕", "finger": "指", "leg": "脚", "trunk": "体幹", "head": "頭"}


def _experiments_by_scene():
    """E/experiments 配下の各JSONの "scene" キーを見て、シーン名 → 使っている
    実験ファイル名の一覧を集計する。読めない・"scene"キーが無いファイルは
    例外で全体を止めずスキップする。
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as fp:
                exp = json.load(fp)
        except Exception:
            continue
        scene_name = exp.get("scene")
        if not scene_name:
            continue
        out.setdefault(scene_name, []).append(os.path.basename(path))
    return out


def _settings_jp(summary):
    """constraint_summary() の戻り値を日本語の一言に並べる。"""
    parts = []
    parts.append(f"リクライニング{summary['recline_deg']:g}度")
    parts.append("おもちゃあり" if summary["toy_enabled"] else "おもちゃなし")
    if summary["root_pinned"]:
        parts.append("体そのものを固定（root_pinned）")
    if summary["pinned_groups"]:
        jp = "・".join(GROUP_JP.get(g, g) for g in summary["pinned_groups"])
        parts.append(f"固定={jp}")
    if summary["free_groups"]:
        jp = "・".join(GROUP_JP.get(g, g) for g in summary["free_groups"])
        parts.append(f"自由={jp}")
    parts.append("flexionあり" if summary["flexion"] else "flexionなし")
    return "／".join(parts)


def build_rows():
    """シーン名ごとの表の1行分を、辞書のリストとして組み立てる。"""
    exp_map = _experiments_by_scene()
    rows = []
    for name in e_scene.list_scenes():
        try:
            scene = e_scene.load(name)
        except Exception as e:
            rows.append({
                "name": name, "error": str(e),
            })
            continue
        summary = e_scene.constraint_summary(scene)
        exps = exp_map.get(name) or []
        rows.append({
            "name": name,
            "age_months": scene["body"]["age_months"],
            "created": scene.get("created") or "（不明）",
            "settings": _settings_jp(summary),
            "note": scene.get("note") or "",
            "experiments": exps,
        })
    return rows


def render_markdown(rows):
    lines = []
    lines.append("# シーン一覧カタログ（run/scenes 配下）")
    lines.append("")
    lines.append("このファイルは自動生成物です。手で編集しないでください")
    lines.append("（`run/scene_tools/catalog.py` が生成）。")
    lines.append("")
    lines.append(f"対象：`run/scenes/*.json` 全{len(rows)}ファイル")
    lines.append("")
    lines.append("## 表の見方（用語のかみ砕き）")
    lines.append("")
    lines.append("- 月齢：シミュレーション上の赤ちゃんの月齢（0ヶ月＝新生児、4ヶ月）")
    lines.append("- 主な設定：`e_scene.constraint_summary()` が計算した「固定/自由」")
    lines.append("  「おもちゃの有無」「flexion（生理的屈曲）の有無」をそのまま並べたもの")
    lines.append("- 注記：シーンJSONの `note` フィールドをそのまま転記したもの"
                  "（自動生成のため要約はしていない）")
    lines.append("- 使われている実験：`E/experiments/*.json` の `\"scene\"` キーが"
                  "このシーン名と一致するファイルの一覧")
    lines.append("")
    lines.append("## シーン一覧表")
    lines.append("")
    lines.append("| ファイル名 | 月齢 | 作成日 | 主な設定 | 使われている実験 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            lines.append(f"| `{r['name']}.json` | エラー | - | "
                          f"読み込みに失敗: {r['error']} | - |")
            continue
        exps = "、".join(r["experiments"]) if r["experiments"] else "見つからなかった"
        lines.append(
            f"| `{r['name']}.json` | {r['age_months']:g}ヶ月 | {r['created']} | "
            f"{r['settings']} | {exps} |")
    lines.append("")
    lines.append("## 各シーンの注記（note フィールドそのまま）")
    lines.append("")
    for r in rows:
        if r.get("error"):
            continue
        note = r["note"] or "（自動生成のため要約なし。必要ならJSONのnoteフィールドを参照）"
        lines.append(f"**{r['name']}**：{note}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_catalog():
    """`E/docs/シーン一覧.md` を実際に作り直す。呼び出し側から見た唯一の入口。"""
    rows = build_rows()
    text = render_markdown(rows)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        fp.write(text)
    print(f"[catalog] シーン一覧を書き直しました → "
          f"{os.path.relpath(OUT_PATH, _ROOT)}（{len(rows)}件）")
    return OUT_PATH


if __name__ == "__main__":
    write_catalog()
