# -*- coding: utf-8 -*-
"""走行を1つ渡すと「親が差し出している物の、正しい名前を言えたか」を数える。

【なぜ要るか・2026-09-16】目標Fで一番知りたいのはこれなのに、走行中に出ていた
`完全一致率` は **太郎が言った語 と 太郎が言おうとした語** を比べていた（どちらも
太郎の側＝循環）。実測でこれは常に 1.000 になり、何も分からない。
正解は走る前から場面に書かれている（`world.parent_labeling.utterances`）ので、
**太郎に一切依存しない判定**がここで初めてできる。

【この道具の立ち位置・ユーザー指示 2026-09-16】
    「出せる値は plugin で出力して、その出力をもとに測定する」
  プラグイン … 観測できる値を列として出すだけ（数えない・率を出さない）
  この道具   … 出てきた列を**あとから**読んで数える
  走行中には動かない。何度でも数え直せる（列さえ正しければ走行のやり直しは不要）。

【読むもの】走行フォルダの中の2つだけ。太郎にも環境にも触らない。
    太郎の発話.csv   … word_production が出した列（step / 発話した /
                       generated_word / 親の的 / 視線の先）
    run.meta.json    … その走行で実際に使われた場面。正解表はこの中の
                       world.parent_labeling.utterances

【使い方】
    python F/scripts/正しい名前を言えたか.py F/logs/<走行フォルダ>
    python F/scripts/正しい名前を言えたか.py F/logs/<走行フォルダ> --csv 出力先.csv

【名前に実験番号を入れていない理由】`f16_` `f95_` のように番号を付けた道具は
その実験だけのものになり、次の実験でまた1本増える（2026-09-16 に 330本を書庫へ
移した原因がこれ）。実験ごとに作らず、この1本に走行フォルダを渡す。
"""
import argparse
import csv
import io
import json
import os
import sys

要る列 = ["発話した", "generated_word", "親の的"]


def 正解表を読む(meta_path):
    """run.meta.json から {スロット: [正解の語, ...]} を作る。

    `utterances` の値は場面によって2通りある：
      文字列                     … "わんわん"
      [[語, 重み], ...] の並び   … [["おわんだね", 0.41], ["おわん", 0.18]]
    どちらも「その物を指す語の集合」なので、重みは落として語だけ集める。
    """
    with io.open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)
    u = ((meta.get("world") or {}).get("parent_labeling") or {}).get("utterances") or {}
    表 = {}
    for slot, v in u.items():
        if not v:
            continue
        if isinstance(v, str):
            表[slot] = [v]
        elif isinstance(v, (list, tuple)):
            語 = []
            for x in v:
                if isinstance(x, str):
                    語.append(x)
                elif isinstance(x, (list, tuple)) and x and isinstance(x[0], str):
                    語.append(x[0])
            if 語:
                表[slot] = 語
    return 表, meta


def 発話を読む(csv_path):
    with io.open(csv_path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def 走行フォルダを見る(d):
    """フォルダの中から発話のCSVと run.meta.json を探す。"""
    meta = os.path.join(d, "run.meta.json")
    if not os.path.isfile(meta):
        raise SystemExit("run.meta.json がありません: %s\n"
                         "  走行フォルダ（F/logs/<実験名>/）を渡してください" % d)
    cand = os.path.join(d, "太郎の発話.csv")
    if os.path.isfile(cand):
        return cand, meta
    # 名前が違うことがある（events_out は実験ファイルで指定できる）ので、
    # 要る列を持っているCSVを探す。当てずっぽうで拾わないための条件。
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        p = os.path.join(d, f)
        try:
            with io.open(p, encoding="utf-8") as fp:
                cols = next(csv.reader(fp))
        except Exception:       # noqa: BLE001
            continue
        if all(c in cols for c in 要る列):
            return p, meta
    raise SystemExit("発話のCSVが見つかりません（要る列: %s）: %s" % ("・".join(要る列), d))


def 数える(rows, 表):
    """行と正解表から、判定済みの行と集計を返す。"""
    判定 = []
    for r in rows:
        言った = (r.get("generated_word") or "").strip()
        的 = (r.get("親の的") or "").strip()
        声 = str(r.get("発話した", "")).strip() in ("1", "True", "true")
        if not 声:
            continue
        正解語 = 表.get(的) or []
        判定.append({
            "step": r.get("step", ""),
            "親の的": 的,
            "言った": 言った,
            "正解語": 正解語,
            "視線の先": (r.get("視線の先") or "").strip(),
            # 親が何も差し出していない歩は正誤を付けられない（分母から外す）
            "判定できる": bool(的) and bool(正解語),
            "言えた": bool(的) and 言った in 正解語,
        })
    return 判定


def 出す(判定, rows, 表, 走行名):
    分母 = [x for x in 判定 if x["判定できる"]]
    正 = [x for x in 分母 if x["言えた"]]
    誤 = [x for x in 分母 if not x["言えた"]]
    見ていた = [x for x in 分母 if x["視線の先"] and x["視線の先"] == x["親の的"]]
    print("走行 %s" % 走行名)
    print("  記録した歩数            %d" % len(rows))
    print("  そのうち声を出した歩    %d" % len(判定))
    print("  判定できる歩            %d  （親が何かを差し出していた歩）" % len(分母))
    if not 分母:
        print("\n  判定できる歩が0でした。親が物を差し出している間に一度も喋っていません。")
        return 1
    print()
    print("  **正しい名前を言えた    %d / %d = %.1f%%**"
          % (len(正), len(分母), 100.0 * len(正) / len(分母)))
    print("  （参考）そのとき的を見ていた %d / %d = %.1f%%"
          % (len(見ていた), len(分母), 100.0 * len(見ていた) / len(分母)))
    if 誤:
        print("\n  外した %d 回:" % len(誤))
        print("    %-7s %-7s %-14s %-7s %s" % ("step", "親の的", "言った", "視線の先", "正解だった語"))
        for x in 誤:
            print("    %-7s %-7s %-14s %-7s %s"
                  % (x["step"], x["親の的"], x["言った"] or "（空）",
                     x["視線の先"] or "―", "・".join(x["正解語"]) or "（正解表に無い）"))
    無し = [x for x in 判定 if not x["判定できる"]]
    if 無し:
        print("\n  判定できなかった %d 回（親が何も差し出していない・正解表に無いスロット）:" % len(無し))
        for x in 無し[:10]:
            print("    step %-7s 親の的 %-7s 言った %s"
                  % (x["step"], x["親の的"] or "―", x["言った"] or "（空）"))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="走行フォルダを1つ渡すと「親の的の正しい名前を言えたか」を数える")
    ap.add_argument("走行フォルダ", help="F/logs/<実験名>/")
    ap.add_argument("--csv", default=None, help="判定した行をCSVに書き出す先")
    a = ap.parse_args()
    d = a.走行フォルダ
    if not os.path.isdir(d):
        raise SystemExit("フォルダがありません: %s" % d)
    csv_path, meta_path = 走行フォルダを見る(d)
    rows = 発話を読む(csv_path)
    if not rows:
        raise SystemExit("発話のCSVが空です: %s" % csv_path)
    足りない = [c for c in 要る列 if c not in rows[0]]
    if 足りない:
        raise SystemExit(
            "このCSVには %s の列がありません: %s\n"
            "  2026-09-16 より前の走行には `発話した` 列がありません（当時は黙った歩も\n"
            "  1発話として数えていました）。壁ありの土台で走らせ直してください。"
            % ("・".join(足りない), csv_path))
    表, _meta = 正解表を読む(meta_path)
    if not 表:
        raise SystemExit("run.meta.json に正解表（world.parent_labeling.utterances）が"
                         "ありません: %s" % meta_path)
    判定 = 数える(rows, 表)
    code = 出す(判定, rows, 表, os.path.basename(os.path.normpath(d)))
    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or ".", exist_ok=True)
        with io.open(a.csv, "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["step", "親の的", "言った", "視線の先", "判定できる", "言えた", "正解語"])
            for x in 判定:
                w.writerow([x["step"], x["親の的"], x["言った"], x["視線の先"],
                            int(x["判定できる"]), int(x["言えた"]), "・".join(x["正解語"])])
        print("\n  判定した行を書き出しました: %s" % a.csv)
    return code


if __name__ == "__main__":
    sys.exit(main())
