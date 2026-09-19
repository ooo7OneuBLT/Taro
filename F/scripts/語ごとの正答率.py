# -*- coding: utf-8 -*-
"""走行フォルダを渡すと「どの語をどれだけ言えたか」を語ごとに数える。

【なぜ要るか・2026-09-17】`正しい名前を言えたか.py` は全体の率と外した一覧しか
出さない。「8語のうちどれが得意でどれが苦手か」が分からないので、語ごとに割る。
判定の中身（正解表の読み方・言えたの定義）は `正しい名前を言えたか.py` と同じ。

【使い方】
    python F/scripts/語ごとの正答率.py F/logs/<走行フォルダ>
    python F/scripts/語ごとの正答率.py F/logs/F2-133_壁あり語学習_*  ← 複数まとめて
    python F/scripts/語ごとの正答率.py F/logs/<走行フォルダ> --csv 出力先.csv

複数フォルダを渡すと、全部を合算した表と、フォルダごとの全体率の両方を出す。
"""
import argparse
import csv
import io
import json
import os
import sys

要る列 = ["発話した", "generated_word", "親の的"]


def 正解表を読む(meta_path):
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
    return 表


def 走行フォルダを見る(d):
    meta = os.path.join(d, "run.meta.json")
    if not os.path.isfile(meta):
        return None, None
    cand = os.path.join(d, "太郎の発話.csv")
    if os.path.isfile(cand):
        return cand, meta
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
    return None, None


def 数える(csv_path, 表):
    """{スロット: [分母, 正解]} と、外した内訳 {スロット: {言った語: 回数}} を返す。"""
    語ごと = {}
    誤り = {}
    with io.open(csv_path, encoding="utf-8") as fp:
        for r in csv.DictReader(fp):
            if str(r.get("発話した", "")).strip() not in ("1", "True", "true"):
                continue
            的 = (r.get("親の的") or "").strip()
            言った = (r.get("generated_word") or "").strip()
            正解語 = 表.get(的) or []
            if not 的 or not 正解語:
                continue                    # 親が何も差し出していない歩は分母から外す
            語ごと.setdefault(的, [0, 0])
            語ごと[的][0] += 1
            if 言った in 正解語:
                語ごと[的][1] += 1
            else:
                誤り.setdefault(的, {})
                キー = 言った or "（空）"
                誤り[的][キー] = 誤り[的].get(キー, 0) + 1
    return 語ごと, 誤り


def 棒(率, 幅=20):
    n = int(round(率 * 幅))
    return "#" * n + "." * (幅 - n)


def main():
    ap = argparse.ArgumentParser(description="語ごとに「正しい名前を言えたか」を数える")
    ap.add_argument("走行フォルダ", nargs="+", help="F/logs/<実験名>/ （複数可）")
    ap.add_argument("--csv", default=None, help="語ごとの表をCSVに書き出す先")
    a = ap.parse_args()

    合算 = {}
    誤り合算 = {}
    走行別 = []
    語名 = {}       # スロット名 → 正解の語（表示用。書類にはこちらを出す）
    for d in a.走行フォルダ:
        if not os.path.isdir(d):
            print("  （飛ばす：フォルダがありません）%s" % d)
            continue
        csv_path, meta_path = 走行フォルダを見る(d)
        if csv_path is None:
            print("  （飛ばす：発話CSVか run.meta.json がありません）%s" % d)
            continue
        表 = 正解表を読む(meta_path)
        if not 表:
            print("  （飛ばす：正解表がありません）%s" % d)
            continue
        for _k, _v in 表.items():
            語名.setdefault(_k, _v[0])
        語ごと, 誤り = 数える(csv_path, 表)
        分母 = sum(v[0] for v in 語ごと.values())
        正 = sum(v[1] for v in 語ごと.values())
        走行別.append((os.path.basename(os.path.normpath(d)), 分母, 正))
        for k, (n, c) in 語ごと.items():
            合算.setdefault(k, [0, 0])
            合算[k][0] += n
            合算[k][1] += c
        for k, d2 in 誤り.items():
            誤り合算.setdefault(k, {})
            for w, n in d2.items():
                誤り合算[k][w] = 誤り合算[k].get(w, 0) + n

    if not 合算:
        raise SystemExit("数えられる走行がありませんでした。")

    print()
    print("語ごとの正答率（%d本の走行を合算）" % len(走行別))
    print("  %-10s %-8s %5s %5s %7s  %s" % ("語", "スロット", "回数", "正解", "正答率", ""))
    for k in sorted(合算, key=lambda x: -合算[x][1] / max(合算[x][0], 1)):
        n, c = 合算[k]
        率 = c / n if n else 0.0
        print("  %-10s %-8s %5d %5d %6.1f%%  %s"
              % (語名.get(k, "?"), k, n, c, 100 * 率, 棒(率)))
    分母 = sum(v[0] for v in 合算.values())
    正 = sum(v[1] for v in 合算.values())
    print("  %-10s %-8s %5d %5d %6.1f%%" % ("合計", "", 分母, 正, 100.0 * 正 / 分母))
    print("  当てずっぽう（%d択）      %6.1f%%" % (len(合算), 100.0 / len(合算)))

    if len(走行別) > 1:
        print()
        print("走行ごとの全体率")
        for name, n, c in 走行別:
            print("  %-48s %4d/%4d = %5.1f%%" % (name, c, n, 100.0 * c / n if n else 0))

    if 誤り合算:
        print()
        print("外したとき、何と言ったか")
        for k in sorted(誤り合算):
            内訳 = sorted(誤り合算[k].items(), key=lambda x: -x[1])
            print("  %-10s %s" % (語名.get(k, k),
                                  "　".join("%s×%d" % (w, n) for w, n in 内訳[:6])))

    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or ".", exist_ok=True)
        with io.open(a.csv, "w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["語", "回数", "正解", "正答率"])
            for k in sorted(合算):
                n, c = 合算[k]
                w.writerow([k, n, c, round(100.0 * c / n, 1) if n else 0])
        print("\n  書き出しました: %s" % a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
