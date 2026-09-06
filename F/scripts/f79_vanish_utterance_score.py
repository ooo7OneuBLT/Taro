# -*- coding: utf-8 -*-
"""F2-76b M4 消えた物について「○○ないね」と言う、の採点。

【仕様】F/docs/二語文/仕様_M4_消えた物について「○○ないね」と言う_2026-09-06.md
「後半：実装担当向け技術付録」「採点」節。

【読むだけ】F/logs/F2-76b_M4_テスト/ の下の
  太郎の発話.csv（run/plugins/common/word_production.py。M4で追加した
    gate/gone/attended_id列を使う。既存列順は不変なのでDictReaderで読めば安全）
  発話イベント.csv（word_learning.py。cause=vanish の行が「親が『○○ないね』と
    言った瞬間」の直接の証拠。今回は参考表示のみに使う）
  注意.csv（object_files.py。attended_id ごとに「親が最初に命名した語」を
    target_word列に持つ。gone=1の行の「消えた物の名前」をここから引く）

【判定】語の一致は太郎のチャンク粒度（拗音・撥音などで文字数がずれる）を
  考慮し、仕様書の指示どおり「親の語の先頭2文字を含む」で判定する。

出力（このスクリプト自身の新規出力。触ってよいファイル）：
  F/logs/F2-76b_M4_テスト/採点.csv
  F/logs/F2-76b_M4_テスト/結果.json
  F/logs/F2-76b_M4_テスト/図_消えた窓の発話.png
"""
import csv
import json
import os
import sys

# 【2026-09-06】LOG_DIR を引数で差し替え可能にする（既定は不変）。
#   仕様：F/docs/二語文/仕様_ひらがな化してM4をやり直す_2026-09-06.md「4. f79 の引数化」。
LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("F", "logs", "F2-76b_M4_テスト")
UTTER_CSV = os.path.join(LOG_DIR, "太郎の発話.csv")
EVENTS_CSV = os.path.join(LOG_DIR, "発話イベント.csv")
ATTEND_CSV = os.path.join(LOG_DIR, "注意.csv")
OUT_CSV = os.path.join(LOG_DIR, "採点.csv")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")
OUT_PNG = os.path.join(LOG_DIR, "図_消えた窓の発話.png")

# 【参考比較】F2-74（M2・GONE配線の前）の同じ形式の発話ログ。仕様書「合否」節
#   「見えている窓で従来どおり『だね／だよ』が言えている率」「『くつ』が出た率
#   （F2-74は4/4）」の比較対象。読むだけ（変更しない）。
F274_DIR = os.path.join("F", "logs", "F2-74_M2_消失発話")
F274_UTTER_CSV = os.path.join(F274_DIR, "太郎の発話.csv")


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def nearest_row(rows, t, key_time="sim_time"):
    """|時刻差|最小の行を返す（無ければNone）。f74_vanish_utterance_check.pyの
    nearest_n_dets と同じ「前後どちらの近い方を選ぶ」考え方。"""
    best, best_diff = None, None
    for r in rows:
        rt = float(r[key_time])
        diff = abs(rt - t)
        if best_diff is None or diff < best_diff:
            best_diff, best = diff, r
    return best


def contains_dane_dayo(word):
    return ("だね" in word) or ("だよ" in word)


def main():
    utter_rows = load_rows(UTTER_CSV)
    attend_rows = load_rows(ATTEND_CSV)
    events_rows = load_rows(EVENTS_CSV)

    # ---- 消えた窓（gone=1）の発話 -----------------------------------------
    gone_rows = [r for r in utter_rows if r.get("gone") == "1"]
    table = []
    for r in gone_rows:
        t = float(r["sim_sec"])
        att_id = r.get("attended_id", "")
        # 同じattended_idの注意.csv行のうち時刻最近傍から「親が最初に命名した語」を引く。
        same_id_rows = [a for a in attend_rows if a.get("attended_id") == att_id] \
            if att_id else []
        att_row = nearest_row(same_id_rows, t) if same_id_rows else None
        target_full = att_row["target_word"] if att_row is not None else ""
        prefix = target_full[:2] if target_full else ""
        gen = r.get("generated_word", "")
        has_nai = "ない" in gen
        has_prefix = bool(prefix) and (prefix in gen)
        if has_prefix and has_nai:
            judge = "○正解（語＋ない）"
        elif has_nai and not has_prefix:
            judge = "×染み出し（ないはあるが語が違う）"
        elif not has_nai:
            judge = "×ないを含まない"
        else:
            judge = "?判定不能（親の語が不明）"
        table.append({
            "step": r["step"], "sim_sec": r["sim_sec"],
            "attended_id": att_id, "親の語（先頭2文字）": prefix,
            "親の語_全体（注意.csv target_word）": target_full,
            "太郎の生成文字列": gen, "判定": judge,
            "くつが混入": int("くつ" in gen),
        })

    n_gone = len(gone_rows)
    n_correct = sum(1 for row in table if row["判定"].startswith("○"))
    n_leak = sum(1 for row in table if "染み出し" in row["判定"])
    n_no_nai = sum(1 for row in table if row["判定"] == "×ないを含まない")
    n_kutsu = sum(row["くつが混入"] for row in table)

    # ---- 見えている窓（gate=ok・gone=0）の発話：「だね／だよ」率 -----------
    visible_rows = [r for r in utter_rows
                    if r.get("gate", "ok") == "ok" and r.get("gone") == "0"]
    n_visible = len(visible_rows)
    n_visible_dane = sum(1 for r in visible_rows
                         if contains_dane_dayo(r.get("generated_word", "")))

    # ---- gate=no_object の件数と、その間の発話が0件であること -------------
    no_object_rows = [r for r in utter_rows if r.get("gate") == "no_object"]
    n_no_object = len(no_object_rows)
    n_no_object_spoke = sum(1 for r in no_object_rows if r.get("generated_word"))

    # ---- F2-74（GONE配線前）の参考値 ---------------------------------------
    f274_rows = load_rows(F274_UTTER_CSV)
    f274_n = len(f274_rows)
    f274_dane = sum(1 for r in f274_rows
                    if contains_dane_dayo(r.get("generated_word", "")))
    f274_kutsu = sum(1 for r in f274_rows if "くつ" in r.get("generated_word", ""))

    # ---- 親のvanish発話（参考表示） -----------------------------------------
    vanish_events = [r for r in events_rows if r.get("cause") == "vanish"]

    result = {
        "消えた窓の発話数": n_gone,
        "率_語＋ないで言えた": round(n_correct / n_gone, 4) if n_gone else None,
        "率_染み出し（ないはあるが語違い）": round(n_leak / n_gone, 4) if n_gone else None,
        "率_ないを含まない": round(n_no_nai / n_gone, 4) if n_gone else None,
        "見えている窓の発話数": n_visible,
        "見えている窓_だね/だよ率": (
            round(n_visible_dane / n_visible, 4) if n_visible else None),
        "gate_no_object件数": n_no_object,
        "gate_no_object中の発話数（0であるべき）": n_no_object_spoke,
        "消えた窓でくつが出た率": round(n_kutsu / n_gone, 4) if n_gone else None,
        "消えた窓でくつが出た件数_分母": [n_kutsu, n_gone],
        "参考_F2-74（GONE配線前）": {
            "発話数": f274_n,
            "だね/だよ率": round(f274_dane / f274_n, 4) if f274_n else None,
            "くつが出た率": round(f274_kutsu / f274_n, 4) if f274_n else None,
            "くつが出た件数_分母": [f274_kutsu, f274_n],
        },
        "親のvanish発話件数（参考）": len(vanish_events),
        "表": table,
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["step", "sim_sec", "attended_id", "親の語（先頭2文字）",
                   "親の語_全体（注意.csv target_word）", "太郎の生成文字列",
                   "判定", "くつが混入"])
        for row in table:
            w.writerow([row["step"], row["sim_sec"], row["attended_id"],
                       row["親の語（先頭2文字）"], row["親の語_全体（注意.csv target_word）"],
                       row["太郎の生成文字列"], row["判定"], row["くつが混入"]])

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    # ---- 図：窓ごとの生成文字列を時系列に -----------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rcParams
        for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
            if any(c in f.name for f in font_manager.fontManager.ttflist):
                rcParams["font.family"] = c
                break
        fig, ax = plt.subplots(figsize=(11, 4))
        all_sec = [float(r["sim_sec"]) for r in utter_rows] or [0.0]
        ax.set_xlim(min(all_sec) - 1, max(all_sec) + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["見えている(gone=0)", "消えた(gone=1)"])
        ax.set_xlabel("sim_sec")
        ax.set_title("F2-76b 消えた窓の発話（○=語＋ない ×=違う ・くつ混入は赤字）")
        for r in utter_rows:
            if r.get("gate") == "no_object":
                continue
            t = float(r["sim_sec"])
            gone = r.get("gone") == "1"
            gen = r.get("generated_word", "")
            y = 1 if gone else 0
            mark = "o" if not gone else (
                "o" if any(row["sim_sec"] == r["sim_sec"] and
                           row["判定"].startswith("○") for row in table) else "x")
            color = "red" if "くつ" in gen else ("black" if not gone else
                    ("green" if mark == "o" else "orange"))
            ax.plot([t], [y], marker="s", color=color)
            ax.annotate(gen, (t, y), textcoords="offset points",
                        xytext=(0, 8 if gone else -14), fontsize=8, color=color,
                        rotation=45, ha="left")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
    except Exception as e:
        print("[図の作成に失敗・採点結果は出力済み]", repr(e))

    print(json.dumps({k: v for k, v in result.items() if k != "表"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
