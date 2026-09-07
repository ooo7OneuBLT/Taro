# -*- coding: utf-8 -*-
"""短期B 「ないね」と一緒に聞いたことのない物が消えたとき「○○ないね」と言えるか、の採点。

【仕様】F/docs/二語文/仕様_短期B_教えていない物が消えたら「○○ないね」と言えるか_2026-09-06.md
「後半：実装担当向け技術付録」「2. 採点」節。

引数：ログディレクトリ（例 F/logs/F2-82b_短期B_テスト）。

【読むだけ】ログディレクトリ配下の
  注意.csv（object_files.py）。attended_idごとの vanished 列の「偽→真」を
    消失イベント（時刻T）とする。
  発話イベント.csv（word_learning.py）。cause列が空/"repeat"/"voice"の行が
    親の通常の命名。targetが消えた物のtoy識別子。cause="vanish"/"vanish_timeout"は
    親が「○○ないね」と言った行（黙る的ではこの行は存在しない＝仕様書の指示）。
  太郎の発話.csv（word_production.py）。gone列が"1"の行を「消えた窓の発話」とする。
  run.meta.json（run/main.py が実験ファイルの内容を書き出したもの）。scene名を
    ここから読み、run/scenes/<scene>.json の world.parent_labeling.utterances /
    vanish_silent_targets を「消えた物の単独語」「黙る側(3語)/教えた側(5語)」の
    判定に使う。

【時計補正・落とし穴 2026-09-06】発話イベント.csv・太郎の発話.csv の sim_sec は、
  注意.csv の sim_time より一定量小さい（実測：F2-81bログで0.5秒）。生の値のまま
  比較すると窓の境界がずれる。補正量は「注意.csv の sim_time − step*0.1」の中央値を
  実測して使う（発話イベント.csv・太郎の発話.csv 側は sim_sec − step*0.1 の中央値が
  0であることを実測済み＝これらの2ファイルは同じ時計）。
  corrected_sim_sec = sim_sec + correction とすれば 注意.csv の sim_time と同じ時計になる。

【窓の終わり】仕様書は「次の親の命名（補正後）まで」とだけ書く。文字どおり
  「時刻がTより後の最初の発話イベント行」を境にすると、消えた物自身についての
  「○○ないね」（cause=vanish/vanish_timeout、targetが消えた物と同じ）が真っ先に
  ヒットしてしまい、窓が本来より早く閉じる（太郎の反応を待つ前に閉じる）。
  ここでは「消えた物と異なるtargetを持つ、時刻Tより後の最初の発話イベント行」を
  次の命名とみなす（＝新しい物の提示が始まった時点）。黙る的では「○○ないね」の
  行自体が存在しないためこの区別は問題にならないが、教えた側(5語)との扱いを
  揃えるために両方へ同じ規則を適用した。[実装判断・仕様書に明記が無いため
  ここに理由を残す]

出力（このスクリプト自身の新規出力）：
  <ログディレクトリ>/採点.csv
  <ログディレクトリ>/結果.json
  <ログディレクトリ>/図_短期B.png
"""
import ast
import csv
import json
import os
import statistics
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    "F", "logs", "F2-82b_短期B_テスト")
ATTEND_CSV = os.path.join(LOG_DIR, "注意.csv")
EVENTS_CSV = os.path.join(LOG_DIR, "発話イベント.csv")
UTTER_CSV = os.path.join(LOG_DIR, "太郎の発話.csv")
META_JSON = os.path.join(LOG_DIR, "run.meta.json")
OUT_CSV = os.path.join(LOG_DIR, "採点.csv")
OUT_JSON = os.path.join(LOG_DIR, "結果.json")
OUT_PNG = os.path.join(LOG_DIR, "図_短期B.png")

# run/scenes/ の場所（このファイルは F/scripts/ 直下にある前提。落とし穴108と同じ
# 「テンプレの値は実行して検証」の流儀で、ROOT算出も本体側の実在パスで確認済み）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
SCENES_DIR = os.path.join(ROOT, "run", "scenes")

# 偽陽性判定の猶予（仕様書「2. 採点」節）。
FALSE_POSITIVE_WINDOW_SEC = 2.0


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def single_word(utterances, target):
    """parent_labeling.ParentLabeling._single_word と同じ規約。

    utterances[target] が [[text, weight], ...] のリストなら最短のtext、
    文字列ならそのまま。無ければ空文字列。
    """
    u = utterances.get(target)
    if u is None:
        return ""
    if isinstance(u, list):
        if not u:
            return ""
        return min((str(item[0]) for item in u), key=len)
    return str(u)


def load_scene_info(log_dir):
    """run.meta.json の scene名から run/scenes/<scene>.json を読み、
    utterances / vanish_silent_targets を返す。見つからなければ空扱い。
    """
    if not os.path.exists(META_JSON):
        return {}, set()
    with open(META_JSON, encoding="utf-8") as fp:
        meta = json.load(fp)
    scene_name = meta.get("scene")
    if not scene_name:
        return {}, set()
    scene_path = os.path.join(SCENES_DIR, scene_name + ".json")
    if not os.path.exists(scene_path):
        return {}, set()
    with open(scene_path, encoding="utf-8") as fp:
        scene = json.load(fp)
    pl = scene.get("world", {}).get("parent_labeling", {}) or {}
    utterances = pl.get("utterances", {}) or {}
    silent = set(pl.get("vanish_silent_targets", []) or [])
    return utterances, silent


def compute_correction(attend_rows):
    """注意.csv の sim_time − step*0.1 の中央値（発話イベント.csv・太郎の発話.csvの
    sim_secへ足すと注意.csvのsim_timeと同じ時計になる補正量）。"""
    diffs = []
    for r in attend_rows:
        try:
            diffs.append(float(r["sim_time"]) - int(r["step"]) * 0.1)
        except (KeyError, ValueError):
            continue
    if not diffs:
        return 0.0
    return statistics.median(diffs)


def intended_word(row):
    """太郎が発音する前に選んだ語の文字列（＝「意図した語」）を返す。

    【なぜ、2026-09-07】word_production.py は choice_gru / choice_hippo に
    候補ごとの ['語', 確信度] を残す。chosen列（"gru"/"hippo"）でどちらを
    選んだかが分かる。generated_word はそのあとの発音（モーラ連鎖）の結果で、
    「っ」「ぷ」等が崩れて字面が変わることがある（例：F2-85bログ、こっぷが
    消えた窓でgenerated_word="ここあないね"だがchoice_gruは
    ['こっぷないね', 0.772]＝選んだ語自体は正しい）。採点を発音の崩れに
    引きずられさせないため、選んだ時点の文字列を「意図した語」として使う。
    """
    chosen = row.get("chosen")
    col = None
    if chosen == "gru":
        col = "choice_gru"
    elif chosen == "hippo":
        col = "choice_hippo"
    raw = row.get(col) if col else None
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return row.get("generated_word", "")


def find_vanish_events(attend_rows):
    """attended_idごとに vanished列の偽→真を検出し、[{"T":秒, "attended_id":id}]を返す。

    最初に見えた行がすでにvanished=Trueの場合は「遷移」ではない（既にログの外で
    消えていた可能性がある）ため事象に数えない。
    """
    rows_sorted = sorted(
        attend_rows, key=lambda r: (int(r["step"]), r.get("attended_id", "")))
    prev = {}
    events = []
    for r in rows_sorted:
        aid = r.get("attended_id", "")
        v = (r.get("vanished") == "True")
        if aid not in prev:
            prev[aid] = v
            continue
        if v and not prev[aid]:
            events.append({"T": float(r["sim_time"]), "attended_id": aid})
        prev[aid] = v
    return events


def main():
    attend_rows = load_rows(ATTEND_CSV)
    events_rows = load_rows(EVENTS_CSV)
    utter_rows = load_rows(UTTER_CSV)
    utterances, silent_targets = load_scene_info(LOG_DIR)

    correction = compute_correction(attend_rows)

    # 発話イベント.csv・太郎の発話.csv に補正後の時刻を付けて時刻順に並べる。
    ev_sorted = sorted(events_rows, key=lambda r: int(r["step"]))
    for r in ev_sorted:
        r["_t"] = float(r["sim_sec"]) + correction
    ut_sorted = sorted(utter_rows, key=lambda r: int(r["step"]))
    for r in ut_sorted:
        r["_t"] = float(r["sim_sec"]) + correction

    vanish_events = find_vanish_events(attend_rows)

    table = []
    false_positive_table = []
    for ev in vanish_events:
        T = ev["T"]
        # 消えた物＝Tより前の最後の発話イベント行のtarget。
        prior = [r for r in ev_sorted if r["_t"] < T]
        target = prior[-1].get("target") if prior else None
        if not target:
            table.append({
                "T": round(T, 3), "消えた物": "", "側": "不明",
                "太郎の発話": "", "判定": "?対象不明（発話イベント無し）",
                "厳密": "?対象不明（発話イベント無し）",
                "意図した語": "", "意図判定": "?対象不明（発話イベント無し）",
            })
            continue

        # 偽陽性：Tの後2秒以内に、同じtargetを「ないね」以外の原因で
        #   親がまた命名していれば、実際には消えていなかった扱い。
        soon = [r for r in ev_sorted
                if T < r["_t"] <= T + FALSE_POSITIVE_WINDOW_SEC
                and r.get("target") == target
                and r.get("cause") not in ("vanish", "vanish_timeout")]
        if soon:
            false_positive_table.append({
                "T": round(T, 3), "消えた物": target,
                "再命名時刻": round(soon[0]["_t"], 3),
            })
            continue

        # 窓の終わり＝Tより後、targetと異なる的を命名した最初の発話イベント行。
        nxt = [r for r in ev_sorted if r["_t"] > T and r.get("target") != target]
        window_end = nxt[0]["_t"] if nxt else float("inf")

        side = "黙る側(3語)" if target in silent_targets else "教えた側(5語)"
        word = single_word(utterances, target)

        in_window = [r for r in ut_sorted
                     if r.get("gone") == "1" and T <= r["_t"] < window_end]
        gens = [r.get("generated_word", "") for r in in_window]
        intended_gens = [intended_word(r) for r in in_window]

        # 厳密判定（従来）：語で始まり、かつ「ない」を含む。
        has_correct_strict = any(g.startswith(word) and ("ない" in g) and word
                                  for g in gens)
        has_nai_only_strict = any(
            ("ない" in g) and not (g.startswith(word) and word) for g in gens)
        if has_correct_strict:
            judge_strict = "○正解（語＋ない）"
        elif has_nai_only_strict:
            judge_strict = "×語違い（ないはあるが語が違う）"
        else:
            judge_strict = "×ない無し"

        # 断片許容判定（2026-09-07 仕様_M5_段2）：「generatedに『ない』を含み、
        #   かつ（語で始まる or 語の連続2モーラ以上の部分文字列が『ない』より
        #   前に含まれる）」を正解とする。ユーザー決定「語の一部が崩れるのは
        #   直さない」（2026-09-07）を採点側に反映した。
        def _has_fragment(g, w):
            if not w or "ない" not in g:
                return False
            nai_pos = g.find("ない")
            head = g[:nai_pos]
            if head.startswith(w):
                return True
            # wの中の連続2モーラ以上の部分文字列がheadに含まれるか。
            for length in range(len(w), 1, -1):
                for start in range(0, len(w) - length + 1):
                    sub = w[start:start + length]
                    if len(sub) >= 2 and sub in head:
                        return True
            return False

        has_correct_fragment = any(_has_fragment(g, word) for g in gens)
        has_nai_only_fragment = any(
            ("ない" in g) and not _has_fragment(g, word) for g in gens)
        if has_correct_fragment:
            judge_fragment = "○正解（語＋ない・断片可）"
        elif has_nai_only_fragment:
            judge_fragment = "×語違い（ないはあるが語が違う）"
        else:
            judge_fragment = "×ない無し"

        # 既定の「判定」列は断片許容（仕様書の採点方針）。厳密判定は別列で残す。
        judge = judge_fragment

        # 意図判定：発音される前に太郎が選んだ語（intended_gens）に断片許容と
        # 同じ規則を適用する。発音の崩れ（モーラ連鎖の失敗）に引きずられずに
        # 「何を言おうとしたか」を採点する。
        has_correct_intended = any(_has_fragment(g, word) for g in intended_gens)
        has_nai_only_intended = any(
            ("ない" in g) and not _has_fragment(g, word) for g in intended_gens)
        if has_correct_intended:
            judge_intended = "○正解（語＋ない・意図）"
        elif has_nai_only_intended:
            judge_intended = "×語違い（ないはあるが語が違う）"
        else:
            judge_intended = "×ない無し"

        table.append({
            "T": round(T, 3), "消えた物": target, "側": side,
            "太郎の発話": "; ".join(g for g in gens if g), "判定": judge,
            "厳密": judge_strict,
            "意図した語": "; ".join(g for g in intended_gens if g),
            "意図判定": judge_intended,
        })

    # 側ごとの集計。
    def rate(rows, pred):
        n = len(rows)
        if n == 0:
            return None, 0
        k = sum(1 for r in rows if pred(r))
        return round(k / n, 4), n

    def side_summary(side_name):
        rows = [r for r in table if r["側"] == side_name]
        n = len(rows)
        n_correct = sum(1 for r in rows if r["判定"].startswith("○"))
        n_wrong_word = sum(1 for r in rows if "語違い" in r["判定"])
        n_no_nai = sum(1 for r in rows if r["判定"] == "×ない無し")
        # 厳密判定（従来の判定基準）側の集計。列「厳密」を使う。
        n_correct_strict = sum(1 for r in rows if r["厳密"].startswith("○"))
        n_wrong_word_strict = sum(1 for r in rows if "語違い" in r["厳密"])
        n_no_nai_strict = sum(1 for r in rows if r["厳密"] == "×ない無し")
        # 意図判定（発音前に選んだ語での判定）側の集計。列「意図判定」を使う。
        n_correct_intended = sum(1 for r in rows if r["意図判定"].startswith("○"))
        n_wrong_word_intended = sum(1 for r in rows if "語違い" in r["意図判定"])
        n_no_nai_intended = sum(1 for r in rows if r["意図判定"] == "×ない無し")
        return {
            "件数": n,
            "語＋ない率": round(n_correct / n, 4) if n else None,
            "語違い率": round(n_wrong_word / n, 4) if n else None,
            "ない無し率": round(n_no_nai / n, 4) if n else None,
            "厳密_語＋ない率": round(n_correct_strict / n, 4) if n else None,
            "厳密_語違い率": round(n_wrong_word_strict / n, 4) if n else None,
            "厳密_ない無し率": round(n_no_nai_strict / n, 4) if n else None,
            "意図_語＋ない率": round(n_correct_intended / n, 4) if n else None,
            "意図_語違い率": round(n_wrong_word_intended / n, 4) if n else None,
            "意図_ない無し率": round(n_no_nai_intended / n, 4) if n else None,
        }

    # 見えている窓（gone=0）に「ない」が漏れていないか。
    visible_rows = [r for r in ut_sorted if r.get("gone") == "0"]
    n_visible_leak = sum(1 for r in visible_rows
                         if "ない" in r.get("generated_word", ""))

    result = {
        "教えた側(5語)": side_summary("教えた側(5語)"),
        "黙る側(3語)": side_summary("黙る側(3語)"),
        "対象不明の事象件数": sum(1 for r in table if r["側"] == "不明"),
        "偽陽性件数": len(false_positive_table),
        "偽陽性": false_positive_table,
        "見えている窓の発話数": len(visible_rows),
        "見えている窓の「ない」漏れ件数": n_visible_leak,
        "時計補正_秒": round(correction, 4),
        "表": table,
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["T", "消えた物", "側", "太郎の発話", "判定", "厳密",
                    "意図した語", "意図判定"])
        for row in table:
            w.writerow([row["T"], row["消えた物"], row["側"],
                       row["太郎の発話"], row["判定"], row["厳密"],
                       row["意図した語"], row["意図判定"]])

    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    # ---- 図：消失イベントを側で色分けし、太郎の発話を文字で -------------------
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
        all_t = [r["_t"] for r in ut_sorted] or [0.0]
        ax.set_xlim(min(all_t) - 1, max(all_t) + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["教えた側(5語)", "黙る側(3語)"])
        ax.set_xlabel("sim_time（注意.csv基準に補正済み）")
        ax.set_title("短期B 消失窓での太郎の発話（○=語＋ない ×=違う）")
        for row in table:
            if row["側"] not in ("教えた側(5語)", "黙る側(3語)"):
                continue
            y = 1 if row["側"] == "黙る側(3語)" else 0
            color = "green" if row["判定"].startswith("○") else (
                "orange" if "語違い" in row["判定"] else "gray")
            ax.plot([row["T"]], [y], marker="s", color=color)
            label = row["太郎の発話"] or "(発話なし)"
            ax.annotate(f"{row['消えた物']}:{label}", (row["T"], y),
                        textcoords="offset points",
                        xytext=(0, 8 if y == 1 else -18), fontsize=8,
                        color=color, rotation=45, ha="left")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
    except Exception as e:
        print("[図の作成に失敗・採点結果は出力済み]", repr(e))

    print(json.dumps({k: v for k, v in result.items() if k != "表"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
