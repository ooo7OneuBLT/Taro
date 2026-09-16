# -*- coding: utf-8 -*-
"""名づけ正解率（2026-09-07・仕様_M5_塊レベル層.md 後半「7. 名づけ正解率」節）。

太郎の発話.csv の exact_match 率と、generated_word が直前の親の語（発話イベント.csv、
時計補正0.5秒）の名詞で始まる率を出す。段1（F2-84鎖）で F2-84 r1/r2（塊ON）と
F2-84c r1/r2（OFF）を並べて見るための道具。

「時計補正0.5秒」：太郎の発話.csv の sim_sec と 発話イベント.csv の sim_sec は
別々のtickで記録されるため、厳密一致では拾えない。太郎の発話時刻より前で、
0.5秒以内に最も近い親の発話イベントを「直前に聞いた語」とみなす。

    .venv/Scripts/python.exe F/scripts/f86_naming_accuracy.py <ログディレクトリ> [...]

各ログディレクトリに 太郎の発話.csv と 発話イベント.csv があることを前提にする
（run/plugins/common/word_production.py・word_learning.py の events_out 出力）。
出力: <ログディレクトリ>/名づけ正解率.md（無ければ標準出力のみ）
"""
import csv
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")


def _add_scripts_path():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
    sys.path.insert(0, os.path.join(root, "taro_core", "src", "senses"))


_add_scripts_path()

# 8語の名詞リスト（F/scripts/f_gen_f49.py WORDS由来）。太郎の発話.csv/発話イベント.csv の
# text/generated_wordは耳(hearing.hear)を通った時点でnormalize_kana済み（カタカナ→ひらがな・
# 長音展開）のため、比較対象も同じ正規化を通す（f50_lexicon_units.pyはexpand_long_vowelのみで
# カタカナ→ひらがなを省いていたため「バス」等カタカナを含む語が拾えていなかった疑いがある。
# ここでは正しくnormalize_kanaを使う。実装時の判断＝作業記録「仕様に無かった判断」に記載）。
# 長い語から先に判定する（一致判定を安定させるため）。
from hearing import normalize_kana  # noqa: E402

NOUNS = sorted(
    [normalize_kana(w) for w in
     ["くつ", "コップ", "おさら", "おわん", "かばん", "がおー", "バス", "ボール"]],
    key=len, reverse=True)

TIME_CORRECTION_SEC = 0.5


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with io.open(path, encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _noun_in(text):
    """textの中に含まれる既知の名詞（見つからなければNone）。"""
    if not text:
        return None
    for w in NOUNS:
        if w in text:
            return w
    return None


def _nearest_parent_noun(events, taro_sim_sec, correction=TIME_CORRECTION_SEC):
    """「直前に聞いた語」＝ev_sec <= taro_sim_sec + correction を満たす中で
    ev_secが最大（最も新しい）の親発話イベントの名詞。

    correction（既定0.5秒）は太郎の発話.csvと発話イベント.csvのtickずれの補正
    （プラグインの評価順で1tick分の差が出ることがあるため）。見つからなければNone。
    """
    best_ev, best_sec = None, None
    for ev in events:
        try:
            ev_sec = float(ev.get("sim_sec", ""))
        except (TypeError, ValueError):
            continue
        if ev_sec <= taro_sim_sec + correction:
            if best_sec is None or ev_sec > best_sec:
                best_ev, best_sec = ev, ev_sec
    if best_ev is None:
        return None
    return _noun_in(best_ev.get("text", ""))


def analyze(log_dir):
    utt_path = os.path.join(log_dir, "太郎の発話.csv")
    ev_path = os.path.join(log_dir, "発話イベント.csv")
    utterances = _read_csv(utt_path)
    events = _read_csv(ev_path)
    if not utterances:
        return {"log_dir": log_dir, "n": 0, "exact_match_rate": None,
                "noun_start_rate": None, "n_events": len(events)}
    n = len(utterances)
    n_exact = sum(1 for r in utterances if str(r.get("exact_match", "0")) == "1")
    n_noun_start = 0
    n_scored = 0
    for r in utterances:
        gw = r.get("generated_word", "")
        if not gw:
            continue
        try:
            t_sec = float(r.get("sim_sec", ""))
        except (TypeError, ValueError):
            continue
        noun = _nearest_parent_noun(events, t_sec)
        if noun is None:
            continue
        n_scored += 1
        if gw.startswith(noun):
            n_noun_start += 1
    return {
        "log_dir": log_dir, "n": n, "n_events": len(events),
        "exact_match_rate": n_exact / n if n else None,
        "n_exact": n_exact,
        "n_scored": n_scored,
        "noun_start_rate": (n_noun_start / n_scored) if n_scored else None,
        "n_noun_start": n_noun_start,
    }


def main():
    if len(sys.argv) < 2:
        print("使い方: f86_naming_accuracy.py <ログディレクトリ> [...]")
        sys.exit(1)
    results = [analyze(d) for d in sys.argv[1:]]
    lines = ["# f86 名づけ正解率\n",
             "| ログディレクトリ | 発話行数 | exact_match率 | 親イベント数 | "
             "名詞で始まる率(採点対象数) |",
             "|---|---|---|---|---|"]
    for r in results:
        em = f"{r['exact_match_rate']:.3f}" if r["exact_match_rate"] is not None else "n/a"
        nr = (f"{r['noun_start_rate']:.3f}(n={r['n_scored']})"
              if r["noun_start_rate"] is not None else "n/a")
        lines.append(f"| {r['log_dir']} | {r['n']} | {em} | {r['n_events']} | {nr} |")
        print(f"{r['log_dir']}: n={r['n']} exact_match={em} "
              f"n_events={r['n_events']} noun_start={nr}", flush=True)
    md = "\n".join(lines) + "\n"
    for r in results:
        if r["n"] > 0:
            out_path = os.path.join(r["log_dir"], "名づけ正解率.md")
            try:
                with io.open(out_path, "w", encoding="utf-8") as fp:
                    fp.write(md)
                print("  ->", out_path, flush=True)
            except OSError as e:
                print(f"  [警告] {out_path} へ書けなかった: {e}", flush=True)
    print("[OK] 完了", flush=True)


if __name__ == "__main__":
    main()
