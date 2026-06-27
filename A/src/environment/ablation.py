"""
アブレーション解析 — 各部品を無効化して貢献度を測定する

【人間模倣】神経科学の損傷実験（lesion study）と同じ手法。
脳の一部を損傷させて「何ができなくなるか」を観察し、
その部品の機能を明らかにする。

条件：
  full        : 全部品あり（ベースライン）
  no_brocas   : ブローカ野なし（発話計画なし、1文字ずつ生成）
  no_cerebellum: 小脳なし（運動記憶なし、毎回ゼロから探索）
  no_lc       : 青斑核なし（NE固定、探索の自動調整なし）
  no_habit    : 馴化なし（飽きない、同じ出力を繰り返し続ける）
  no_social   : 社会的報酬なし（親が笑わない＝ネグレクト環境）
  no_coupling : 連動制御なし（最初から全パラメータ独立）
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.parent_sim import run_simulation, ParentSmile, StageTracker
from environment.core import TaroEnvironment


ABLATION_CONDITIONS = {
    "full": {},
    "no_brocas": {"brocas_area": True},
    "no_cerebellum": {"cerebellum": True},
    "no_lc": {"locus_coeruleus": True},
    "no_habit": {"habituation": True},
    "no_social": {"social_reward": True},
    "no_coupling": {"coupling": True},
}

CONDITION_LABELS = {
    "full": "全部品あり（ベースライン）",
    "no_brocas": "ブローカ野なし（発話計画なし）",
    "no_cerebellum": "小脳なし（運動記憶なし）",
    "no_lc": "青斑核なし（NE固定）",
    "no_habit": "馴化なし（飽きなし）",
    "no_social": "社会的報酬なし（ネグレクト）",
    "no_coupling": "連動制御なし（最初から独立）",
}


def run_ablation_single(condition_name, ablation_flags, max_turns=2000,
                        phrases=None, test_phrases=None, verbose=True):
    """1条件のアブレーション実験を実行する。"""
    if phrases is None:
        phrases = ["まま", "ばば", "ないない", "まんま", "ねんね"]
    if test_phrases is None:
        test_phrases = ["ぱぱ", "だだ", "いやいや", "ぶぶ", "にに"]

    label = CONDITION_LABELS.get(condition_name, condition_name)
    run_name = f"ablation_{condition_name}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"条件: {condition_name} — {label}")
        print(f"無効化: {ablation_flags if ablation_flags else 'なし'}")
        print(f"{'='*60}")

    env = TaroEnvironment(run_name=run_name, ablation=ablation_flags)
    smile = ParentSmile()
    tracker = StageTracker(window=20)

    description = f"Ablation: {condition_name} ({label})"
    env.logger.save_run_info(description, env.cfg, phrases=phrases)

    prev_r_imit = 0.0
    prev_taro_output = ""
    partial_target = env.partial_streak_target
    best_partial = 0
    stage4_turn = None

    r_imit_history = []
    exact_match_history = []

    for turn_idx in range(max_turns):
        phrase = phrases[turn_idx % len(phrases)]
        r_social = smile.compute_smile(prev_r_imit, prev_taro_output) if turn_idx > 0 else 0.0

        result = env.step(phrase, r_social=r_social)
        prev_r_imit = result["r_imit"]
        prev_taro_output = result["taro"]

        tracker.update(result["taro"], result["turn"])
        tracker.check_all(result["turn"])

        r_imit_history.append(result["r_imit"])
        exact_match_history.append(1 if result["exact_match"] else 0)

        ps = result["partial_streak"]
        if ps > best_partial:
            best_partial = ps

        if verbose and (turn_idx < 10 or turn_idx % 200 == 0):
            print(
                f"  t{result['turn']:4d} | "
                f"S{env.vocal_tract.stage} | "
                f"[{result['parent']}]->[{result['taro']:<8s}] | "
                f"模倣={result['r_imit']:.2f} "
                f"R={result['R']:.2f}"
            )

        if stage4_turn is None and ps >= partial_target:
            stage4_turn = result["turn"]
            tracker.stage_achieved[4] = True
            tracker.stage_achieved_turn[4] = result["turn"]
            if verbose:
                print(f"  *** Stage 4 達成！ turn {result['turn']} ***")

    # 転移テスト
    transfer_results = []
    for tp in test_phrases:
        scores = []
        exact_count = 0
        for _ in range(5):
            r = env.step(tp, r_social=0.0)
            scores.append(r["r_imit"])
            if r["exact_match"]:
                exact_count += 1
        avg = sum(scores) / len(scores)
        transfer_results.append({
            "phrase": tp,
            "avg_r_imit": round(avg, 3),
            "exact_rate": exact_count / 5,
        })

    # 区間ごとの模倣スコア平均
    window = 100
    r_imit_by_window = []
    for i in range(0, len(r_imit_history), window):
        chunk = r_imit_history[i:i+window]
        r_imit_by_window.append(round(sum(chunk) / len(chunk), 3))

    exact_by_window = []
    for i in range(0, len(exact_match_history), window):
        chunk = exact_match_history[i:i+window]
        exact_by_window.append(round(sum(chunk) / len(chunk), 3))

    env.logger.plot_learning_curve()

    summary = {
        "condition": condition_name,
        "label": label,
        "ablation": ablation_flags,
        "max_turns": max_turns,
        "stage4_achieved": stage4_turn is not None,
        "stage4_turn": stage4_turn,
        "stages": dict(tracker.stage_achieved),
        "stage_turns": dict(tracker.stage_achieved_turn),
        "best_partial_streak": best_partial,
        "final_r_imit_avg": round(sum(r_imit_history[-100:]) / min(100, len(r_imit_history)), 3),
        "final_exact_rate": round(sum(exact_match_history[-100:]) / min(100, len(exact_match_history)), 3),
        "r_imit_by_100turns": r_imit_by_window,
        "exact_by_100turns": exact_by_window,
        "transfer_results": transfer_results,
    }

    if verbose:
        print(f"\n  結果: Stage4={'達成(t'+str(stage4_turn)+')' if stage4_turn else '未達成'}")
        print(f"  最終模倣スコア平均: {summary['final_r_imit_avg']:.3f}")
        print(f"  最終完全一致率: {summary['final_exact_rate']:.3f}")
        print(f"  転移テスト:")
        for tr in transfer_results:
            print(f"    [{tr['phrase']}] 模倣={tr['avg_r_imit']:.3f} 完全一致={tr['exact_rate']:.0%}")

    return summary


def run_all_ablations(conditions=None, max_turns=2000, phrases=None,
                      test_phrases=None, verbose=True):
    """全条件のアブレーション実験を実行し、比較表を出力する。"""
    if conditions is None:
        conditions = list(ABLATION_CONDITIONS.keys())

    results = {}
    start_time = time.time()

    for cond in conditions:
        flags = ABLATION_CONDITIONS[cond]
        results[cond] = run_ablation_single(
            cond, flags, max_turns=max_turns,
            phrases=phrases, test_phrases=test_phrases, verbose=verbose,
        )

    elapsed = time.time() - start_time

    # 比較表の出力
    print(f"\n{'='*80}")
    print(f"アブレーション解析結果（{len(conditions)}条件, {max_turns}ターン, {elapsed:.0f}秒）")
    print(f"{'='*80}")
    print(f"{'条件':<16s} | {'Stage4':>8s} | {'模倣平均':>8s} | {'完全一致':>8s} | {'転移平均':>8s}")
    print(f"{'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for cond in conditions:
        r = results[cond]
        s4 = f"t{r['stage4_turn']}" if r['stage4_achieved'] else "---"
        tr_avg = sum(t["avg_r_imit"] for t in r["transfer_results"]) / len(r["transfer_results"])
        print(f"{cond:<16s} | {s4:>8s} | {r['final_r_imit_avg']:>8.3f} | {r['final_exact_rate']:>8.3f} | {tr_avg:>8.3f}")

    # 結果をJSONで保存
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = os.path.join(root, "logs", "ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果を保存: {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="アブレーション解析")
    parser.add_argument("--turns", type=int, default=2000, help="最大ターン数")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="実行する条件（省略で全条件）")
    parser.add_argument("--quiet", action="store_true", help="ターンごとの出力を省略")
    args = parser.parse_args()

    conditions = args.conditions
    if conditions:
        for c in conditions:
            if c not in ABLATION_CONDITIONS:
                print(f"不明な条件: {c}")
                print(f"利用可能: {list(ABLATION_CONDITIONS.keys())}")
                sys.exit(1)

    run_all_ablations(
        conditions=conditions,
        max_turns=args.turns,
        verbose=not args.quiet,
    )
