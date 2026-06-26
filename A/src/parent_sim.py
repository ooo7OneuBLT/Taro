"""
親シミュレータ — 対乳児発話を自動投入して収束テストを行う

【人間模倣】親は乳児に短くやさしい言葉で話しかける（motherese）。
これを自動化し、「本能のみからオウム返しが創発する」証拠を取る。

A2追加：
- 笑顔の自動ロジック（驚き→馴化→進歩で再び喜ぶ）
- 転移テスト（未見フレーズでの汎化確認）
- 成功条件の2段階化（部分一致≥0.8 ＋ 完全一致）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from environment import TaroEnvironment


class ParentSmile:
    """
    親の笑顔の自動ロジック【人間模倣】

    - 初めて似た音を出した → 大きく笑う（驚き）
    - 同じことを繰り返しできる → 反応が薄くなる（馴化）
    - 前より上手くなった → また大きく笑う（進歩への反応）
    """

    def __init__(self, history_size=10):
        self.history = []
        self.history_size = history_size
        self.best_r_imit = 0.0

    def compute_smile(self, r_imit):
        avg = sum(self.history) / len(self.history) if self.history else 0.0
        smile = 0.0

        if r_imit > self.best_r_imit:
            smile = min(1.0, 0.5 + (r_imit - self.best_r_imit))
            self.best_r_imit = r_imit
        elif r_imit > avg + 0.1:
            smile = min(1.0, 0.3 + (r_imit - avg))
        elif r_imit >= 0.8:
            smile = max(0.1, 0.4 - len(self.history) * 0.02)
        else:
            smile = 0.0

        self.history.append(r_imit)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return max(0.0, min(1.0, smile))


def run_simulation(max_turns=1000, phrases=None, test_phrases=None,
                    verbose=True, run_name=None, description=None):
    """
    親シミュレータを実行する。

    max_turns: 最大ターン数
    phrases: 親が繰り返す発話リスト（motherese）
    test_phrases: 転移テスト用の未見フレーズ（学習には使わない）
    verbose: 経過表示するか
    run_name: ログフォルダ名
    description: この実験の説明
    """
    if phrases is None:
        phrases = ["まま", "ばぁ", "ないない"]
    if test_phrases is None:
        test_phrases = ["ぱぱ", "だっこ"]

    env = TaroEnvironment(run_name=run_name)
    smile = ParentSmile()

    if description is None:
        description = f"A2 phrases={phrases}, test={test_phrases}, max_turns={max_turns}"
    env.logger.save_run_info(description, env.cfg, phrases=phrases)

    env.save(tag="day1_before_learning")

    partial_target = env.partial_streak_target
    exact_target = env.exact_streak_target

    if verbose:
        print(f"=== 親シミュレータ開始（A2） ===")
        print(f"訓練フレーズ: {phrases}")
        print(f"テストフレーズ（未見）: {test_phrases}")
        print(f"基本達成: 類似度≥{env.partial_threshold} が{partial_target}回連続")
        print(f"完全達成: 完全一致が{exact_target}回連続")
        print(f"最大ターン: {max_turns}")
        print()

    best_partial = 0
    best_exact = 0
    partial_success_turn = None
    exact_success_turn = None

    for turn_idx in range(max_turns):
        phrase = phrases[turn_idx % len(phrases)]

        # 前回のr_imitで笑顔を計算（初回は0）
        r_social = 0.0
        if turn_idx > 0:
            r_social = smile.compute_smile(prev_r_imit)

        result = env.step(phrase, r_social=r_social)
        prev_r_imit = result["r_imit"]

        ps = result["partial_streak"]
        es = result["exact_streak"]
        if ps > best_partial:
            best_partial = ps
        if es > best_exact:
            best_exact = es

        if verbose and (turn_idx < 20 or turn_idx % 50 == 0
                        or result["exact_match"] or result["partial_match"]):
            p_mark = "P" if result["partial_match"] else " "
            e_mark = "O" if result["exact_match"] else " "
            print(
                f"[{e_mark}{p_mark}] turn {result['turn']:4d} | "
                f"age={result['age']:>6s} | "
                f"parent=[{result['parent']}] taro=[{result['taro']:<8s}] | "
                f"r_imit={result['r_imit']:.2f} smile={r_social:.2f} "
                f"R={result['R']:.2f} | "
                f"p_streak={ps}/{partial_target} "
                f"e_streak={es}/{exact_target} "
                f"stam={result['stamina']:.1f} "
                f"tau={result['temperature']:.3f}"
            )

        if partial_success_turn is None and ps >= partial_target:
            partial_success_turn = result["turn"]
            if verbose:
                print(f"\n*** 基本達成！ turn {partial_success_turn}, "
                      f"age={result['age']} ***\n")

        if exact_success_turn is None and es >= exact_target:
            exact_success_turn = result["turn"]
            if verbose:
                print(f"\n*** 完全達成！ turn {exact_success_turn}, "
                      f"age={result['age']} ***")
            break

    # --- 転移テスト ---
    if verbose:
        print(f"\n=== 転移テスト（未見フレーズ） ===")

    transfer_results = []
    for tp in test_phrases:
        scores = []
        for _ in range(5):
            result = env.step(tp, r_social=0.0)
            scores.append(result["r_imit"])
        avg_score = sum(scores) / len(scores)
        transfer_results.append({"phrase": tp, "avg_r_imit": avg_score, "scores": scores})
        if verbose:
            print(f"  [{tp}] avg_r_imit={avg_score:.2f}  scores={[f'{s:.2f}' for s in scores]}")

    env.logger.plot_learning_curve()
    env.save(tag="final")

    if verbose:
        if partial_success_turn is None:
            print(f"\n{max_turns}ターンで基本達成に未到達。最大連続: {best_partial}")
        print(f"\nログ: {env.logger.log_dir}")

    return {
        "partial_success": partial_success_turn is not None,
        "partial_success_turn": partial_success_turn,
        "exact_success": exact_success_turn is not None,
        "exact_success_turn": exact_success_turn,
        "best_partial_streak": best_partial,
        "best_exact_streak": best_exact,
        "total_turns": env.clock.total_turns,
        "total_tokens_heard": env.clock.total_tokens_heard,
        "final_age": env.clock.age_str(),
        "transfer_results": transfer_results,
    }


if __name__ == "__main__":
    result = run_simulation(
        max_turns=2000,
        phrases=["まま", "ばぁ", "ないない", "まんま", "ねんね"],
        test_phrases=["ぱぱ", "だっこ", "いやいや"],
        run_name="A2-1_initial",
        description="A2-1初回テスト：発声体力＋edit distance＋τ適応減衰＋笑顔自動＋転移テスト",
    )
    print(f"\n結果: {result}")
