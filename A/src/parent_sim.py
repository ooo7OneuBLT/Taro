"""
親シミュレータ — 対乳児発話を自動投入して収束テストを行う

【人間模倣】親は乳児に短くやさしい言葉で話しかける（motherese）。
これを自動化し、「本能のみからオウム返しが創発する」証拠を取る。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from environment import TaroEnvironment


def run_simulation(max_turns=1000, success_streak=10, phrases=None,
                    verbose=True, run_name=None, description=None):
    """
    親シミュレータを実行する。

    max_turns: 最大ターン数
    success_streak: 何回連続一致で成功とするか
    phrases: 親が繰り返す発話リスト（motherese）
    verbose: 経過表示するか
    run_name: ログフォルダ名（Noneなら自動タイムスタンプ）
    description: この実験の説明（何を変えたか・なぜ）
    """
    if phrases is None:
        phrases = ["あ", "い", "う"]

    env = TaroEnvironment(run_name=run_name)

    if description is None:
        description = f"phrases={phrases}, max_turns={max_turns}"
    env.logger.save_run_info(description, env.cfg, phrases=phrases)

    # Day1（学習前）を保存
    env.save(tag="day1_before_learning")

    if verbose:
        print(f"=== 親シミュレータ開始 ===")
        print(f"発話セット: {phrases}")
        print(f"成功基準: {success_streak}回連続一致")
        print(f"最大ターン: {max_turns}")
        print()

    best_streak = 0
    success_turn = None

    for turn_idx in range(max_turns):
        phrase = phrases[turn_idx % len(phrases)]
        result = env.step(phrase, r_social=0.0)

        streak = result["consecutive_matches"]
        if streak > best_streak:
            best_streak = streak

        if verbose and (turn_idx < 20 or turn_idx % 50 == 0 or result["exact_match"]):
            mark = "O" if result["exact_match"] else "X"
            print(
                f"[{mark}] turn {result['turn']:4d} | "
                f"age={result['age']:>6s} | "
                f"parent=[{result['parent']}] taro=[{result['taro']:<6s}] | "
                f"r_imit={result['r_imit']:.2f} R={result['R']:.2f} "
                f"delta={result['delta']:+.3f} | "
                f"streak={streak}/{success_streak} "
                f"tau={result['temperature']:.3f}"
            )

        if streak >= success_streak:
            success_turn = result["turn"]
            if verbose:
                print(f"\n*** A1目標達成！ turn {success_turn}, "
                      f"age={result['age']}, "
                      f"tokens_heard={env.clock.total_tokens_heard} ***")
            break

    env.logger.plot_learning_curve()
    env.save(tag="final")

    if verbose:
        if success_turn is None:
            print(f"\n{max_turns}ターンで目標未達成。最大連続一致: {best_streak}")
        print(f"\nログ: {env.logger.log_dir}")
        print(f"学習曲線: {env.logger.log_dir}/learning_curve.png")

    return {
        "success": success_turn is not None,
        "success_turn": success_turn,
        "best_streak": best_streak,
        "total_turns": env.clock.total_turns,
        "total_tokens_heard": env.clock.total_tokens_heard,
        "final_age": env.clock.age_str(),
    }


if __name__ == "__main__":
    result = run_simulation(max_turns=2000, success_streak=10)
    print(f"\n結果: {result}")
