"""
モデルBで目標Aが達成されるか確認する

Bの構成（身体シミュレーション＋島皮質）でゼロから学習して、
オウム返し（Stage 4）に到達するかを検証する。

親が頻繁に話しかける設定で、Aと同等の条件を再現する。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.core_b import TaroEnvironmentB
from environment.parent_sim import ParentSmile

PHRASES = ["まま", "ばば", "ないない", "まんま", "ねんね"]
TEST_PHRASES = ["ぱぱ", "だだ", "いやいや", "ぶぶ", "にに"]
MAX_INTERACTIONS = 3000


def main():
    env = TaroEnvironmentB(run_name="B_goalA_test")
    smile = ParentSmile()

    prev_r_imit = 0.0
    prev_taro = ""
    target = env.partial_streak_target
    stage4_turn = None
    best_partial = 0

    print("=== モデルBで目標A（オウム返し）達成テスト ===")
    print(f"フレーズ: {PHRASES}")
    print(f"成功条件: 模倣スコア≥0.8が{target}回連続")
    print()

    for i in range(MAX_INTERACTIONS):
        # 毎回身体を少し進める（食事間隔をシミュレート）
        env.tick_body(elapsed_seconds=5)

        # 定期的に食事（空腹を低く保ってオウム返し学習に集中させる）
        if i % 20 == 0:
            env.feed(0.6)

        phrase = PHRASES[i % len(PHRASES)]
        r_social = smile.compute_smile(prev_r_imit, prev_taro) if i > 0 else 0.0

        result = env.step(phrase, r_social=r_social)
        prev_r_imit = result["r_imit"]
        prev_taro = result["taro"]

        ps = result["partial_streak"]
        if ps > best_partial:
            best_partial = ps

        if i < 20 or i % 100 == 0:
            print(f"  t{i:4d} | S{env.vocal_tract.stage} | "
                  f"[{result['parent']}]->[{result['taro']:<8s}] | "
                  f"模倣={result['r_imit']:.2f} R={result['R']:.2f} "
                  f"hunger={result['hunger']:.2f} "
                  f"arousal={result['arousal']:.2f} "
                  f"streak={ps}")

        if stage4_turn is None and ps >= target:
            stage4_turn = i
            print(f"\n*** Stage 4（オウム返し）達成！ interaction {i} ***\n")
            break

    # 転移テスト
    print(f"\n=== 転移テスト ===")
    for tp in TEST_PHRASES:
        scores, exact = [], 0
        for _ in range(5):
            r = env.step(tp, r_social=0.0)
            scores.append(r["r_imit"])
            if r["exact_match"]:
                exact += 1
        avg = sum(scores) / len(scores)
        print(f"  [{tp}] 模倣={avg:.2f} 完全一致={exact}/5")

    print(f"\n=== 結果 ===")
    print(f"Stage 4: {'達成(t' + str(stage4_turn) + ')' if stage4_turn else '未達成'}")
    print(f"最大streak: {best_partial}")


if __name__ == "__main__":
    main()
