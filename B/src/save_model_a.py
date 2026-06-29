"""
モデルA完了版の学習・保存

目標B（内部状態あり・ゼロから連続学習）の比較用ベースラインとして、
内部状態なしのモデルA（オウム返し完成版）を学習して保存する。

保存タグ：A_complete
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.core import TaroEnvironment
from environment.parent_sim import ParentSmile, StageTracker

PHRASES = ["まま", "ばば", "ないない", "まんま", "ねんね"]
TEST_PHRASES = ["ぱぱ", "だだ", "いやいや", "ぶぶ", "にに"]
MAX_TURNS = 2000


def main():
    env = TaroEnvironment(run_name="modelA_complete")
    smile = ParentSmile()
    tracker = StageTracker(window=20)
    env.logger.save_run_info("モデルA完了版（目標Bのベースライン）", env.cfg, phrases=PHRASES)

    prev_r_imit = 0.0
    prev_taro = ""
    target = env.partial_streak_target
    stage4_turn = None

    print("=== モデルA 学習開始 ===")
    for i in range(MAX_TURNS):
        phrase = PHRASES[i % len(PHRASES)]
        r_social = smile.compute_smile(prev_r_imit, prev_taro) if i > 0 else 0.0
        result = env.step(phrase, r_social=r_social)
        prev_r_imit = result["r_imit"]
        prev_taro = result["taro"]

        if stage4_turn is None and result["partial_streak"] >= target:
            stage4_turn = result["turn"]
            print(f"*** Stage 4（オウム返し）達成！ turn {stage4_turn} ***")
            break

    # 転移テスト（学習していないフレーズ）
    print("\n=== 転移テスト ===")
    transfer = []
    for tp in TEST_PHRASES:
        scores, exact = [], 0
        for _ in range(5):
            r = env.step(tp, r_social=0.0)
            scores.append(r["r_imit"])
            if r["exact_match"]:
                exact += 1
        avg = sum(scores) / len(scores)
        transfer.append(avg)
        print(f"  [{tp}] 模倣={avg:.2f} 完全一致={exact/5:.0%}")

    transfer_avg = sum(transfer) / len(transfer)
    print(f"\n転移テスト平均模倣スコア：{transfer_avg:.3f}")

    # 保存
    path = env.save(tag="A_complete")
    print(f"\n=== モデルA完了版を保存 ===")
    print(f"Stage4到達：turn {stage4_turn}")
    print(f"保存先：{path}")


if __name__ == "__main__":
    main()
