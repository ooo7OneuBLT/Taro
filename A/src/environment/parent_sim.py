"""
親シミュレータ — 発達段階ごとの成功基準で太郎を育てる

A2-5：成功基準を人間の発達段階に合わせて再定義。
Stage 1-3は口の発達の準備段階。オウム返しはStage 4で初めて求める。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import TaroEnvironment


class ParentSmile:
    """
    親の笑顔の自動ロジック【人間模倣】

    A2-8：親の笑顔を2条件の掛け合わせで決定。
    ① 親の発話に似ているか（r_imitが閾値以上か）
    ② 繰り返しではないか（同じ出力は回数で減衰）

    全然違う音（r_imit < 閾値）→ 笑わない（反応しない）
    似ている音 → 笑う
    同じ音の繰り返し → 回数ごとに笑顔が半減（親も飽きる）
    """

    def __init__(self, min_r_imit=0.3, history_size=10):
        self.min_r_imit = min_r_imit
        self.output_history = []
        self.history_size = history_size
        self.best_r_imit = 0.0

    def compute_smile(self, r_imit, taro_output):
        """
        r_imit: 親の発話との類似度
        taro_output: 太郎が出した文字列
        """
        # ① 類似度が閾値未満 → 反応しない
        if r_imit < self.min_r_imit:
            self._add(taro_output)
            return 0.0

        # 基本笑顔：類似度に比例
        base_smile = r_imit

        # 過去最高を超えたらボーナス
        if r_imit > self.best_r_imit:
            base_smile = min(1.0, base_smile + 0.3)
            self.best_r_imit = r_imit

        # ② 繰り返しによる減衰：同じ出力の回数で半減
        repeat_count = sum(1 for h in self.output_history if h == taro_output)
        decay = 0.5 ** repeat_count  # 1回目=1.0, 2回目=0.5, 3回目=0.25...

        smile = base_smile * decay

        self._add(taro_output)
        return max(0.0, min(1.0, smile))

    def _add(self, output):
        self.output_history.append(output)
        if len(self.output_history) > self.history_size:
            self.output_history.pop(0)


class StageTracker:
    """
    発達段階の進捗を追跡し、各Stageの成功を判定する。

    Stage 1（クーイング）：直近20ターンで3種類以上の母音。固着していない
    Stage 2（声の探索）：直近20ターンで5種類以上の文字。子音が出始める
    Stage 3（規準喃語）：子音＋母音の組み合わせが半分以上のターンで出る
    Stage 4（意図的模倣）：親との類似度≥0.8が10回連続
    """

    VOWELS = set("あいうえお")

    def __init__(self, window=20):
        self.window = window
        self.recent_outputs = []
        self.stage_achieved = {1: False, 2: False, 3: False, 4: False}
        self.stage_achieved_turn = {}

    def update(self, taro_text, turn):
        self.recent_outputs.append(taro_text)
        if len(self.recent_outputs) > self.window:
            self.recent_outputs.pop(0)

    def check_stage1(self):
        """クーイング：3種類以上の母音が出ている"""
        if len(self.recent_outputs) < self.window:
            return False
        all_chars = "".join(self.recent_outputs)
        vowels_used = set(ch for ch in all_chars if ch in self.VOWELS)
        most_common = max(set(self.recent_outputs), key=self.recent_outputs.count) if self.recent_outputs else ""
        fixation = self.recent_outputs.count(most_common) / len(self.recent_outputs)
        return len(vowels_used) >= 3 and fixation < 0.75

    def check_stage2(self):
        """声の探索：5種類以上の文字（子音含む）"""
        if len(self.recent_outputs) < self.window:
            return False
        all_chars = "".join(self.recent_outputs)
        unique_chars = set(all_chars)
        has_consonant = any(ch not in self.VOWELS and ch != "" for ch in unique_chars)
        return len(unique_chars) >= 5 and has_consonant

    def check_stage3(self):
        """規準喃語：半分以上のターンで子音＋母音の組み合わせ"""
        if len(self.recent_outputs) < self.window:
            return False
        cv_count = 0
        for text in self.recent_outputs:
            has_c = any(ch not in self.VOWELS for ch in text)
            has_v = any(ch in self.VOWELS for ch in text)
            if has_c and has_v:
                cv_count += 1
        return cv_count >= self.window // 2

    def check_all(self, turn):
        """全Stageをチェックし、新たに達成したStageを返す。"""
        newly_achieved = []
        for stage, check_fn in [(1, self.check_stage1), (2, self.check_stage2),
                                 (3, self.check_stage3)]:
            if not self.stage_achieved[stage] and check_fn():
                self.stage_achieved[stage] = True
                self.stage_achieved_turn[stage] = turn
                newly_achieved.append(stage)
        return newly_achieved


def run_simulation(max_turns=3000, phrases=None, test_phrases=None,
                    verbose=True, run_name=None, description=None):
    if phrases is None:
        phrases = ["まま", "ばぁ", "ないない"]
    if test_phrases is None:
        test_phrases = ["ぱぱ", "だっこ"]

    env = TaroEnvironment(run_name=run_name)
    smile = ParentSmile()
    tracker = StageTracker(window=20)

    if description is None:
        description = f"A2-5 Stage-based: phrases={phrases}, max_turns={max_turns}"
    env.logger.save_run_info(description, env.cfg, phrases=phrases)
    env.save(tag="day1_before_learning")

    if verbose:
        print(f"=== 親シミュレータ開始（Stage別評価） ===")
        print(f"訓練: {phrases}")
        print(f"テスト: {test_phrases}")
        print(f"Stage 1: 3種類以上の母音（クーイング）")
        print(f"Stage 2: 5種類以上の文字＋子音（声の探索）")
        print(f"Stage 3: 子音＋母音の組み合わせが半数以上（規準喃語）")
        print(f"Stage 4: 類似度≥0.8が10回連続（オウム返し）")
        print()

    prev_r_imit = 0.0
    prev_taro_output = ""
    partial_target = env.partial_streak_target
    best_partial = 0

    for turn_idx in range(max_turns):
        phrase = phrases[turn_idx % len(phrases)]
        r_social = smile.compute_smile(prev_r_imit, prev_taro_output) if turn_idx > 0 else 0.0

        result = env.step(phrase, r_social=r_social)
        prev_r_imit = result["r_imit"]
        prev_taro_output = result["taro"]

        tracker.update(result["taro"], result["turn"])
        newly = tracker.check_all(result["turn"])

        ps = result["partial_streak"]
        if ps > best_partial:
            best_partial = ps

        if verbose and (turn_idx < 20 or turn_idx % 100 == 0 or newly):
            print(
                f"t{result['turn']:4d} | "
                f"age={result['age']:>6s} "
                f"S{env.vocal_tract.stage} | "
                f"[{result['parent']}]->[{result['taro']:<8s}] | "
                f"r_imit={result['r_imit']:.2f} "
                f"tau={result['temperature']:.2f} "
                f"stam={result['stamina']:.1f}"
            )

        for s in newly:
            if verbose:
                print(f"\n*** Stage {s} 達成！ turn {result['turn']}, age={result['age']} ***\n")

        if not tracker.stage_achieved[4] and ps >= partial_target:
            tracker.stage_achieved[4] = True
            tracker.stage_achieved_turn[4] = result["turn"]
            if verbose:
                print(f"\n*** Stage 4（オウム返し）達成！ turn {result['turn']}, age={result['age']} ***")
            break

    # 転移テスト
    if verbose:
        print(f"\n=== 転移テスト ===")
    transfer_results = []
    for tp in test_phrases:
        scores = []
        for _ in range(5):
            r = env.step(tp, r_social=0.0)
            scores.append(r["r_imit"])
        avg = sum(scores) / len(scores)
        transfer_results.append({"phrase": tp, "avg_r_imit": avg})
        if verbose:
            print(f"  [{tp}] avg={avg:.2f}")

    env.logger.plot_learning_curve()
    env.save(tag="final")

    if verbose:
        print(f"\n=== Stage達成状況 ===")
        for s in [1, 2, 3, 4]:
            if tracker.stage_achieved[s]:
                t = tracker.stage_achieved_turn[s]
                print(f"  Stage {s}: 達成 (turn {t})")
            else:
                print(f"  Stage {s}: 未達成")
        print(f"\n最大partial streak: {best_partial}")

    return {
        "stages": dict(tracker.stage_achieved),
        "stage_turns": dict(tracker.stage_achieved_turn),
        "best_partial_streak": best_partial,
        "total_turns": env.clock.total_turns,
        "final_age": env.clock.age_str(),
        "transfer_results": transfer_results,
    }


if __name__ == "__main__":
    result = run_simulation(
        max_turns=3000,
        phrases=["まま", "ばぁ", "ないない", "まんま", "ねんね"],
        test_phrases=["ぱぱ", "だっこ", "いやいや"],
        run_name="A2-5_stage_based",
        description="A2-5: Stage別成功基準で発達を追跡",
    )
    print(f"\n結果: {result}")
