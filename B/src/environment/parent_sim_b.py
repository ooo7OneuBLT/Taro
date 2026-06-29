"""
親シミュレータ（目標B用）— イベント駆動で太郎を育てる

モデルAの親：毎ターン話しかけるだけ。
モデルBの親：在/不在があり、泣いたら来て、世話をしながら言葉を添える。

スケジュールは config/schedule.yaml で実験ごとに変更可能。
"""

import sys
import os
import random
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core_b import TaroEnvironmentB


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ParentSchedule:
    """
    親の在/不在スケジュール。schedule.yamlから読み込む。
    """

    def __init__(self, schedule_path=None):
        if schedule_path is None:
            schedule_path = os.path.join(_project_root(), "config", "schedule.yaml")

        with open(schedule_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("presence", {})
        self.presence_prob = p.get("prob", 0.7)
        self.check_interval = p.get("check_interval", 100)

        r = cfg.get("respond", {})
        self.respond_prob = r.get("prob", 0.9)
        self.respond_prob_absent = r.get("prob_absent", 0.1)
        self.respond_delay_min = r.get("delay_min", 1)
        self.respond_delay_max = r.get("delay_max", 30)
        self.respond_delay_max_absent = r.get("delay_max_absent", 90)

        fd = cfg.get("feeding", {})
        self.feed_interval = fd.get("interval", 600)
        self.feed_amount = fd.get("amount", 0.6)

        sp = cfg.get("speech", {})
        self.speak_with_care = sp.get("enabled", True)
        self.words = sp.get("words", {"feed": "まんま", "comfort": "よしよし", "hold": "まま"})

        sim = cfg.get("simulation", {})
        self.max_seconds = sim.get("max_seconds", 3600)
        self.log_interval = sim.get("log_interval", 300)

        self.present = True
        self.last_feed_time = 0
        self._pending_respond = None

    def update_presence(self, sim_seconds):
        if sim_seconds % self.check_interval == 0:
            self.present = random.random() < self.presence_prob

    def should_feed(self, sim_seconds):
        return sim_seconds - self.last_feed_time >= self.feed_interval

    def on_cry(self, sim_seconds):
        if not self.present:
            if random.random() < self.respond_prob_absent:
                delay = random.randint(self.respond_delay_max, self.respond_delay_max_absent)
                self._pending_respond = sim_seconds + delay
                self.present = True
            return False

        if random.random() < self.respond_prob:
            delay = random.randint(self.respond_delay_min, self.respond_delay_max)
            self._pending_respond = sim_seconds + delay
            return True
        return False

    def should_respond_now(self, sim_seconds):
        if self._pending_respond is not None and sim_seconds >= self._pending_respond:
            self._pending_respond = None
            return True
        return False

    def choose_word(self, care_type):
        if not self.speak_with_care:
            return None
        return self.words.get(care_type, None)


def run_simulation_b(max_sim_seconds=None, verbose=True, run_name=None,
                     schedule_path=None):
    """
    B用シミュレーション。イベント駆動で太郎を育てる。
    """
    env = TaroEnvironmentB(run_name=run_name or "B_sim")
    schedule = ParentSchedule(schedule_path=schedule_path)

    if max_sim_seconds is None:
        max_sim_seconds = schedule.max_seconds

    cry_count = 0
    feed_count = 0
    speak_count = 0
    sim_seconds = 0

    if verbose:
        print(f"=== 目標B シミュレーション開始 ===")
        print(f"最大: {max_sim_seconds}秒 ({max_sim_seconds//60}分)")
        print(f"親在確率: {schedule.presence_prob} 反応確率: {schedule.respond_prob}")
        print(f"食事間隔: {schedule.feed_interval}秒 食事量: {schedule.feed_amount}")
        print()

    sleep_count = 0
    was_crying = False
    was_drowsy = False

    def fmt_time(s):
        """秒数を 時:分:秒 に変換する。"""
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    while sim_seconds < max_sim_seconds:
        # 寝ている間は時間を飛ばす
        if env.internal_state.is_sleeping():
            skip = env.internal_state._sleep_remaining
            env.tick_body(elapsed_seconds=skip)
            sim_seconds += skip
            sleep_count += 1
            if verbose:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"起きた（{skip//60}分寝た）| "
                      f"hunger={env.internal_state.hunger:.2f}")
            continue

        # うとうと中は時間を飛ばす
        if env.internal_state.is_drowsy():
            if not was_drowsy and verbose:
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | うとうと...")
            was_drowsy = True
            skip = env.internal_state._drowsy_remaining
            env.tick_body(elapsed_seconds=skip)
            sim_seconds += skip
            was_drowsy = False
            continue

        env.tick_body(elapsed_seconds=1)
        sim_seconds += 1
        schedule.update_presence(sim_seconds)

        # 泣きの検出（泣き始めたときだけ通知）
        crying_now, intensity = env.check_cry()
        if crying_now and not was_crying:
            cry_count += 1
            if verbose and (cry_count <= 20 or cry_count % 50 == 0):
                print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                      f"泣き始めた（強さ{intensity:.2f}）| "
                      f"hunger={env.internal_state.hunger:.2f} "
                      f"つらさ={env.internal_state.get_arousal():.2f} "
                      f"親{'在' if schedule.present else '不在'}")
            schedule.on_cry(sim_seconds)
        was_crying = crying_now

        if schedule.should_respond_now(sim_seconds):
            care_type = "feed" if env.internal_state.hunger > 0.5 else "comfort"
            if care_type == "feed":
                env.feed(schedule.feed_amount)
                schedule.last_feed_time = sim_seconds
                feed_count += 1
            env.comfort(care_type)

            word = schedule.choose_word(care_type)
            if word:
                result = env.step(word, r_social=0.5)
                speak_count += 1
                if verbose and (speak_count <= 30 or speak_count % 50 == 0):
                    print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                          f"親「{word}」→ 太郎「{result['taro']}」| "
                          f"模倣={result['r_imit']:.2f} hunger={result['hunger']:.2f}")

        # 定期的な授乳（泣かなくても）。授乳中でなければ開始
        if (schedule.present and schedule.should_feed(sim_seconds)
                and not env.stomach.is_feeding()):
            env.feed(schedule.feed_amount)
            schedule.last_feed_time = sim_seconds
            feed_count += 1
            word = schedule.choose_word("feed")
            if word:
                result = env.step(word, r_social=0.3)
                speak_count += 1
                if verbose:
                    print(f"  t={sim_seconds:5d}s ({fmt_time(sim_seconds)}) | "
                          f"授乳開始「{word}」→ 太郎「{result['taro']}」| "
                          f"hunger={result['hunger']:.2f}")

        if verbose and sim_seconds % schedule.log_interval == 0:
            state = "寝" if env.internal_state.is_sleeping() else \
                    "うとうと" if env.internal_state.is_drowsy() else \
                    "泣き" if env.internal_state.is_crying() else "起きてる"
            feeding = " 授乳中" if env.stomach.is_feeding() else ""
            print(f"  --- {fmt_time(sim_seconds)} | "
                  f"泣き{cry_count} 食事{feed_count} 発話{speak_count} 睡眠{sleep_count} | "
                  f"hunger={env.internal_state.hunger:.2f} "
                  f"つらさ={env.internal_state.get_arousal():.2f} | "
                  f"{state}{feeding}")

    if verbose:
        print(f"\n=== シミュレーション完了 ===")
        print(f"時間: {sim_seconds}秒 ({sim_seconds//60}分 = {sim_seconds/3600:.1f}時間)")
        print(f"泣き: {cry_count}回  食事: {feed_count}回  発話: {speak_count}回  睡眠: {sleep_count}回")

    return {
        "sim_seconds": sim_seconds,
        "cry_count": cry_count,
        "feed_count": feed_count,
        "speak_count": speak_count,
        "sleep_count": sleep_count,
        "env": env,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="目標B シミュレーション")
    parser.add_argument("--seconds", type=int, default=None)
    parser.add_argument("--schedule", type=str, default=None, help="schedule.yamlのパス")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_simulation_b(
        max_sim_seconds=args.seconds,
        verbose=not args.quiet,
        schedule_path=args.schedule,
    )
