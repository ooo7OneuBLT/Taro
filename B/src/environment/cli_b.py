"""
CLI（目標B用）— あなたが太郎の親になって直接話しかける

モデルAのCLIとの違い：
- 太郎は話しかけていない間も身体が動いている（空腹が進む）
- 世話コマンドがある（/feed, /comfort, /hold）
- 太郎の内部状態（空腹・arousal）が見える
- 太郎は自分で泣く
"""

import sys
import os
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core_b import TaroEnvironmentB


class CliParent:
    """手動モードの環境ループ。裏で身体シミュレーションが進む。"""

    def __init__(self):
        self.env = TaroEnvironmentB(run_name="cli_b_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.sim_seconds = 0
        self.running = True
        self._cry_log = []

    def tick_loop(self):
        """バックグラウンドで身体を進めるループ。1秒=1sim秒。"""
        while self.running:
            self.env.tick_body(elapsed_seconds=1)
            self.sim_seconds += 1

            cry, intensity = self.env.check_cry()
            if cry:
                msg = f"[t={self.sim_seconds}s] 太郎が泣いている（{intensity:.2f}）"
                self._cry_log.append(msg)
                if len(self._cry_log) <= 5 or len(self._cry_log) % 20 == 0:
                    print(f"\n  ** {msg}")
                    print("親: ", end="", flush=True)

            time.sleep(0.05)

    def start(self):
        self.thread = threading.Thread(target=self.tick_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False


def print_header():
    print("=" * 60)
    print("  太郎 B — 初語を目指して（身体シミュレーション搭載）")
    print("  あなたは太郎の「親」です。話しかけてください。")
    print("  太郎は話しかけていない間も空腹になり、泣きます。")
    print()
    print("  話しかける: そのまま文字を入力")
    print("  世話コマンド:")
    print("    /feed [量]    ごはんをあげる（量: 0.1〜1.0、省略=0.6）")
    print("    /comfort      あやす")
    print("    /hold         抱っこする")
    print("    /status       太郎の状態を見る")
    print("    /pause        時間を止める")
    print("    /resume       時間を再開する")
    print("    /save [名前]  保存する")
    print("    /quit         終了")
    print("=" * 60)
    print()


def run_cli_b():
    parent = CliParent()
    env = parent.env
    print_header()

    print("時間が流れ始めます。太郎が泣いたら世話をしてください。")
    parent.start()

    while True:
        try:
            user_input = input("親: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                env.save(tag="quit")
                parent.stop()
                print("保存して終了します。")
                break

            elif cmd == "/feed":
                amount = 0.6
                if arg:
                    try:
                        amount = max(0.1, min(1.0, float(arg)))
                    except ValueError:
                        pass
                env.feed(amount)
                result = env.step("まんま", r_social=0.3)
                print(f"  ごはんをあげた（量={amount:.1f}）「まんま」")
                print(f"  太郎: {result['taro']} | hunger={result['hunger']:.2f}")

            elif cmd == "/comfort":
                env.comfort("comfort")
                result = env.step("よしよし", r_social=0.5)
                print(f"  あやした「よしよし」")
                print(f"  太郎: {result['taro']} | arousal={result['arousal']:.2f}")

            elif cmd == "/hold":
                env.comfort("hold")
                result = env.step("まま", r_social=0.5)
                print(f"  抱っこした「まま」")
                print(f"  太郎: {result['taro']} | arousal={result['arousal']:.2f}")

            elif cmd == "/status":
                s = env.internal_state
                print(f"  sim時間: {parent.sim_seconds}秒 ({parent.sim_seconds//60}分)")
                print(f"  空腹: {s.hunger:.3f}")
                print(f"  眠さ: {s.sleepiness:.3f}")
                print(f"  不快: {s.discomfort:.3f}")
                print(f"  興奮: {s.get_arousal():.3f}")
                print(f"  胃の中身: {env.stomach.contents:.3f} / {env.stomach.capacity:.3f}")
                print(f"  肺: {env.lungs.get_max_mora()}モーラ分")
                print(f"  声道Stage: {env.vocal_tract.stage}")
                print(f"  泣いた回数: {len(parent._cry_log)}")

            elif cmd == "/pause":
                parent.running = False
                print("  時間を止めました。/resume で再開。")

            elif cmd == "/resume":
                if not parent.running:
                    parent.running = True
                    parent.thread = threading.Thread(target=parent.tick_loop, daemon=True)
                    parent.thread.start()
                    print("  時間を再開しました。")

            elif cmd == "/save":
                tag = arg if arg else "manual"
                path = env.save(tag=tag)
                print(f"  保存: {path}")

            else:
                print(f"  不明なコマンド: {cmd}")

            continue

        # 通常の発話
        result = env.step(user_input, r_social=0.0)
        mark = "O" if result["exact_match"] else " "
        print(f"  太郎: {result['taro']}")
        print(f"  [{mark}] 模倣={result['r_imit']:.2f} R={result['R']:.2f} "
              f"恒常性={result['r_home']:.2f} hunger={result['hunger']:.2f} "
              f"arousal={result['arousal']:.2f}")

        try:
            smile = input("  笑顔(0〜1, Enter=0): ").strip()
            if smile:
                r = max(0.0, min(1.0, float(smile)))
                if r > 0:
                    env.step(user_input, r_social=r)
                    print(f"  (笑顔 {r:.1f})")
        except ValueError:
            pass


if __name__ == "__main__":
    run_cli_b()
