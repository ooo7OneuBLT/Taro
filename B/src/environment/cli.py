"""
CLI — 親（あなた）と太郎の対話インターフェース
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.core import TaroEnvironment


def print_header():
    print("=" * 56)
    print("  太郎 A2 — オウム返し（声道シミュレータ搭載）")
    print("  あなたは太郎の「親」です。話しかけてください。")
    print("  太郎の返答のあと、笑顔の強さ(0.0〜1.0)を入力。")
    print("  コマンド: /save /load /fork /stats /plot /quit")
    print("=" * 56)
    print()


def run_cli():
    env = TaroEnvironment(run_name="cli_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    env.logger.save_run_info("CLI対話セッション", env.cfg)
    env.save(tag="day1_cli")
    print_header()

    while True:
        try:
            parent_input = input("親: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not parent_input:
            continue

        if parent_input.startswith("/"):
            parts = parent_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                env.save(tag="quit")
                print("保存して終了します。")
                break
            elif cmd == "/save":
                tag = arg if arg else "manual"
                path = env.save(tag=tag)
                print(f"保存しました: {path}")
            elif cmd == "/load":
                snaps = env.archive.list_snapshots()
                if not snaps:
                    print("スナップショットがありません。")
                else:
                    for i, s in enumerate(snaps):
                        print(f"  [{i}] {s['name']} (turn={s['turn']})")
                    try:
                        choice = input("番号を入力: ").strip()
                        if choice:
                            env.load(snaps[int(choice)]["path"])
                            print(f"復元しました")
                    except (ValueError, IndexError):
                        print("キャンセル")
            elif cmd == "/stats":
                print(f"  ターン: {env.clock.total_turns}")
                print(f"  発達年齢: {env.clock.age_str()}")
                print(f"  声道ステージ: {env.vocal_tract.stage}")
                print(f"  体力: {env.stamina.get():.1f}")
                print(f"  温度(tau): {env.brain.temperature:.4f}")
                print(f"  語彙数: {env.vocab.size}")
            elif cmd == "/plot":
                path = env.logger.plot_learning_curve()
                print(f"学習曲線: {path}" if path else "データ不足")
            else:
                print(f"不明なコマンド: {cmd}")
            continue

        result = env.step(parent_input, r_social=0.0)

        mark = "O" if result["exact_match"] else " "
        print(f"太郎: {result['taro']}")
        print(f"  [{mark}] r_imit={result['r_imit']:.2f} R={result['R']:.2f} "
              f"delta={result['delta']:+.3f} streak={result['partial_streak']} "
              f"age={result['age']} stage={env.vocal_tract.stage}")

        try:
            smile_input = input("  笑顔(0.0〜1.0, Enter=0): ").strip()
            if smile_input:
                r_social = max(0.0, min(1.0, float(smile_input)))
                if r_social > 0:
                    env.step(parent_input, r_social=r_social)
                    print(f"  (笑顔 {r_social:.1f})")
        except ValueError:
            pass


if __name__ == "__main__":
    run_cli()
