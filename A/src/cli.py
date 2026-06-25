"""
CLI — 親（あなた）と太郎の対話インターフェース

コマンド（暫定・後で追加可能）：
  /save [tag]  — スナップショット保存
  /load        — スナップショット一覧＆復元
  /fork [path] — スナップショットからフォーク
  /stats       — 現在の統計
  /plot        — 学習曲線を生成
  /quit        — 終了
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from environment import TaroEnvironment


def print_header():
    print("=" * 56)
    print("  太郎 A1 — オウム返し試作")
    print("  あなたは太郎の「親」です。話しかけてください。")
    print("  太郎の返答のあと、笑顔の強さ(0.0〜1.0)を入力。")
    print("  コマンド: /save /load /fork /stats /plot /quit")
    print("=" * 56)
    print()


def run_cli():
    env = TaroEnvironment(run_name="cli_" + __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"))
    env.logger.save_run_info("CLI対話セッション", env.cfg)

    # Day1を保存
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

        # --- コマンド処理 ---
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
                        choice = input("番号を入力 (キャンセル: Enter): ").strip()
                        if choice:
                            idx = int(choice)
                            env.load(snaps[idx]["path"])
                            print(f"復元しました: {snaps[idx]['name']}")
                    except (ValueError, IndexError):
                        print("キャンセルしました。")

            elif cmd == "/fork":
                snaps = env.archive.list_snapshots()
                if not snaps:
                    print("スナップショットがありません。")
                else:
                    for i, s in enumerate(snaps):
                        print(f"  [{i}] {s['name']} (turn={s['turn']})")
                    try:
                        choice = input("フォーク元の番号: ").strip()
                        if choice:
                            idx = int(choice)
                            env.load(snaps[idx]["path"])
                            print(f"フォークしました: {snaps[idx]['name']} から再開")
                    except (ValueError, IndexError):
                        print("キャンセルしました。")

            elif cmd == "/stats":
                print(f"  ターン数: {env.clock.total_turns}")
                print(f"  発達年齢: {env.clock.age_str()}")
                print(f"  聞いた語数: {env.clock.total_tokens_heard}")
                print(f"  温度(tau): {env.brain.temperature:.4f}")
                print(f"  baseline: {env.dopamine.get_baseline():.4f}")
                print(f"  連続一致: {env.consecutive_matches}")
                print(f"  語彙数: {env.vocab.size}")

            elif cmd == "/plot":
                path = env.logger.plot_learning_curve()
                if path:
                    print(f"学習曲線を保存しました: {path}")
                else:
                    print("データが少なすぎます（2ターン以上必要）。")

            else:
                print(f"不明なコマンド: {cmd}")

            continue

        # --- 対話ループ ---
        result = env.step(parent_input, r_social=0.0)

        mark = "O" if result["exact_match"] else " "
        print(f"太郎: {result['taro']}")
        print(f"  [{mark}] r_imit={result['r_imit']:.2f} "
              f"r_pred={result['r_pred']:.2f} R={result['R']:.2f} "
              f"delta={result['delta']:+.3f} "
              f"streak={result['consecutive_matches']} "
              f"age={result['age']}")

        # 笑顔入力
        try:
            smile_input = input("  笑顔(0.0〜1.0, Enter=0): ").strip()
            if smile_input:
                r_social = float(smile_input)
                r_social = max(0.0, min(1.0, r_social))
                if r_social > 0:
                    result2 = env.step(parent_input, r_social=r_social)
                    print(f"  (笑顔 {r_social:.1f} を受けて学習しました)")
        except ValueError:
            pass


if __name__ == "__main__":
    run_cli()
