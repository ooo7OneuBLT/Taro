# -*- coding: utf-8 -*-
"""実験を、GPUの上限を超えないように機械的に順番へ並べて実行する。

【なぜ要るか、2026-08-02】
vision:true（画面描画あり）の学習を同時に何本まで動かせるかは、GPU1枚の
制約で決まる（検証の落とし穴チェックリスト 項91）。「同時3本まで」という
ルールを人が手で数えて守る運用にしていたが、同じ日のうちに
2回破った。1回目は8本を無警戒に同時起動、2回目は「3本を起動中に、
別の3本をさらに起動した」という単純な数え忘れ。

**手で数える限り、何度でも起きる。** このスクリプトは、いま何本動いているかを
自分で数えて、上限に達していたら次を待つ。人が数える必要を無くす。

【使い方】
    .venv/Scripts/python.exe run/tools/queue_train.py --max-concurrent 3 ^
        E/experiments/実験1.json E/experiments/実験2.json ...

    足りなければ実験ファイルをいくつでも渡してよい。上限を守りながら
    順に埋めて実行し、全部終わるまで待つ。

【落ちたときの扱い】
    1本が失敗（エラー終了）しても、残りは止めずに続ける。
    最後に「成功何本・失敗何本（どれが失敗したか）」をまとめて表示する。
    失敗の理由（GPUメモリ不足かどうか）はログファイルを見て自分で判断すること
    （このスクリプトは推測でラベルを付けない）。
"""
import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
_MAIN = os.path.join(_ROOT, "run", "main.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+", help="実験ファイル（JSON）のパス。いくつでも")
    ap.add_argument("--max-concurrent", type=int, default=3,
                     help="同時に動かす上限（既定3。検証の落とし穴チェックリスト項91の目安）")
    ap.add_argument("--log-dir", default=None,
                     help="各実験のログを書き出すフォルダ（既定：実験ファイルと同じ場所に .log）")
    a = ap.parse_args()

    残り = list(a.experiments)
    走行中 = []          # [(Popen, 実験パス, ログファイルハンドル)]
    結果 = {"成功": [], "失敗": []}

    print(f"[{time.strftime('%H:%M:%S')}] {len(残り)}本を、同時最大{a.max_concurrent}本で実行する")

    while 残り or 走行中:
        # 空きがあれば、上限まで詰める
        while 残り and len(走行中) < a.max_concurrent:
            exp = 残り.pop(0)
            log_path = (os.path.join(a.log_dir, os.path.basename(exp) + ".log")
                        if a.log_dir else exp + ".log")
            fp = open(log_path, "w", encoding="utf-8")
            p = subprocess.Popen([_PYTHON, _MAIN, exp], stdout=fp, stderr=subprocess.STDOUT,
                                  cwd=_ROOT)
            走行中.append((p, exp, fp))
            print(f"[{time.strftime('%H:%M:%S')}] 起動: {os.path.basename(exp)}"
                  f"（いま{len(走行中)}本・残り{len(残り)}本）")

        time.sleep(5)

        # 終わったものを回収する
        まだ走行中 = []
        for p, exp, fp in 走行中:
            ret = p.poll()
            if ret is None:
                まだ走行中.append((p, exp, fp))
                continue
            fp.close()
            if ret == 0:
                結果["成功"].append(exp)
                print(f"[{time.strftime('%H:%M:%S')}] 完了: {os.path.basename(exp)}")
            else:
                結果["失敗"].append(exp)
                print(f"[{time.strftime('%H:%M:%S')}] 失敗（終了コード{ret}）: "
                      f"{os.path.basename(exp)}　→ ログ: {exp}.log")
        走行中 = まだ走行中

    print(f"\n[{time.strftime('%H:%M:%S')}] 全部終了。"
          f"成功{len(結果['成功'])}本・失敗{len(結果['失敗'])}本")
    if 結果["失敗"]:
        print("失敗した実験（ログを確認すること）:")
        for exp in 結果["失敗"]:
            print(f"  {exp}")
    return 1 if 結果["失敗"] else 0


if __name__ == "__main__":
    sys.exit(main())
