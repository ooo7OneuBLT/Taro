# -*- coding: utf-8 -*-
"""F2-21：聞く学習（GRUの次トークン予測）つきで10語の学習をやり直す。

【F2-20との違いは produce の3キーだけ】シーン・XML・個体・ステップ数は F2-20 を
そのまま使う（変えたのは聞く学習だけ、という綺麗な比較にするため）。
  context=true       文脈（GRUの隠れ状態）を保持・話者の印つき
  listen_learn=true  聞いた発話で次トークン予測を学習（誤差逆伝搬・lr=0.005はB実績値）
  context_lambda=0   学習中は語の選択を変えない（λは テストで振る）

【テストは各個体 λ=0 と λ=0.3 の2本】測るもの：
  ① 正解率がF2-20（94.9%）から崩れていないか
  ② 保続：「直前と同じ語を言う率」が λ=0.3 > λ=0 になるか（合否判定）

    .venv/Scripts/python.exe F/scripts/f_gen_listen_learn.py
"""
import os
import sys
import io
import json

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

DATE = "2026-08-31"
OLD = "F2-20_10語"
TAG = "F2-21_聞く学習"
START = "F/models/F2-6_P1_喃語900s_12ヶ月_seed10_2026-08-24.pt"
# 発話ありのproduce設定の台（F2-15の産出テストと同じ値）＋聞く学習の3キー
PRODUCE_BASE = None   # 最初の学習実験から拾う


def main():
    chain = [l for l in io.open("F/logs/_f220_chain.txt", encoding="utf-8")
             .read().splitlines() if l.strip()]
    tests = [l for l in io.open("F/logs/_f220_tests.txt", encoding="utf-8")
             .read().splitlines() if l.strip()]
    # produce設定はF2-20テスト（産出あり）から借りる
    pd = json.load(io.open(tests[0], encoding="utf-8"))["taro"]["produce"]

    made, prev = [], START
    for p in chain:
        ex = json.load(io.open(p, encoding="utf-8"))
        name = ex["name"].replace(OLD, TAG)
        ex["name"] = name
        ex["note"] = ("2026-08-31 F2-21。F2-20と同一条件＋聞く学習"
                      "（context/listen_learn）。" + ex.get("note", ""))
        ex["taro"]["produce"] = dict(pd, context=True, listen_learn=True,
                                     context_lambda=0.0)
        ex["taro"]["model"] = prev
        ex["taro"]["save"] = ex["taro"]["save"].replace(OLD, TAG).replace(
            "2026-08-30", DATE)
        ex["run"]["csv"] = ex["run"]["csv"].replace(OLD, TAG)
        out = p.replace(OLD, TAG).replace("2026-08-30", DATE)
        io.open(out, "w", encoding="utf-8").write(json.dumps(ex, ensure_ascii=False, indent=2))
        made.append(out)
        prev = ex["taro"]["save"]
    print("学習 %d本（最後: %s）" % (len(made), os.path.basename(prev)))

    tmade = []
    for p in tests:
        for lam in (0.0, 0.3):
            ex = json.load(io.open(p, encoding="utf-8"))
            suffix = "_λ%02d" % int(lam * 10)
            name = ex["name"].replace(OLD, TAG) + suffix
            ex["name"] = name
            ex["note"] = ("2026-08-31 F2-21 テスト（λ=%.1f）。正解率と保続を測る。"
                          % lam)
            ex["taro"]["produce"] = dict(ex["taro"]["produce"], context=True,
                                         listen_learn=False, context_lambda=lam)
            ex["taro"]["model"] = prev
            ex["taro"]["save"] = ex["taro"]["save"].replace(OLD, TAG).replace(
                ".pt", suffix + ".pt")
            d = os.path.dirname(ex["run"]["csv"]).replace(OLD + "_テスト",
                                                          TAG + "_テスト") + suffix
            ex["run"]["csv"] = d + "/run.csv"
            ex["plugins"]["word_production"]["events_out"] = d + "/発話イベント.csv"
            ex["plugins"]["produce_snapshot"]["out_dir"] = d + "/中心窩"
            ex["plugins"]["produce_snapshot"]["full_dir"] = d + "/両目"
            os.makedirs(d, exist_ok=True)
            out = p.replace(OLD, TAG).replace("2026-08-30", DATE).replace(
                ".json", suffix + ".json")
            io.open(out, "w", encoding="utf-8").write(
                json.dumps(ex, ensure_ascii=False, indent=2))
            tmade.append(out)
    print("テスト %d本（各個体 λ=0 / λ=0.3）" % len(tmade))
    io.open("F/logs/_f221_chain.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(made) + "\n")
    io.open("F/logs/_f221_tests.txt", "w", encoding="utf-8",
            newline="\n").write("\n".join(tmade) + "\n")


if __name__ == "__main__":
    main()
