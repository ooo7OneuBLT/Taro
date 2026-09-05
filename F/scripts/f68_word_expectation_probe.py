# -*- coding: utf-8 -*-
"""「語が呼ぶ期待」で驚きを測る（2026-09-04夜・ユーザー発案）。

角度違いを予測する方式（F2-67c）は失敗した（既知0.209 vs 未知0.125、向きが逆転）。
ユーザーの言い換え：「コップだよと言われてボールを見せられたら驚く」＝
比べる相手は「同じ物の別角度」ではなく「その語が指すはずの見た目（連合器の記憶）」。

これは新しい仕組みを作らず、既存の語彙（lexicon.proto、各語の見た目プロトタイプ）を
そのまま使う。物の見た目が、①正しい語のプロトタイプ、②間違った語のプロトタイプ、
のどちらに近いかを比べるだけ。

    .venv/Scripts/python.exe F/scripts/f68_word_expectation_probe.py
出力: F/logs/F2-68_語の期待プローブ/図_正しい語 vs 間違った語.png
"""
import os, sys, io, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import f_gen_f49 as G
import f49_test as T
from senses.object_detector import patch_features, detect

OUT = "F/logs/F2-68_語の期待プローブ"


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a / na) @ (b / nb))


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    st = torch.load("F/models/F2-49c_r3_seed93_%s.pt" % G.DATE, map_location="cpu", weights_only=False)
    idx2char = st["hearing_vocab"]["idx2char"]

    def _decode(chunk):
        return "".join(idx2char.get(i, "?") for i in chunk)

    raw_protos = st["lexicon"]["proto"]
    known_words = [w for w, _, _ in G.WORDS]
    # 各語について、余計な語尾（だよ/だね等）が付かない一番短い一致チャンクを採用する
    protos = {}
    for w in known_words:
        w_norm = w.replace("ー", "")  # 長音は耳の入口で別表記になる（例：がおー→がおお）
        cands = [(chunk, v) for chunk, v in raw_protos.items() if _decode(chunk).startswith(w_norm)]
        if not cands:
            print("語→プロトタイプ: 見つからず", w)
            continue
        chunk, v = min(cands, key=lambda cv: len(cv[0]))
        protos[w] = np.asarray(v, dtype=np.float64)
        print("語→プロトタイプ:", w, "<-", _decode(chunk))
    words = sorted(protos.keys())
    print("語彙:", words)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval()

    stim = T.render_stimuli()
    rng = random.Random(0)
    match_surprisal, mismatch_surprisal = [], []
    per_word = {}
    n_skipped = 0
    for word, idx, unseen, yaw, img, name in stim:
        if word not in protos:
            n_skipped += 1
            continue
        p, n = patch_features(model, img)
        dets = detect(p, n, thresh=0.55)
        if not dets:
            n_skipped += 1
            continue
        v = np.asarray(dets[0]["appearance"], dtype=np.float64)
        correct = 1.0 - _cos(v, protos[word])
        wrong_word = rng.choice([w for w in words if w != word])
        wrong = 1.0 - _cos(v, protos[wrong_word])
        match_surprisal.append(correct)
        mismatch_surprisal.append(wrong)
        per_word.setdefault(word, {"correct": [], "wrong": []})
        per_word[word]["correct"].append(correct)
        per_word[word]["wrong"].append(wrong)

    print("\n語ごとの内訳：")
    print("%-6s %6s %12s %12s %8s" % ("語", "件数", "正しい語", "間違った語", "逆転数"))
    for w in sorted(per_word.keys()):
        c = np.array(per_word[w]["correct"]); m = np.array(per_word[w]["wrong"])
        n_bad = int(np.sum(c >= m))
        print("%-6s %6d %6.3f±%.3f %6.3f±%.3f %8d" % (w, len(c), c.mean(), c.std(), m.mean(), m.std(), n_bad))

    # 較正の参照点：プロトタイプ同士（語彙全体・全ペア）の似方の分布
    proto_list = list(protos.values())
    cross_proto_sims = [1.0 - _cos(proto_list[i], proto_list[j])
                         for i in range(len(proto_list)) for j in range(len(proto_list)) if i != j]
    print("\n参照点：語彙のプロトタイプ同士の驚き（全ペア %d 組） 平均%.3f±%.3f（min%.3f max%.3f）"
          % (len(cross_proto_sims), np.mean(cross_proto_sims), np.std(cross_proto_sims),
             np.min(cross_proto_sims), np.max(cross_proto_sims)))

    match_surprisal = np.array(match_surprisal); mismatch_surprisal = np.array(mismatch_surprisal)
    print("正しい語（例：コップと言われてコップを見る）：驚き 平均%.3f±%.3f（%d件、除外%d）"
          % (match_surprisal.mean(), match_surprisal.std(), len(match_surprisal), n_skipped))
    print("間違った語（例：コップと言われてボールを見る）：驚き 平均%.3f±%.3f（%d件）"
          % (mismatch_surprisal.mean(), mismatch_surprisal.std(), len(mismatch_surprisal)))
    gap = float(np.percentile(mismatch_surprisal, 5) - np.percentile(match_surprisal, 95))
    print("分布の隙間（5%%点と95%%点の差。正なら重ならない）：%.3f" % gap)
    acc = float(np.mean(match_surprisal < mismatch_surprisal))
    print("1件ずつの大小比較：正しい語の方が驚き小 %d/%d（%.1f%%）"
          % (int(acc * len(match_surprisal)), len(match_surprisal), acc * 100))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    groups_plot = [("正しい語\n(コップ→コップ)", match_surprisal, "#2b6cb0"),
                   ("間違った語\n(コップ→ボール)", mismatch_surprisal, "#c53030")]
    for i, (nm, dat, col) in enumerate(groups_plot):
        ax.scatter(np.full(len(dat), i) + np.random.RandomState(0).uniform(-.1, .1, len(dat)),
                   dat, alpha=.5, s=25, color=col)
        ax.plot([i - .22, i + .22], [np.mean(dat)] * 2, color="k", lw=2)
    ax.set_xticks(range(2)); ax.set_xticklabels([g[0] for g in groups_plot], fontsize=10)
    ax.set_ylabel("驚き（1 − 語のプロトタイプとの似方）"); ax.set_ylim(0, 1.0); ax.grid(alpha=.3, axis="y")
    ax.set_title("語が指す見た目と実際の見た目のズレは、正誤で分かれるか", fontsize=12)
    fig.tight_layout()
    p = OUT + "/図_正しい語 vs 間違った語.png"; fig.savefig(p, dpi=110); print("図:", p)


if __name__ == "__main__":
    main()
