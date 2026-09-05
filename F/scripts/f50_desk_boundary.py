# -*- coding: utf-8 -*-
"""机上確認（2026-09-03・分節の第2案）：「沈黙（発話の終わり）を教師にすると、語の切れ目に
「ここで終わる確率」の山ができるか」を、走行なしで確かめる。

材料：F2-49 の最終モデル（r8）と、鎖8本で親が実際に言ったセリフ（発話イベント.csv・約900回）。
手順：①今の脳で各位置の p(終わり) を読む（対照）②同じセリフを末尾に <EOS> を付けて聞く学習と
同じ計算（Adam・lr=listen_lr・交差エントロピー）で1回ずつ聞かせ直す ③もう一度 p(終わり) を読む。
判定：語の内側の真の切れ目（「おわん|だよ」の | ）で p(終わり) が、それ以外の位置より高いか。
切る規準は Christiansen et al. 1998 と同じ「その発話の平均より高い位置」（固定閾値なし）。
本走行との差：文脈hiddenの持ち越し無し・視覚トークン無し（机上の簡略）。

    .venv/Scripts/python.exe F/scripts/f50_desk_boundary.py
出力: F/logs/F2-50_分節/机上_境界確率.png・机上_境界確率.txt
"""
import os, sys, io, csv, glob, warnings, itertools
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src")); sys.path.insert(0, os.path.abspath("taro_core/src/brain")); sys.path.insert(0, os.path.abspath("taro_core/src/senses")); sys.path.insert(0, os.path.abspath("F/scripts"))
import numpy as np, torch, torch.nn.functional as F
import f_gen_f49 as G
from cerebral_cortex.recurrent_core import TaroBrain
from hearing import Vocabulary, expand_long_vowel

MODEL = "F/models/F2-49_r8_seed98_2026-09-03.pt"
LR = 0.001            # 本走行の listen_lr
PASSES = (1, 3)       # 聞かせ直しの周回数（1周＝本走行と同じ回数、3周＝量の効果）
OUT = "F/logs/F2-50_分節"
EOS, PARENT_NAME = 2, "<PARENT>"
PROBES = ["おわんだよ", "がおーいるね", "くつだね", "コップだよ", "かばんだね", "バス"]


def load():
    blob = torch.load(MODEL, map_location="cpu", weights_only=False)
    pv = Vocabulary(); pv.char2idx = dict(blob["brain_vocab"]["char2idx"])
    pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}; pv.size = max(pv.idx2char) + 1
    b = TaroBrain(vocab_size=3); b.resize_embedding(blob["brain"]["embedding.weight"].shape[0])
    b.load_state_dict({k: v for k, v in blob["brain"].items()
                       if k in b.state_dict() and b.state_dict()[k].shape == v.shape}, strict=False)
    return b, pv


def ids_of(pv, text, cap):
    return [pv.char2idx[c] for c in expand_long_vowel(text) if c in pv.char2idx and pv.char2idx[c] < cap]


def p_end(b, seq):
    """seq=[PARENT]+tokens。位置 i（トークン i を聞き終えた直後）の p(EOS)。"""
    with torch.no_grad():
        logits, _ = b.forward_perception(torch.tensor([seq], dtype=torch.long))
        return F.softmax(logits[0], dim=-1)[:, EOS].numpy()


def true_boundary(pv, text, cap):
    """発話の内側の真の切れ目（語の直後）のトークン位置（PARENT を 0 とする列での index）。無ければ None。"""
    for w, _, _ in G.WORDS:
        if text.startswith(w) and text != w:
            return len(ids_of(pv, w, cap))          # PARENT=index0, 語の最後のトークン=index len(w)
    return None


def evaluate(b, pv, utts, cap):
    """全発話で、真の切れ目の p(EOS) と、それ以外（末尾を除く）の p(EOS) の平均。Christiansen規準の適合率/再現率。"""
    at_b, at_o, tp, fp, fn = [], [], 0, 0, 0
    for text in utts:
        toks = ids_of(pv, text, cap)
        if len(toks) < 2:
            continue
        seq = [pv.char2idx[PARENT_NAME]] + toks
        p = p_end(b, seq)
        bi = true_boundary(pv, text, cap)
        inner = p[1:len(seq) - 1]                     # 各トークン直後、最後のトークン直後は除く
        if bi is None:
            continue
        at_b.append(p[bi]); at_o.extend(np.delete(inner, bi - 1))
        cut = set(int(i) + 1 for i in np.where(inner > inner.mean())[0])
        tp += int(bi in cut); fp += len(cut - {bi}); fn += int(bi not in cut)
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return float(np.mean(at_b)), float(np.mean(at_o)), prec, rec


def relisten(b, pv, utts, cap, passes):
    params = itertools.chain(b.embedding.parameters(), b.gru.parameters(), b.perception_head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    par = pv.char2idx[PARENT_NAME]
    for _ in range(passes):
        for text in utts:
            ids = [par] + ids_of(pv, text, cap) + [EOS]
            if len(ids) < 3:
                continue
            xin = torch.tensor([ids[:-1]]); tgt = torch.tensor([ids[1:]])
            out, _ = b.forward_hidden(xin)
            loss = F.cross_entropy(b.perception_head(out)[0], tgt[0])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()


def main():
    os.makedirs(OUT, exist_ok=True)
    utts = []
    for r in range(1, 9):
        for row in csv.DictReader(io.open("F/logs/F2-49_実物スキャン/r%d/発話イベント.csv" % r, encoding="utf-8")):
            utts.append(row["text"])
    b, pv = load(); cap = b.embedding.num_embeddings
    print("親のセリフ", len(utts), "回")
    conds = []
    def snapshot(label):
        mb, mo, pr, rc = evaluate(b, pv, utts, cap)
        rows = {}
        for t in PROBES:
            seq = [pv.char2idx[PARENT_NAME]] + ids_of(pv, t, cap)
            rows[t] = p_end(b, seq)[1:]               # 各トークン直後
        conds.append((label, mb, mo, pr, rc, rows))
        print("%-14s 切れ目のp(終)=%.3f  それ以外=%.3f  適合率=%.2f 再現率=%.2f" % (label, mb, mo, pr, rc))
    snapshot("聞かせ直し前")
    done = 0
    for n in PASSES:
        relisten(b, pv, utts, cap, n - done); done = n
        snapshot("%d周" % n)
    # 図
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams
    for c in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if any(c in f.name for f in font_manager.fontManager.ttflist):
            rcParams["font.family"] = c; break
    fig, axes = plt.subplots(len(PROBES), len(conds), figsize=(3.6 * len(conds), 1.9 * len(PROBES)), squeeze=False)
    for j, (label, mb, mo, pr, rc, rows) in enumerate(conds):
        for i, t in enumerate(PROBES):
            ax = axes[i][j]; chars = list(expand_long_vowel(t)); p = rows[t]
            bi = true_boundary(pv, t, cap)
            cols = ["tab:red" if (bi is not None and k + 1 == bi) else "tab:blue" for k in range(len(p))]
            ax.bar(range(len(p)), p, color=cols); ax.set_xticks(range(len(p))); ax.set_xticklabels(chars[:len(p)])
            ax.set_ylim(0, 1); ax.axhline(p.mean(), color="gray", ls=":", lw=1)
            if i == 0: ax.set_title("%s\n切れ目%.2f／他%.2f 適合%.2f 再現%.2f" % (label, mb, mo, pr, rc), fontsize=9)
            if j == 0: ax.set_ylabel("p(終わり)", fontsize=8)
    fig.suptitle("机上確認：各音を聞いた直後の「ここで終わる確率」（赤＝語の直後の真の切れ目、点線＝その発話の平均）", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "/机上_境界確率.png", dpi=110)
    io.open(OUT + "/机上_境界確率.txt", "w", encoding="utf-8").write(
        "\n".join("%s 切れ目%.3f 他%.3f 適合率%.2f 再現率%.2f" % c[:5] for c in conds) + "\n")
    print("図:", OUT + "/机上_境界確率.png")


if __name__ == "__main__":
    main()
