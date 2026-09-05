# -*- coding: utf-8 -*-
"""並列に学習した複数モデルを1つに混ぜる（F2-25・並列化の検証）。

【方式】連合学習（FedAvg）型：同じ出発点から短く分かれた枝の重みを平均する。
LLMの分散学習（勾配平均）の粗い近似。分かれが短いうちは同じ「答えの谷」に
居るので平均が壊れない、という仮説を検証するための道具。

【急所＝語彙の位置合わせ】各枝は別の語を聞くため、音のトークン番号の割り振りが
枝ごとにズレる（例：「ゃ」が枝Aでは78番・枝Bでは80番）。番号のまま平均すると
別の音の行が混ざる。embedding と知覚ヘッドだけは**文字で照合**して合併語彙に
並べ直してから平均する。それ以外の重み（GRU・運動系・小脳NN等）は語彙に
依存しないので形が合えばそのまま平均。

混ぜ方の内訳：
  ・tensor（重み）      形が全枝で同じ → 単純平均／embedding・知覚ヘッド → 文字照合平均
  ・lexicon             counts=足し算、proto/view=重みつき平均（数学的に逐次と等価）
  ・hearing/brain語彙   文字の合併（union）
  ・発話小脳の帳面      辞書の合併（同じキーは先勝ち。枝間で内容はほぼ同一）

    .venv/Scripts/python.exe F/scripts/f_merge_models.py 出力.pt 入力1.pt 入力2.pt ...
"""
import sys
import copy

import torch

sys.stdout.reconfigure(encoding="utf-8")


def merge_vocab(vocabs):
    """char2idx の合併。最初の語彙の番号を保ち、無い文字を後ろへ足す。"""
    out = dict(vocabs[0])
    nxt = max(out.values()) + 1
    for v in vocabs[1:]:
        for ch in v:
            if ch not in out:
                out[ch] = nxt
                nxt += 1
    return out


def remap_rows(mat, char2idx_src, char2idx_dst, n_dst):
    """行列の行をトークン番号→文字→合併番号で並べ直す（無い行はゼロ＋マスク）。"""
    out = torch.zeros((n_dst,) + tuple(mat.shape[1:]), dtype=mat.dtype)
    mask = torch.zeros(n_dst)
    for ch, i in char2idx_src.items():
        if i < mat.shape[0]:
            j = char2idx_dst[ch]
            out[j] = mat[i]
            mask[j] = 1.0
    return out, mask


def main():
    out_path, in_paths = sys.argv[1], sys.argv[2:]
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in in_paths]
    n = len(blobs)
    print("混ぜる枝: %d本" % n)

    merged = copy.deepcopy(blobs[0])

    # ---- 語彙の合併 --------------------------------------------------------
    bv = merge_vocab([b["brain_vocab"]["char2idx"] for b in blobs])
    hv = merge_vocab([b["hearing_vocab"]["char2idx"] for b in blobs])
    n_bv = max(bv.values()) + 1
    merged["brain_vocab"] = {"char2idx": bv}
    # 保存形式の完全再現：idx2char を忘れると読み込みで KeyError（実測で踏んだ）
    merged["hearing_vocab"] = {"char2idx": hv,
                               "idx2char": {i: c for c, i in hv.items()},
                               "size": max(hv.values()) + 1}

    # ---- 脳の重み ----------------------------------------------------------
    VOCAB_KEYS = ("embedding.weight", "perception_head.weight", "perception_head.bias")
    brain = {}
    for k in blobs[0]["brain"].keys():
        vals = [b["brain"][k] for b in blobs]
        if k in VOCAB_KEYS:
            # 文字照合で合併語彙に並べ直してから、行ごとに「持っている枝」だけで平均
            acc = None
            cnt = None
            for b in blobs:
                m, mask = remap_rows(b["brain"][k], b["brain_vocab"]["char2idx"], bv, n_bv)
                acc = m if acc is None else acc + m
                cnt = mask if cnt is None else cnt + mask
            cnt = cnt.clamp(min=1.0)
            if acc.dim() == 2:
                brain[k] = acc / cnt.unsqueeze(1)
            else:
                brain[k] = acc / cnt
        elif all(v.shape == vals[0].shape for v in vals):
            brain[k] = sum(vals) / float(n)
        else:
            print("  形不一致（先勝ち）:", k)
            brain[k] = vals[0]
    merged["brain"] = brain

    # ---- その他のtensor群（運動・感覚・小脳NN） ------------------------------
    for key in ("fusion_insula", "fusion_proprio", "fusion_vestibular",
                "fusion_touch", "cereb"):
        if key not in blobs[0]:
            continue
        sd = {}
        for k in blobs[0][key].keys():
            vals = [b[key][k] for b in blobs if key in b]
            if all(hasattr(v, "shape") and v.shape == vals[0].shape for v in vals):
                sd[k] = sum(vals) / float(len(vals))
            else:
                sd[k] = vals[0]
        merged[key] = sd

    # ---- lexicon（数学的に逐次と等価な合併） --------------------------------
    if all("lexicon" in b for b in blobs):
        lx = merged["lexicon"]
        # counts: 足し算
        counts = {}
        for b in blobs:
            for k, v in b["lexicon"]["counts"].items():
                counts[k] = counts.get(k, 0) + v
        lx["counts"] = counts
        # proto: countsを重みにした平均（EMAの近似。枝間の完全等価ではない点に注意）
        proto = {}
        for b in blobs:
            for k, p in b["lexicon"].get("proto", {}).items():
                w = b["lexicon"]["counts"].get(k, 1)
                if k not in proto:
                    proto[k] = [w, [x * w for x in p]]
                else:
                    proto[k][0] += w
                    proto[k][1] = [a + x * w for a, x in zip(proto[k][1], p)]
        lx["proto"] = {k: [x / w for x in acc] for k, (w, acc) in proto.items()}
        # view（見慣れた景色の平均）: 重みつき平均
        vs = None
        vn = 0
        for b in blobs:
            v, m = b["lexicon"].get("view_sum"), b["lexicon"].get("view_n", 0)
            if v and m:
                vs = list(v) if vs is None else [a + x for a, x in zip(vs, v)]
                vn += m
        lx["view_sum"], lx["view_n"] = vs, vn
        if "channels" in lx and "vision" in lx.get("channels", {}):
            lx["channels"]["vision"]["proto"] = lx["proto"]
            lx["channels"]["vision"]["view_sum"] = vs
            lx["channels"]["vision"]["view_n"] = vn

    # ---- 発話小脳の帳面（合併・先勝ち） -------------------------------------
    if all("produce_cerebellum" in b for b in blobs):
        pc = {"forward_map": {}, "inverse_map": {}, "experience_count": {}}
        for b in blobs:
            for f in pc:
                for k, v in b["produce_cerebellum"].get(f, {}).items():
                    pc[f].setdefault(k, v)
        merged["produce_cerebellum"] = pc

    torch.save(merged, out_path)
    print("保存:", out_path, "／ 脳の語彙", n_bv, "音")


if __name__ == "__main__":
    main()
