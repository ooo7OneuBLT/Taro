# -*- coding: utf-8 -*-
"""並列学習の混ぜ方 v2（TIES方式）— F2-25の失敗と文献調査を受けた改良版。

【v1（単純平均）の失敗】枝ごとに違う語を学ぶ非IID条件では、平均で語固有の
学習が1/Nに薄まる（実測：既知と未知の差 0.105→0.049）。

【v2の方式＝TIES-Merging（Yadav et al., NeurIPS 2023・一次確認）＋差分ベース】
出発点(base)からの差分 delta_k = w_k - base に対して：
  ① 刈り込み(trim)     各枝で大きさ上位ρ%だけ残す（既定20%）
  ② 符号の多数決(elect) パラメータごとに残った差分の合計の符号を取る
  ③ 多数派の平均(mean)  符号が多数派に一致する枝だけで平均
  merged = base + λ × ③   （λ既定1.0・引数で変更可）
1枝しか触っていないパラメータ（その語専用の埋め込み行など）は、他の枝の差分が
刈り込みで消えるため**その枝の値がそのまま残る＝薄まらない**。これが本命の効能。

【語彙・表の差分化】counts等は base + Σ(枝 − base) ＝ 数学的に逐次と等価。
v1の「継承分の重複加算」バグ（280→1540回に膨張）もこれで直る。

    .venv/Scripts/python.exe F/scripts/f_merge_models_v2.py 出力.pt --base 出発点.pt \
        [--rho 0.2] [--lam 1.0] 枝1.pt 枝2.pt ...
"""
import sys
import copy
import argparse

import torch

sys.stdout.reconfigure(encoding="utf-8")


def merge_vocab(vocabs):
    out = dict(vocabs[0])
    nxt = max(out.values()) + 1
    for v in vocabs[1:]:
        for ch in v:
            if ch not in out:
                out[ch] = nxt
                nxt += 1
    return out


def remap_rows(mat, char2idx_src, char2idx_dst, n_dst, fill=None):
    """行をトークン番号→文字→合併番号で並べ直す。無い行は fill（無ければ0）。"""
    out = torch.zeros((n_dst,) + tuple(mat.shape[1:]), dtype=mat.dtype)
    if fill is not None:
        out[:fill.shape[0]] = fill[:n_dst]
    have = torch.zeros(n_dst, dtype=torch.bool)
    for ch, i in char2idx_src.items():
        if i < mat.shape[0]:
            j = char2idx_dst[ch]
            out[j] = mat[i]
            have[j] = True
    return out, have


def plain_mean(base_t, deltas, lam):
    return (base_t.float() + lam * sum(d.float() for d in deltas) / len(deltas)).to(base_t.dtype)


def ties(base_t, deltas, rho, lam):
    """TIES：trim → 符号の多数決 → 多数派平均。deltas: list of tensor（同形）。"""
    trimmed = []
    for d in deltas:
        flat = d.abs().flatten()
        if flat.numel() == 0:
            trimmed.append(d)
            continue
        k = max(1, int(flat.numel() * rho))
        thresh = flat.kthvalue(flat.numel() - k + 1).values
        trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
    total = sum(trimmed)
    sign = torch.sign(total)
    num = torch.zeros_like(base_t, dtype=torch.float32)
    cnt = torch.zeros_like(base_t, dtype=torch.float32)
    for d in trimmed:
        agree = (torch.sign(d) == sign) & (d != 0)
        num = num + torch.where(agree, d.float(), torch.zeros_like(d, dtype=torch.float32))
        cnt = cnt + agree.float()
    mean = num / cnt.clamp(min=1.0)
    return (base_t.float() + lam * mean).to(base_t.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("branches", nargs="+")
    ap.add_argument("--base", required=True)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--plain", action="store_true",
                    help="TIESでなく素の平均 base+mean(delta)。同期が頻繁なとき用")
    a = ap.parse_args()

    base = torch.load(a.base, map_location="cpu", weights_only=False)
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in a.branches]
    n = len(blobs)
    print("混ぜる枝 %d本 ／ TIES rho=%.2f lam=%.2f" % (n, a.rho, a.lam))
    merged = copy.deepcopy(blobs[0])

    # ---- 語彙の合併 --------------------------------------------------------
    bv = merge_vocab([b["brain_vocab"]["char2idx"] for b in blobs])
    hv = merge_vocab([b["hearing_vocab"]["char2idx"] for b in blobs])
    n_bv = max(bv.values()) + 1
    merged["brain_vocab"] = {"char2idx": bv}
    merged["hearing_vocab"] = {"char2idx": hv,
                               "idx2char": {i: c for c, i in hv.items()},
                               "size": max(hv.values()) + 1}
    base_bv = base.get("brain_vocab", {}).get("char2idx", {})

    # ---- 脳の重み（TIES・語彙キーは文字照合してから） ------------------------
    VOCAB_KEYS = ("embedding.weight", "perception_head.weight", "perception_head.bias")
    brain = {}
    for k in blobs[0]["brain"].keys():
        base_t = base["brain"].get(k)
        if k in VOCAB_KEYS:
            # base を合併語彙の形に並べ直す（baseに無い行＝新しい音は0起点）
            if base_t is not None and base_bv:
                base_al, _ = remap_rows(base_t, base_bv, bv, n_bv)
            else:
                base_al = torch.zeros((n_bv,) + tuple(blobs[0]["brain"][k].shape[1:]),
                                      dtype=blobs[0]["brain"][k].dtype)
            deltas = []
            for b in blobs:
                m, have = remap_rows(b["brain"][k], b["brain_vocab"]["char2idx"], bv, n_bv)
                d = m - base_al
                d[~have] = 0            # その枝が持たない行は差分ゼロ扱い
                deltas.append(d)
            brain[k] = (plain_mean(base_al, deltas, a.lam) if a.plain
                        else ties(base_al, deltas, a.rho, a.lam))
        else:
            vals = [b["brain"][k] for b in blobs]
            if base_t is not None and all(v.shape == base_t.shape for v in vals):
                brain[k] = (plain_mean(base_t, [v - base_t for v in vals], a.lam)
                            if a.plain else ties(base_t, [v - base_t for v in vals], a.rho, a.lam))
            else:
                brain[k] = vals[0]
    merged["brain"] = brain

    # ---- その他のtensor群もTIES --------------------------------------------
    for key in ("fusion_insula", "fusion_proprio", "fusion_vestibular",
                "fusion_touch", "cereb", "visual_projection"):
        if key not in blobs[0]:
            continue
        sd = {}
        for k in blobs[0][key].keys():
            vals = [b[key][k] for b in blobs if key in b]
            base_t = base.get(key, {}).get(k)
            if base_t is not None and all(
                    hasattr(v, "shape") and v.shape == base_t.shape for v in vals):
                sd[k] = (plain_mean(base_t, [v - base_t for v in vals], a.lam)
                         if a.plain else ties(base_t, [v - base_t for v in vals], a.rho, a.lam))
            else:
                sd[k] = vals[0]
        merged[key] = sd

    # ---- lexicon：差分の合計（逐次と等価・v1の重複加算バグ修正） --------------
    if all("lexicon" in b for b in blobs):
        lb = base.get("lexicon", {"counts": {}, "proto": {},
                                  "view_sum": None, "view_n": 0})
        lx = merged["lexicon"]
        counts = dict(lb.get("counts", {}))
        for b in blobs:
            for k, v in b["lexicon"]["counts"].items():
                counts[k] = counts.get(k, 0) + (v - lb.get("counts", {}).get(k, 0))
        lx["counts"] = counts
        proto = {k: [1, list(p)] for k, p in lb.get("proto", {}).items()}
        for b in blobs:
            for k, p in b["lexicon"].get("proto", {}).items():
                w = max(b["lexicon"]["counts"].get(k, 1)
                        - lb.get("counts", {}).get(k, 0), 0)
                if w == 0:
                    continue
                if k not in proto:
                    proto[k] = [w, [x * w for x in p]]
                else:
                    proto[k][0] += w
                    proto[k][1] = [aa + x * w for aa, x in zip(proto[k][1], [y * 1 for y in p])]
        lx["proto"] = {k: [x / w for x in acc] for k, (w, acc) in proto.items()}
        vs = list(lb.get("view_sum") or [])
        vn = lb.get("view_n", 0)
        for b in blobs:
            v, m = b["lexicon"].get("view_sum"), b["lexicon"].get("view_n", 0)
            if v and m:
                dv = [x - y for x, y in zip(v, lb.get("view_sum") or [0.0] * len(v))]
                vs = dv if not vs else [aa + x for aa, x in zip(vs, dv)]
                vn += m - lb.get("view_n", 0)
        lx["view_sum"], lx["view_n"] = (vs or None), vn
        if "channels" in lx and "vision" in lx.get("channels", {}):
            lx["channels"]["vision"]["proto"] = lx["proto"]
            lx["channels"]["vision"]["view_sum"] = lx["view_sum"]
            lx["channels"]["vision"]["view_n"] = vn

    # ---- 発話小脳の帳面（合併・先勝ち） -------------------------------------
    if all("produce_cerebellum" in b for b in blobs):
        pc = {"forward_map": {}, "inverse_map": {}, "experience_count": {}}
        for b in blobs:
            for f in pc:
                for k, v in b["produce_cerebellum"].get(f, {}).items():
                    pc[f].setdefault(k, v)
        merged["produce_cerebellum"] = pc

    torch.save(merged, a.out)
    print("保存:", a.out, "／ 脳の語彙", n_bv, "音 ／ 語彙counts合計",
          sum(merged["lexicon"]["counts"].values()) if "lexicon" in merged else "-")


if __name__ == "__main__":
    main()
