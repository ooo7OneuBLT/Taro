# -*- coding: utf-8 -*-
"""F2-78 GRUと海馬の静止測定（「GRUがおかしいのか」を切り分ける、2026-09-06）。

仕様：F/docs/二語文/仕様_f81_GRUと海馬の静止測定_2026-09-06.md

前半（日常語）で書いたとおり、モデルは走らせない。8語（くつ・こっぷ・おさら・おわん・
かばん・がおお・ばす・ぼおる）＋空の机の代理「ないね」の、語彙の表（lexicon.proto）に
入っている見た目の代表ベクトル（プロトタイプ）を入力として使い、GRU（本体の自力生成）と
海馬（記憶の想起）にそれぞれ答えさせて机上で確かめる。

生成の手順は run/trainer.py 760-800行（`gru_hippo` ブロック）をそのまま写した
（`key = visual_projection(vec)` → `forward_hidden([[<PARENT>]], hidden=H, prefix_vec=key)`
→ `<GONE>` を強制するときは続けて `forward_hidden([[<GONE>]], hidden)` →
先頭実文字のsoftmax最大＝確信度 → argmaxで最長8文字、EOS(2)で止める）。
文脈Hの作り方は run/trainer.py 336-380行（`_context_feed`）の「聞く学習を通らない
場合」の最終ブロック（424-428行）をそのまま写した（`[<PARENT>, <GONE>] + encode(text)`
を forward_hidden に通した最終hidden）。
海馬の想起は run/trainer.py 1191-1218行（`_hippo_recall_filtered`）をそのまま写した
（`language_hippocampus.py` は変更禁止のため複製）。

    .venv/Scripts/python.exe F/scripts/f81_gru_hippo_static_probe.py

出力: F/logs/F2-78b_M4_ひらがなテスト/静止測定/gru.csv
      F/logs/F2-78b_M4_ひらがなテスト/静止測定/hippo.csv
      F/logs/F2-78b_M4_ひらがなテスト/静止測定/hippo_episodes.csv
      F/logs/F2-78b_M4_ひらがなテスト/静止測定/結果.md
"""
import os
import sys
import io
import csv
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("F/scripts"))

import numpy as np
import torch

# 【手本・仕様書「後半」節】build_taro() は f71（f77と同じ流用）から import してそのまま使う。
#   import時にf49_test経由でsys.pathへcerebral_cortex等が足される副作用があるため
#   （f71ファイル本体のコメント参照）、このimportを先に済ませる。
import f71_pattern_generalization_disambiguation as f71
from hearing import normalize_kana

MODEL_PATH = "F/logs/F2-78_M4_ひらがな学習/model.pt"
OUT_DIR = "F/logs/F2-78b_M4_ひらがなテスト/静止測定"

WORDS = ["くつ", "こっぷ", "おさら", "おわん", "かばん", "がおお", "ばす", "ぼおる"]
EMPTY = "ないね"                 # 空の机の代理（「ないね」チャンクのプロトタイプ）
INPUTS = WORDS + [EMPTY]

C1_TEXT = "おわんないね"          # 文脈c1で「親が直前に言った」固定文（仕様書の指定どおり）
MAX_LENGTH = 8


# ============================================================================
# ① 語彙の表（lexicon.proto）から9種のプロトタイプベクトルを読む
#    （run/plugins/common/word_similarity_map.py の読み方に倣う：
#     {チャンク(タプル): 384次元} を hearing.vocab で文字列に戻す。ここでは
#     チェックポイントのblobを直接読み、同じ復号ロジックをその場で書く
#     ＝taro_setup.py 1098-1131行の復元手順そのものは動かさない、読み取り専用の近道）。
# ============================================================================
def load_prototypes(blob):
    hv = blob["hearing_vocab"]
    idx2char = {int(i): c for c, i in hv["char2idx"].items()}

    def decode_chunk(chunk):
        return "".join(idx2char.get(i, "?") for i in chunk
                        if idx2char.get(i, "?") not in ("<PAD>", "<BOS>", "<EOS>"))

    proto = blob["lexicon"]["channels"]["vision"]["proto"]
    word_to_vec = {}
    for chunk, v in proto.items():
        s = decode_chunk(chunk)
        if s not in word_to_vec:
            word_to_vec[s] = np.asarray(v, dtype=np.float32)
    missing = [w for w in INPUTS if w not in word_to_vec]
    if missing:
        raise ValueError(f"語彙の表(lexicon.proto)に無い語: {missing}"
                          f"（見つかった語: {sorted(word_to_vec)}）")
    return {w: word_to_vec[w] for w in INPUTS}


# ============================================================================
# ② GRU・海馬の生成手順（run/trainer.py 760-800行・336-380行・1191-1218行を写す）
# ============================================================================
def project_key(t, vec384):
    """視覚ベクトル(384次元)を64次元の鍵に変換する（trainer.py 775-777行と同じ）。"""
    vin = torch.tensor(list(vec384), dtype=torch.float32)
    with torch.no_grad():
        return t._visual_projection(vin)


def build_context_h(t, key, gone, text):
    """文脈c1のH：[<PARENT>](, <GONE>) + encode(text) を forward_hidden に通した
    最終hidden（trainer.py _context_feed 424-428行「聞く学習を通らない場合」の
    最終ブロックと同じ手順。prefix_vecは呼び出し元と同じkeyを使う＝仕様書の指定）。
    """
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    ids = [par] + pv.encode(normalize_kana(text))
    if gone and getattr(t, "_gone_id", None) is not None:
        ids = [ids[0], t._gone_id] + ids[1:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, h = t.brain.forward_hidden(x, hidden=None, prefix_vec=key)
    return h.detach()


def generate(t, key, hidden, force_gone, max_length=MAX_LENGTH):
    """trainer.py 776-799行をそのまま写した生成（本体＝GRU自力）。

    戻り値: (word, conf, top3) 。top3は先頭実文字のsoftmax上位3つ [(文字, 確率), ...]。
    """
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    gone_id = getattr(t, "_gone_id", None)
    with torch.no_grad():
        out, hh = t.brain.forward_hidden(
            torch.tensor([[par]], dtype=torch.long), hidden=hidden, prefix_vec=key)
        logits = t.brain.perception_head(out)[0, -1]
        if force_gone and gone_id is not None:
            out, hh = t.brain.forward_hidden(
                torch.tensor([[gone_id]], dtype=torch.long), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        conf = float(probs.max())
        top3_idx = torch.topk(probs, min(3, probs.shape[-1])).indices.tolist()
        top3 = [(pv.idx2char.get(i, "?"), round(float(probs[i]), 4)) for i in top3_idx]
        seq = []
        for _ in range(max_length):
            tk = int(torch.argmax(logits))
            if tk == 2:
                break
            seq.append(tk)
            out, hh = t.brain.forward_hidden(
                torch.tensor([[tk]], dtype=torch.long), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
    # 【trainer.py 800-803行と同じ】<GONE>を強制した分だけでなく、自力生成の
    #   途中で<GONE>を自発的に選んだ場合もデコードから除く（採点用にwordは
    #   常に見た目の文字列だけにする。「言った/言わなかった」自体はforce_goneと
    #   別に生成結果に<GONE>が混ざるかで見える＝表3の変化行で拾える）。
    seq_clean = strip_gone(seq, gone_id)
    word = pv.decode(seq_clean) if seq_clean else ""
    return word, conf, top3


def hippo_recall_filtered(hip, key_vis, want_gone, gone_id):
    """run/trainer.py 1191-1218行 `_hippo_recall_filtered` の複製（1文字も変えない。
    language_hippocampus.py 自体は変更禁止のためここに複製する、との仕様書の指示どおり）。
    """
    eps = [ep for ep in hip.episodes
           if bool(ep["tokens"]) and (ep["tokens"][0] == gone_id) == bool(want_gone)]
    if not eps:
        return [], 0.0, 0.0
    q = np.asarray(key_vis, dtype=np.float32)
    qn = np.linalg.norm(q) + 1e-8
    best, best_sim = None, -1e9
    for ep in eps:
        k = ep["key_vis"]
        sim = float(np.dot(q, k) / (qn * (np.linalg.norm(k) + 1e-8)))
        if sim > best_sim:
            best_sim, best = sim, ep
    return list(best["tokens"]), best_sim, float(best["strength"])


def top3_to_str(top3):
    return ",".join(f"{ch}:{p:.4f}" for ch, p in top3)


def strip_gone(tokens, gone_id):
    if gone_id is None:
        return tokens
    return [tk for tk in tokens if tk != gone_id]


# ============================================================================
# ③ 机上確認（検証・止まる条件節）：入力「おわん」・GONE無・c0 で
#    「おわんだね」または「おわんだよ」が出ることを最初に確かめる
# ============================================================================
def sanity_check(t, protos):
    key = project_key(t, protos["おわん"])
    word, conf, top3 = generate(t, key, hidden=None, force_gone=False)
    ok = word in ("おわんだね", "おわんだよ")
    print(f"[机上確認] 入力=おわん GONE無 c0 → 「{word}」 conf={conf:.4f} "
          f"top3={top3_to_str(top3)}  判定={'OK' if ok else 'NG'}", flush=True)
    if not ok:
        raise RuntimeError(
            f"机上確認に失敗：入力「おわん」・GONE無・c0で「おわんだね/だよ」が"
            f"出なかった（実際「{word}」conf={conf:.4f}）。全体を回す前に止める。")


# ============================================================================
# ④ 本走行
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1] モデル復元:", MODEL_PATH, flush=True)
    t = f71.build_taro(MODEL_PATH)
    blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    gone_id = t._gone_id = t.produce_vocab.char2idx["<GONE>"]
    hip = t.language_hippocampus
    print(f"  <GONE> id={gone_id}  海馬エピソード数={len(hip.episodes)}", flush=True)

    print("[2] 語彙の表からプロトタイプを読む", flush=True)
    protos = load_prototypes(blob)
    for w in INPUTS:
        print(f"  {w}: ノルム={float(np.linalg.norm(protos[w])):.4f}", flush=True)

    print("[3] 机上確認（先頭）", flush=True)
    sanity_check(t, protos)

    # ---- gru.csv：入力9種 × GONE有無2 × 文脈c0/c1 = 36行 ----
    print("[4] gru.csv 生成", flush=True)
    gru_rows = []
    keys = {w: project_key(t, protos[w]) for w in INPUTS}
    for w in INPUTS:
        key = keys[w]
        h_c1 = build_context_h(t, key, gone=True, text=C1_TEXT)
        for gone in (False, True):
            for ctx_name, hidden in (("c0", None), ("c1", h_c1)):
                word, conf, top3 = generate(t, key, hidden, force_gone=gone)
                gru_rows.append({
                    "input": w, "gone": int(gone), "context": ctx_name,
                    "word": word, "conf": round(conf, 5),
                    "top3_first_tokens": top3_to_str(top3),
                })
    gru_csv = os.path.join(OUT_DIR, "gru.csv")
    with io.open(gru_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "input", "gone", "context", "word", "conf", "top3_first_tokens"])
        w_.writeheader()
        w_.writerows(gru_rows)
    print("  ->", gru_csv, f"({len(gru_rows)}行)", flush=True)

    # ---- hippo.csv：入力9種 × want_gone(真/偽) = 18行 ----
    print("[5] hippo.csv 生成", flush=True)
    hippo_rows = []
    for w in INPUTS:
        key_np = keys[w].detach().cpu().numpy()
        for want_gone in (True, False):
            toks, sim, strength = hippo_recall_filtered(hip, key_np, want_gone, gone_id)
            toks_clean = strip_gone(toks, gone_id)
            word = t.produce_vocab.decode(toks_clean) if toks_clean else ""
            conf = sim * strength
            hippo_rows.append({
                "input": w, "want_gone": int(want_gone), "word": word,
                "sim": round(sim, 5), "strength": round(strength, 5),
                "conf": round(conf, 5),
            })
    hippo_csv = os.path.join(OUT_DIR, "hippo.csv")
    with io.open(hippo_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "input", "want_gone", "word", "sim", "strength", "conf"])
        w_.writeheader()
        w_.writerows(hippo_rows)
    print("  ->", hippo_csv, f"({len(hippo_rows)}行)", flush=True)

    # ---- hippo_episodes.csv：海馬の全エピソード ----
    print("[6] hippo_episodes.csv 生成", flush=True)
    # 9種プロトタイプをvisual_projectionに通した64次元（最近傍探索用）
    proto_keys64 = {w: keys[w].detach().cpu().numpy() for w in INPUTS}
    episode_rows = []
    for idx, ep in enumerate(hip.episodes):
        is_gone = bool(ep["tokens"]) and ep["tokens"][0] == gone_id
        toks_clean = strip_gone(ep["tokens"], gone_id)
        word = t.produce_vocab.decode(toks_clean) if toks_clean else ""
        k = np.asarray(ep["key_vis"], dtype=np.float32)
        kn = np.linalg.norm(k) + 1e-8
        best_w, best_cos = None, -1e9
        for w in INPUTS:
            pk = proto_keys64[w]
            cos = float(np.dot(k, pk) / (kn * (np.linalg.norm(pk) + 1e-8)))
            if cos > best_cos:
                best_w, best_cos = w, cos
        episode_rows.append({
            "idx": idx, "gone": int(is_gone), "word": word,
            "strength": round(float(ep["strength"]), 5),
            "written_at": int(ep["written_at"]),
            "nearest_word": best_w, "nearest_cos": round(best_cos, 5),
        })
    ep_csv = os.path.join(OUT_DIR, "hippo_episodes.csv")
    with io.open(ep_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "idx", "gone", "word", "strength", "written_at",
            "nearest_word", "nearest_cos"])
        w_.writeheader()
        w_.writerows(episode_rows)
    print("  ->", ep_csv, f"({len(episode_rows)}行)", flush=True)

    # ========================================================================
    # ⑤ 結果.md（要点のみ）
    # ========================================================================
    print("[7] 結果.md 集計", flush=True)

    def find_row(input_, gone, context):
        for r in gru_rows:
            if r["input"] == input_ and r["gone"] == int(gone) and r["context"] == context:
                return r
        return None

    # 表1：8語 x GONE有 x c0 で「語+ないね」を言えた数
    table1_lines = []
    n1_ok = 0
    for w in WORDS:
        r = find_row(w, True, "c0")
        want = w + "ないね"
        ok = (r["word"] == want)
        n1_ok += int(ok)
        table1_lines.append(f"| {w} | {r['word']} | {r['conf']:.4f} | {'OK' if ok else '-'} |")

    # 表2：8語 x GONE無 x c0 で「語+だね/だよ」を言えた数
    table2_lines = []
    n2_ok = 0
    for w in WORDS:
        r = find_row(w, False, "c0")
        ok = r["word"] in (w + "だね", w + "だよ")
        n2_ok += int(ok)
        table2_lines.append(f"| {w} | {r['word']} | {r['conf']:.4f} | {'OK' if ok else '-'} |")

    # 表3：文脈c1を入れると表1・表2がどう変わるか（変化した行だけ）
    table3_lines = []
    for w in WORDS:
        for gone, want_fn in ((True, lambda w_: w_ + "ないね"),
                               (False, lambda w_: None)):
            r0 = find_row(w, gone, "c0")
            r1 = find_row(w, gone, "c1")
            if r0["word"] != r1["word"]:
                table3_lines.append(
                    f"| {w} | {'有' if gone else '無'} | {r0['word']}(conf={r0['conf']:.4f}) "
                    f"| {r1['word']}(conf={r1['conf']:.4f}) |")

    # 表4：「ないね」プロトタイプ（空の机）を入れたときのGRUの答え4通り
    table4_lines = []
    for gone in (False, True):
        for ctx in ("c0", "c1"):
            r = find_row(EMPTY, gone, ctx)
            table4_lines.append(
                f"| GONE{'有' if gone else '無'} | {ctx} | {r['word']} | {r['conf']:.4f} | "
                f"{r['top3_first_tokens']} |")

    # 表5：海馬<GONE>エピソードの件数・鍵の内訳・strengthの分布
    gone_eps = [r for r in episode_rows if r["gone"] == 1]
    n_gone = len(gone_eps)
    n_empty_desk_key = sum(1 for r in gone_eps if r["nearest_word"] == EMPTY)
    n_noun_key = n_gone - n_empty_desk_key
    strength_counter = Counter(round(r["strength"], 2) for r in gone_eps)
    table5_strength_lines = [f"| {s} | {c} |" for s, c in
                              sorted(strength_counter.items(), reverse=True)]
    pct_empty_desk = (100.0 * n_empty_desk_key / n_gone) if n_gone else 0.0

    judge1 = f"GRU は正しい見た目＋GONEで {n1_ok}/8 " + ("言えた" if n1_ok == 8 else "言えなかった")
    judge2 = f"海馬の GONE 記憶の鍵は {pct_empty_desk:.1f}% が空の机" \
             f"（{n_empty_desk_key}/{n_gone}件。名詞寄り{n_noun_key}件）" if n_gone else \
             "海馬の GONE 記憶の鍵は 判定不能（<GONE>エピソードが0件）"

    md = []
    md.append("# f81 GRUと海馬の静止測定：結果\n")
    md.append(f"モデル: `{MODEL_PATH}`\n")
    md.append("## 表1：8語 x GONE有 x c0 で GRU が「語＋ないね」を言えた数"
               f"（{n1_ok}/8）\n")
    md.append("| 語 | word | conf | 判定 |")
    md.append("|---|---|---|---|")
    md.extend(table1_lines)
    md.append("")
    md.append(f"## 表2：8語 x GONE無 x c0 で「語＋だね/だよ」を言えた数（{n2_ok}/8）\n")
    md.append("| 語 | word | conf | 判定 |")
    md.append("|---|---|---|---|")
    md.extend(table2_lines)
    md.append("")
    md.append("## 表3：文脈c1を入れると表1・表2がどう変わるか（変化した行だけ）\n")
    if table3_lines:
        md.append("| 語 | GONE | c0の答え | c1の答え |")
        md.append("|---|---|---|---|")
        md.extend(table3_lines)
    else:
        md.append("変化した行なし。")
    md.append("")
    md.append("## 表4：「ないね」プロトタイプ（空の机）を入れたときのGRUの答え4通り\n")
    md.append("| GONE | 文脈 | word | conf | top3先頭文字 |")
    md.append("|---|---|---|---|---|")
    md.extend(table4_lines)
    md.append("")
    md.append(f"## 表5：海馬 <GONE> エピソードの件数・鍵の内訳・strengthの分布"
               f"（全<GONE>件数={n_gone}）\n")
    md.append(f"- 鍵の内訳：空の机（ないね）寄り {n_empty_desk_key}件 "
               f"({pct_empty_desk:.1f}%) / 名詞寄り {n_noun_key}件\n")
    md.append("| strength | 件数 |")
    md.append("|---|---|")
    md.extend(table5_strength_lines if table5_strength_lines else ["| (該当なし) | 0 |"])
    md.append("")
    md.append("## 判定\n")
    md.append(f"- {judge1}")
    md.append(f"- {judge2}")
    md.append("")

    md_path = os.path.join(OUT_DIR, "結果.md")
    with io.open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))
    print("  ->", md_path, flush=True)
    print("\n".join(md))
    print("[OK] 完了", flush=True)


if __name__ == "__main__":
    main()
