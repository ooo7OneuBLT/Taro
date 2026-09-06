# -*- coding: utf-8 -*-
"""F2-81 あるの印・消えたの印の静止測定（f81の複製、2026-09-06）。

仕様：F/docs/二語文/仕様_M4d_あるの印_2026-09-06.md 後半「4. 静止測定 f83」節。

f81（F/scripts/f81_gru_hippo_static_probe.py）をそのまま複製し、印を
「なし／あり(<HERE>)／消えた(<GONE>)」の3通りに広げ、文脈c1を
「自分が直前に『おわんないね』」（話者<SELF>・<GONE>付き）に変更したもの。
モデルは走らせない。8語＋空の机の代理「ないね」の、語彙の表(lexicon.proto)に
入っている見た目の代表ベクトル（プロトタイプ）を入力として使い、GRU（本体の
自力生成）と海馬（記憶の想起）にそれぞれ答えさせて机上で確かめる。

生成の手順は run/trainer.py の gru_hippo ブロック（785-855行付近）をそのまま
写した（f81との違いは、開始トークン直後に強制するのが<GONE>だけでなく<HERE>も
あること、海馬の絞り込みが3値(want="gone"|"here"|"none")になったこと）。

    .venv/Scripts/python.exe F/scripts/f83_here_gone_static_probe.py [モデルパス]

既定モデルパス: F/logs/F2-81_M4d_あるの印学習/model.pt
出力: F/logs/F2-81_M4d_あるの印学習/静止測定/gru.csv
      F/logs/F2-81_M4d_あるの印学習/静止測定/hippo.csv
      F/logs/F2-81_M4d_あるの印学習/静止測定/hippo_episodes.csv
      F/logs/F2-81_M4d_あるの印学習/静止測定/結果.md
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

# 【手本・f81と同じ】build_taro() は f71 から import してそのまま使う。
import f71_pattern_generalization_disambiguation as f71
from hearing import normalize_kana

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "F/logs/F2-81_M4d_あるの印学習/model.pt"
OUT_DIR = "F/logs/F2-81_M4d_あるの印学習/静止測定"

WORDS = ["くつ", "こっぷ", "おさら", "おわん", "かばん", "がおお", "ばす", "ぼおる"]
EMPTY = "ないね"                 # 空の机の代理（「ないね」チャンクのプロトタイプ）
INPUTS = WORDS + [EMPTY]

# 【M4d・仕様書「4. 静止測定」節】文脈c1は「自分が直前に『おわんないね』」
#   （<SELF>話者・<GONE>付き）に変更。build_context_h の speaker 引数で切り替える。
C1_TEXT = "おわんないね"
C1_SPEAKER = "<SELF>"
MAX_LENGTH = 8

MARKS = ("none", "here", "gone")   # 印3種：なし／あり／消えた


# ============================================================================
# ① 語彙の表（lexicon.proto）から9種のプロトタイプベクトルを読む（f81と同一）
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
# ② GRU・海馬の生成手順（run/trainer.py の gru_hippo ブロックを写す）
# ============================================================================
def project_key(t, vec384):
    """視覚ベクトル(384次元)を64次元の鍵に変換する（trainer.py と同じ）。"""
    vin = torch.tensor(list(vec384), dtype=torch.float32)
    with torch.no_grad():
        return t._visual_projection(vin)


def build_context_h(t, key, mark, speaker, text):
    """文脈c1のH：[<話者>](, <GONE>|<HERE>) + encode(text) を forward_hidden に
    通した最終hidden（trainer.py _context_feed の最終ブロックと同じ手順。
    prefix_vecは呼び出し元と同じkeyを使う）。

    mark: "none"|"here"|"gone"。speaker は "<PARENT>" か "<SELF>"。
    """
    pv = t.produce_vocab
    sp = pv.char2idx[speaker]
    ids = [sp] + pv.encode(normalize_kana(text))
    if mark == "gone" and getattr(t, "_gone_id", None) is not None:
        ids = [ids[0], t._gone_id] + ids[1:]
    elif mark == "here" and getattr(t, "_here_id", None) is not None:
        ids = [ids[0], t._here_id] + ids[1:]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, h = t.brain.forward_hidden(x, hidden=None, prefix_vec=key)
    return h.detach()


def generate(t, key, hidden, force_mark, max_length=MAX_LENGTH):
    """trainer.py の gru_hippo ブロックをそのまま写した生成（本体＝GRU自力）。

    force_mark: "none"|"here"|"gone"。開始トークン(<PARENT>)の直後に、
    force_mark=="gone"なら<GONE>を、force_mark=="here"なら<HERE>を強制する
    （trainer.py の「_gone_now なら<GONE>、_here_now なら<HERE>」分岐と同じ）。

    戻り値: (word, conf, top3) 。top3は先頭実文字のsoftmax上位3つ [(文字, 確率), ...]。
    """
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    gone_id = getattr(t, "_gone_id", None)
    here_id = getattr(t, "_here_id", None)
    with torch.no_grad():
        out, hh = t.brain.forward_hidden(
            torch.tensor([[par]], dtype=torch.long), hidden=hidden, prefix_vec=key)
        logits = t.brain.perception_head(out)[0, -1]
        if force_mark == "gone" and gone_id is not None:
            out, hh = t.brain.forward_hidden(
                torch.tensor([[gone_id]], dtype=torch.long), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
        elif force_mark == "here" and here_id is not None:
            out, hh = t.brain.forward_hidden(
                torch.tensor([[here_id]], dtype=torch.long), hidden=hh)
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
    # 【trainer.py と同じ】<GONE>・<HERE>を強制した分だけでなく、自力生成の
    #   途中で自発的に選んだ場合もデコードから除く（採点用にwordは常に見た目の
    #   文字列だけにする）。
    seq_clean = strip_marks(seq, gone_id, here_id)
    word = pv.decode(seq_clean) if seq_clean else ""
    return word, conf, top3


def hippo_recall_filtered(hip, key_vis, want, gone_id, here_id):
    """run/trainer.py `_hippo_recall_filtered`（M4d・3値版）の複製（1文字も
    変えない。language_hippocampus.py 自体は変更禁止のためここに複製する）。
    """
    def cat(tok0):
        if gone_id is not None and tok0 == gone_id:
            return "gone"
        if here_id is not None and tok0 == here_id:
            return "here"
        return "none"

    eps = [ep for ep in hip.episodes
           if bool(ep["tokens"]) and cat(ep["tokens"][0]) == want]
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


def strip_marks(tokens, gone_id, here_id):
    return [tk for tk in tokens if tk != gone_id and tk != here_id]


# ============================================================================
# ③ 机上確認（検証・止まる条件節）：入力「おわん」・印なし・c0 で
#    「おわんだね」または「おわんだよ」が出ることを最初に確かめる
# ============================================================================
def sanity_check(t, protos):
    key = project_key(t, protos["おわん"])
    word, conf, top3 = generate(t, key, hidden=None, force_mark="none")
    ok = word in ("おわんだね", "おわんだよ")
    print(f"[机上確認] 入力=おわん 印なし c0 → 「{word}」 conf={conf:.4f} "
          f"top3={top3_to_str(top3)}  判定={'OK' if ok else 'NG'}", flush=True)
    if not ok:
        raise RuntimeError(
            f"机上確認に失敗：入力「おわん」・印なし・c0で「おわんだね/だよ」が"
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
    if "<HERE>" not in t.produce_vocab.char2idx:
        raise ValueError(
            "チェックポイントに<HERE>トークンが無い（produce.here_input=trueで"
            "訓練されたモデルが必要）。")
    here_id = t._here_id = t.produce_vocab.char2idx["<HERE>"]
    hip = t.language_hippocampus
    print(f"  <GONE> id={gone_id}  <HERE> id={here_id}  "
          f"海馬エピソード数={len(hip.episodes)}", flush=True)

    print("[2] 語彙の表からプロトタイプを読む", flush=True)
    protos = load_prototypes(blob)
    for w in INPUTS:
        print(f"  {w}: ノルム={float(np.linalg.norm(protos[w])):.4f}", flush=True)

    print("[3] 机上確認（先頭）", flush=True)
    sanity_check(t, protos)

    # ---- gru.csv：入力9種 × 印3種 × 文脈c0/c1 = 54行 ----
    print("[4] gru.csv 生成", flush=True)
    gru_rows = []
    keys = {w: project_key(t, protos[w]) for w in INPUTS}
    for w in INPUTS:
        key = keys[w]
        h_c1 = build_context_h(t, key, mark="gone", speaker=C1_SPEAKER, text=C1_TEXT)
        for mark in MARKS:
            for ctx_name, hidden in (("c0", None), ("c1", h_c1)):
                word, conf, top3 = generate(t, key, hidden, force_mark=mark)
                gru_rows.append({
                    "input": w, "mark": mark, "context": ctx_name,
                    "word": word, "conf": round(conf, 5),
                    "top3_first_tokens": top3_to_str(top3),
                })
    gru_csv = os.path.join(OUT_DIR, "gru.csv")
    with io.open(gru_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "input", "mark", "context", "word", "conf", "top3_first_tokens"])
        w_.writeheader()
        w_.writerows(gru_rows)
    print("  ->", gru_csv, f"({len(gru_rows)}行)", flush=True)

    # ---- hippo.csv：入力9種 × 印3種 = 27行 ----
    print("[5] hippo.csv 生成", flush=True)
    hippo_rows = []
    for w in INPUTS:
        key_np = keys[w].detach().cpu().numpy()
        for want in MARKS:
            toks, sim, strength = hippo_recall_filtered(hip, key_np, want, gone_id, here_id)
            toks_clean = strip_marks(toks, gone_id, here_id)
            word = t.produce_vocab.decode(toks_clean) if toks_clean else ""
            conf = sim * strength
            hippo_rows.append({
                "input": w, "want": want, "word": word,
                "sim": round(sim, 5), "strength": round(strength, 5),
                "conf": round(conf, 5),
            })
    hippo_csv = os.path.join(OUT_DIR, "hippo.csv")
    with io.open(hippo_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "input", "want", "word", "sim", "strength", "conf"])
        w_.writeheader()
        w_.writerows(hippo_rows)
    print("  ->", hippo_csv, f"({len(hippo_rows)}行)", flush=True)

    # ---- hippo_episodes.csv：海馬の全エピソード ----
    print("[6] hippo_episodes.csv 生成", flush=True)
    proto_keys64 = {w: keys[w].detach().cpu().numpy() for w in INPUTS}
    episode_rows = []
    for idx, ep in enumerate(hip.episodes):
        tok0 = ep["tokens"][0] if ep["tokens"] else None
        mark = "gone" if tok0 == gone_id else ("here" if tok0 == here_id else "none")
        toks_clean = strip_marks(ep["tokens"], gone_id, here_id)
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
            "idx": idx, "mark": mark, "word": word,
            "strength": round(float(ep["strength"]), 5),
            "written_at": int(ep["written_at"]),
            "nearest_word": best_w, "nearest_cos": round(best_cos, 5),
        })
    ep_csv = os.path.join(OUT_DIR, "hippo_episodes.csv")
    with io.open(ep_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=[
            "idx", "mark", "word", "strength", "written_at",
            "nearest_word", "nearest_cos"])
        w_.writeheader()
        w_.writerows(episode_rows)
    print("  ->", ep_csv, f"({len(episode_rows)}行)", flush=True)

    # ========================================================================
    # ⑤ 結果.md（要点のみ）
    # ========================================================================
    print("[7] 結果.md 集計", flush=True)

    def find_row(input_, mark, context):
        for r in gru_rows:
            if r["input"] == input_ and r["mark"] == mark and r["context"] == context:
                return r
        return None

    # 表1：8語 x 印=here x c0 で「語＋だね/だよ/単独」を言えた数
    #   （仕様書合否節：「正しい見た目＋あるの印で8語すべて『だね／だよ／単独』」）
    table1_lines = []
    n1_ok = 0
    for w in WORDS:
        r = find_row(w, "here", "c0")
        ok = r["word"] in (w + "だね", w + "だよ", w)
        n1_ok += int(ok)
        table1_lines.append(f"| {w} | {r['word']} | {r['conf']:.4f} | {'OK' if ok else '-'} |")

    # 表2：8語 x 印=gone x c0 で「語＋ないね」を言えた数
    table2_lines = []
    n2_ok = 0
    for w in WORDS:
        r = find_row(w, "gone", "c0")
        want = w + "ないね"
        ok = (r["word"] == want)
        n2_ok += int(ok)
        table2_lines.append(f"| {w} | {r['word']} | {r['conf']:.4f} | {'OK' if ok else '-'} |")

    # 表3：文脈c1（自分が直前に「おわんないね」）を入れると表1・表2がどう変わるか
    #   （変化した行だけ。合否節：「あるの印の行が『ないね』に反転しないこと」）
    table3_lines = []
    n1_flipped_to_nai = 0
    for w in WORDS:
        for mark in ("here", "gone"):
            r0 = find_row(w, mark, "c0")
            r1 = find_row(w, mark, "c1")
            if r0["word"] != r1["word"]:
                table3_lines.append(
                    f"| {w} | {mark} | {r0['word']}(conf={r0['conf']:.4f}) "
                    f"| {r1['word']}(conf={r1['conf']:.4f}) |")
                if mark == "here" and r1["word"] == w + "ないね":
                    n1_flipped_to_nai += 1

    # 表4：印3種 x c0/c1 の空の机（「ないね」プロトタイプ）でのGRUの答え
    table4_lines = []
    for mark in MARKS:
        for ctx in ("c0", "c1"):
            r = find_row(EMPTY, mark, ctx)
            table4_lines.append(
                f"| {mark} | {ctx} | {r['word']} | {r['conf']:.4f} | "
                f"{r['top3_first_tokens']} |")

    # 表5：海馬エピソードの印別件数・鍵の内訳・strengthの分布
    gone_eps = [r for r in episode_rows if r["mark"] == "gone"]
    here_eps = [r for r in episode_rows if r["mark"] == "here"]
    n_gone, n_here = len(gone_eps), len(here_eps)
    n_empty_desk_key = sum(1 for r in gone_eps if r["nearest_word"] == EMPTY)
    n_noun_key = n_gone - n_empty_desk_key
    strength_counter = Counter(round(r["strength"], 2) for r in gone_eps)
    table5_strength_lines = [f"| {s} | {c} |" for s, c in
                              sorted(strength_counter.items(), reverse=True)]
    pct_empty_desk = (100.0 * n_empty_desk_key / n_gone) if n_gone else 0.0

    judge1 = f"あるの印で GRU は正しい見た目のとき {n1_ok}/8 " + \
             ("言えた" if n1_ok == 8 else "言えなかった（『だね/だよ/単独』以外）")
    judge2 = f"消えたの印で GRU は正しい見た目のとき {n2_ok}/8 " + \
             ("言えた" if n2_ok == 8 else "言えなかった（『語＋ないね』以外）")
    judge3 = (f"文脈c1（自分が直前に『おわんないね』）を入れても、あるの印の行が"
              f"『語＋ないね』に反転した数＝{n1_flipped_to_nai}/8"
              f"（F2-79では6行反転。目標0）")
    judge4 = (f"海馬 <GONE> エピソード={n_gone}件（空の机寄り{n_empty_desk_key}件・"
              f"名詞寄り{n_noun_key}件） / <HERE> エピソード={n_here}件") if (n_gone or n_here) else \
             "海馬の印付きエピソードが0件（判定不能）"

    md = []
    md.append("# f83 あるの印・消えたの印の静止測定：結果\n")
    md.append(f"モデル: `{MODEL_PATH}`\n")
    md.append(f"## 表1：8語 x あるの印 x c0 で GRU が「だね/だよ/単独」を言えた数（{n1_ok}/8）\n")
    md.append("| 語 | word | conf | 判定 |")
    md.append("|---|---|---|---|")
    md.extend(table1_lines)
    md.append("")
    md.append(f"## 表2：8語 x 消えたの印 x c0 で GRU が「語＋ないね」を言えた数（{n2_ok}/8）\n")
    md.append("| 語 | word | conf | 判定 |")
    md.append("|---|---|---|---|")
    md.extend(table2_lines)
    md.append("")
    md.append("## 表3：文脈c1（自分が直前に「おわんないね」）を入れると"
               "あるの印/消えたの印の答えがどう変わるか（変化した行だけ）\n")
    if table3_lines:
        md.append("| 語 | 印 | c0の答え | c1の答え |")
        md.append("|---|---|---|---|")
        md.extend(table3_lines)
    else:
        md.append("変化した行なし。")
    md.append("")
    md.append("## 表4：空の机（ないねプロトタイプ）でのGRUの答え（印3種 x 文脈2種）\n")
    md.append("| 印 | 文脈 | word | conf | top3先頭文字 |")
    md.append("|---|---|---|---|---|")
    md.extend(table4_lines)
    md.append("")
    md.append(f"## 表5：海馬エピソードの印別件数・鍵の内訳・strengthの分布\n")
    md.append(f"- <GONE>件数={n_gone}（空の机寄り{n_empty_desk_key}件・{pct_empty_desk:.1f}% "
               f"/ 名詞寄り{n_noun_key}件）、<HERE>件数={n_here}\n")
    md.append("| strength(<GONE>) | 件数 |")
    md.append("|---|---|")
    md.extend(table5_strength_lines if table5_strength_lines else ["| (該当なし) | 0 |"])
    md.append("")
    md.append("## 判定\n")
    md.append(f"- {judge1}")
    md.append(f"- {judge2}")
    md.append(f"- {judge3}")
    md.append(f"- {judge4}")
    md.append("")

    md_path = os.path.join(OUT_DIR, "結果.md")
    with io.open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))
    print("  ->", md_path, flush=True)
    print("\n".join(md))
    print("[OK] 完了", flush=True)


if __name__ == "__main__":
    main()
