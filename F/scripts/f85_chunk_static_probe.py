# -*- coding: utf-8 -*-
"""塊レベル層の静止測定（f83の複製、2026-09-07）。

仕様：F/docs/二語文/仕様_M5_塊レベル層_2026-09-07.md 後半「9. 静止測定 f85」節。

f83（F/scripts/f83_here_gone_static_probe.py）と同じ入力（8語のプロトタイプ＋空の机
「ないね」）×印（なし／ある／消えた）×文脈（なし／自分が直前に「おわんないね」）で、
本体（塊GRU＝t.chunk_brain）の出力（塊の列を文字列に組み立てたもの）と確信度、
海馬（塊の列で書かれたエピソード）の想起をそれぞれ机上で確かめる。

モデルは走らせない（学習ステップを進めない）。段1（F2-84鎖）では「書くだけ・
動作確認はF2-84 r2のモデルで」（仕様書§9）という位置づけのため、判定は
仕様書の合否基準（段2用）を出すが、段1のモデル（「くつ」「だね」等の短い教示語彙
しか無い）では期待した語が出なくても異常ではない。

    .venv/Scripts/python.exe F/scripts/f85_chunk_static_probe.py [モデルパス]

既定モデルパス: F/logs/F2-84_塊レベル鎖/r2/model.pt
出力: <モデルと同じ場所>/静止測定/gru.csv
      <モデルと同じ場所>/静止測定/hippo.csv
      <モデルと同じ場所>/静止測定/結果.md
"""
import os
import sys
import io
import csv

sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src"))
for _sub in ("wrapper", "senses", "brain", "body"):
    sys.path.insert(0, os.path.abspath(os.path.join("taro_core", "src", _sub)))
sys.path.insert(0, os.path.abspath("F/scripts"))

import numpy as np
import torch

from cerebral_cortex.recurrent_core import TaroBrain
from cerebral_cortex.visual_projection import VisualProjection
from cerebral_cortex.chunk_vocab import ChunkVocab
from hearing import Vocabulary, normalize_kana
from language_hippocampus.language_hippocampus import LanguageHippocampus

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "F/logs/F2-84_塊レベル鎖/r2/model.pt"
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "静止測定")

WORDS = ["くつ", "こっぷ", "おさら", "おわん", "かばん", "がおお", "ばす", "ぼおる"]
EMPTY = "ないね"
INPUTS = WORDS + [EMPTY]

C1_TEXT = "おわんないね"
C1_SPEAKER = "self"     # 【仕様書§1】chunk_vocab.specials のキーは "parent"/"self"（taro_setup.py参照）
MAX_CHUNKS = 4           # 【仕様書§6】塊は最大4個で止める
MARKS = ("none", "here", "gone")


class HearingVocabView:
    """blob["hearing_vocab"]（モーラの名簿）を chunk_vocab.chunk_string が要求する
    「.decode(list[int]) -> str」インタフェースに合わせるだけの薄いラッパー。
    """

    def __init__(self, idx2char):
        self.idx2char = idx2char

    def decode(self, indices):
        chars = []
        for idx in indices:
            ch = self.idx2char.get(int(idx), "?")
            if ch not in ("<PAD>", "<BOS>", "<EOS>"):
                chars.append(ch)
        return "".join(chars)

    def encode(self, text):
        """テキスト→モーラid列（既知の文字だけを使う。未知文字はNoneを返し
        呼び出し側で弾く。動的追加はしない＝静止測定は保存済みモデルの中身
        だけを見る）。"""
        char2idx = {v: int(k) for k, v in self.idx2char.items()}
        ids = []
        for ch in text:
            if ch not in char2idx:
                return None
            ids.append(char2idx[ch])
        return ids


class ChunkTaro:
    """このテストに要る部分だけを持つ入れ物（f71.Taro71と同型）。"""
    pass


def build_taro_chunk(path):
    """チェックポイントから塊GRU・塊の名簿・視覚投射・言語海馬を復元する。

    フルの run/taro_setup.py は経由しない（重いMuJoCoシーン構築を避けるため。
    f71.build_taro と同じ「必要な部分だけ直接組み立てる」流儀）。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "chunk_vocab" not in blob or "chunk_brain" not in blob:
        raise ValueError(
            f"チェックポイントに塊レベル層のキー（chunk_vocab/chunk_brain）が無い。"
            f"produce.chunk_level=true で訓練されたモデルが必要: {path}")

    hv = blob["hearing_vocab"]
    idx2char = {int(i): c for c, i in hv["char2idx"].items()}
    hearing_vocab = HearingVocabView(idx2char)

    cv = ChunkVocab()
    cv.load_state_dict(blob["chunk_vocab"])

    chunk_brain = TaroBrain(vocab_size=3, embedding_dim=64, hidden_dim=128, body_state_dim=0)
    chunk_brain.resize_embedding(blob["chunk_brain"]["embedding.weight"].shape[0])
    chunk_brain.load_state_dict(
        {k: v for k, v in blob["chunk_brain"].items()
         if k in chunk_brain.state_dict() and chunk_brain.state_dict()[k].shape == v.shape},
        strict=False)

    vp = VisualProjection(384, chunk_brain.embedding.embedding_dim)
    vp.load_state_dict(blob["visual_projection"])

    if "language_hippocampus" not in blob:
        raise ValueError("チェックポイントに language_hippocampus が無い。")
    hippo = LanguageHippocampus()
    hippo.load_state_dict(blob["language_hippocampus"])

    t = ChunkTaro()
    t.chunk_vocab = cv
    t.chunk_brain = chunk_brain
    t._visual_projection = vp
    t.language_hippocampus = hippo
    t.hearing_vocab = hearing_vocab
    t.blob = blob
    return t


def load_prototypes(blob):
    """語彙の表(lexicon.proto)から9種のプロトタイプベクトル(384次元)を読む（f83と同一）。"""
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
        print(f"[警告] 語彙の表(lexicon.proto)に無い語: {missing}"
              f"（見つかった語: {sorted(word_to_vec)}）。F2-84段1の教示語彙は"
              f"「くつ」「だね」等のみのため、欠けは想定内。無い語は0ベクトルで埋める。")
    dim = blob["lexicon"]["channels"]["vision"]["dim"]
    return {w: word_to_vec.get(w, np.zeros(dim, dtype=np.float32)) for w in INPUTS}


def project_key(t, vec384):
    vin = torch.tensor(list(vec384), dtype=torch.float32)
    with torch.no_grad():
        return t._visual_projection(vin)


def build_context_h(t, key, mark, speaker, chunk_ids):
    """文脈c1のH：[<話者>](, <GONE>|<HERE>) + 塊id列 + EOS を forward_hidden に
    通した最終hidden（trainer.py _chunk_context_feed の最終ブロックと同じ手順）。
    """
    cv = t.chunk_vocab
    ids = [cv.specials[speaker]]
    if mark == "gone" and cv.specials.get("gone") is not None:
        ids.append(cv.specials["gone"])
    elif mark == "here" and cv.specials.get("here") is not None:
        ids.append(cv.specials["here"])
    ids = ids + list(chunk_ids) + [2]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        _, h = t.chunk_brain.forward_hidden(x, hidden=None, prefix_vec=key)
    return h.detach()


def generate_chunk(t, key, hidden, force_mark, max_chunks=MAX_CHUNKS):
    """trainer.py の gru_hippo ブロック（chunk_level分岐）をそのまま写した生成。

    戻り値: (word, conf, seq)。seqは選んだ塊idの列（gone/here除く）。
    """
    cv = t.chunk_vocab
    par_c = cv.specials.get("parent")
    gone_id_c = cv.specials.get("gone")
    here_id_c = cv.specials.get("here")
    with torch.no_grad():
        out, hh = t.chunk_brain.forward_hidden(
            torch.tensor([[par_c]], dtype=torch.long), hidden=hidden, prefix_vec=key)
        logits = t.chunk_brain.perception_head(out)[0, -1]
        if force_mark == "gone" and gone_id_c is not None:
            out, hh = t.chunk_brain.forward_hidden(
                torch.tensor([[gone_id_c]], dtype=torch.long), hidden=hh)
            logits = t.chunk_brain.perception_head(out)[0, -1]
        elif force_mark == "here" and here_id_c is not None:
            out, hh = t.chunk_brain.forward_hidden(
                torch.tensor([[here_id_c]], dtype=torch.long), hidden=hh)
            logits = t.chunk_brain.perception_head(out)[0, -1]
        conf = float(torch.softmax(logits, dim=-1).max())
        seq = []
        for _ in range(max_chunks):
            tk = int(torch.argmax(logits))
            if tk == 2:
                break
            seq.append(tk)
            out, hh = t.chunk_brain.forward_hidden(
                torch.tensor([[tk]], dtype=torch.long), hidden=hh)
            logits = t.chunk_brain.perception_head(out)[0, -1]
    seq_clean = [tk for tk in seq if tk != gone_id_c and tk != here_id_c]
    word = "".join(cv.chunk_string(tk, t.hearing_vocab) for tk in seq_clean)
    return word, conf, seq_clean


def hippo_recall_filtered(hip, key_vis, want, gone_id, here_id):
    """run/trainer.py `_hippo_recall_filtered` の複製（f83と同一の複製流儀）。"""
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


def strip_marks(tokens, gone_id, here_id):
    return [tk for tk in tokens if tk != gone_id and tk != here_id]


def chunk_ids_for_text(t, text):
    """テキストをモーラid化し、塊の名簿にあれば1個の塊として、無ければ
    段落ごとに空白なしの1塊としてencode_chunkする（c1文脈用の簡易変換。
    厳密な分節は行わない＝静止測定の入力を作るためだけの割り切り）。
    """
    ids = t.hearing_vocab.encode(text)
    if ids is None:
        return []
    return [t.chunk_vocab.encode_chunk(tuple(ids))]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1] モデル復元:", MODEL_PATH, flush=True)
    t = build_taro_chunk(MODEL_PATH)
    cv = t.chunk_vocab
    hip = t.language_hippocampus
    print(f"  塊の名簿サイズ={cv.size}  海馬unit={hip.unit}  海馬エピソード数={len(hip.episodes)}",
          flush=True)

    print("[2] 語彙の表からプロトタイプを読む", flush=True)
    protos = load_prototypes(t.blob)

    # 【机上確認】まず1つだけ生成して例外なく動くかを見る（f83のsanity_checkに相当）。
    print("[3] 机上確認（先頭1件）", flush=True)
    key0 = project_key(t, protos[WORDS[0]])
    word0, conf0, seq0 = generate_chunk(t, key0, hidden=None, force_mark="none")
    print(f"  入力={WORDS[0]} 印なし c0 → 「{word0}」 conf={conf0:.4f} 塊id列={seq0}", flush=True)

    # ---- gru.csv：入力9種 × 印3種 × 文脈c0/c1 ----
    print("[4] gru.csv 生成", flush=True)
    gru_rows = []
    keys = {w: project_key(t, protos[w]) for w in INPUTS}
    c1_chunk_ids = chunk_ids_for_text(t, C1_TEXT)
    for w in INPUTS:
        key = keys[w]
        h_c1 = build_context_h(t, key, mark="gone", speaker=C1_SPEAKER, chunk_ids=c1_chunk_ids)
        for mark in MARKS:
            for ctx_name, hidden in (("c0", None), ("c1", h_c1)):
                word, conf, seq = generate_chunk(t, key, hidden, force_mark=mark)
                gru_rows.append({
                    "input": w, "mark": mark, "context": ctx_name,
                    "word": word, "conf": round(conf, 5), "n_chunks": len(seq),
                })
    gru_csv = os.path.join(OUT_DIR, "gru.csv")
    with io.open(gru_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=["input", "mark", "context", "word", "conf", "n_chunks"])
        w_.writeheader()
        w_.writerows(gru_rows)
    print("  ->", gru_csv, f"({len(gru_rows)}行)", flush=True)

    # ---- hippo.csv：入力9種 × 印3種 ----
    print("[5] hippo.csv 生成", flush=True)
    gone_id_c = cv.specials.get("gone")
    here_id_c = cv.specials.get("here")
    hippo_rows = []
    for w in INPUTS:
        key_np = keys[w].detach().cpu().numpy()
        for want in MARKS:
            toks, sim, strength = hippo_recall_filtered(hip, key_np, want, gone_id_c, here_id_c)
            toks_clean = strip_marks(toks, gone_id_c, here_id_c)
            word = "".join(cv.chunk_string(tk, t.hearing_vocab) for tk in toks_clean)
            conf = sim * strength
            hippo_rows.append({
                "input": w, "want": want, "word": word,
                "sim": round(sim, 5), "strength": round(strength, 5), "conf": round(conf, 5),
            })
    hippo_csv = os.path.join(OUT_DIR, "hippo.csv")
    with io.open(hippo_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=["input", "want", "word", "sim", "strength", "conf"])
        w_.writeheader()
        w_.writerows(hippo_rows)
    print("  ->", hippo_csv, f"({len(hippo_rows)}行)", flush=True)

    # ---- 結果.md（要点のみ。段2の合否判定は仕様_M5後続で行う。ここは実測の記録のみ）----
    print("[6] 結果.md 記録", flush=True)
    md = []
    md.append("# f85 塊レベル層の静止測定：結果（段1・動作確認）\n")
    md.append(f"モデル: `{MODEL_PATH}`\n")
    md.append(f"塊の名簿サイズ={cv.size}  海馬unit={hip.unit}  海馬エピソード数={len(hip.episodes)}\n")
    md.append("段1のモデルは短期の教示語彙（「くつ」「だね」等）しか持たないため、"
               "ここでの語の一致・不一致は判定材料にしない（動作確認のみ）。"
               "段2（F2-85/F2-85b）で合否判定を行う。\n")
    md_path = os.path.join(OUT_DIR, "結果.md")
    with io.open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))
    print("  ->", md_path, flush=True)
    print("[OK] 完了", flush=True)


if __name__ == "__main__":
    main()
