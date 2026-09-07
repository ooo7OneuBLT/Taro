# -*- coding: utf-8 -*-
"""言いたいことの層の静止測定（f85の複製、2026-09-07）。

仕様：F/docs/二語文/仕様_M6_言いたいことの層_2026-09-07.md 後半「5. 静止測定 f87」節。
【M6b改訂・2026-09-07】F/docs/二語文/仕様_M6b_役割は見た目との結び付きで_2026-09-07.md
後半。判定文とrole_labelsの出力を修正。

f85（F/scripts/f85_chunk_static_probe.py）と同じ入力（8語のプロトタイプ＋空の机
「ないね」）×印（ある／消えた）×文脈（なし／c1）で、生成部だけを仕様書§3の
組み立て（先頭塊の分布→役割nounで絞る→compose）に差し替える。表には
noun・pred・組み立てた文字列を出す。判定文は「消えたの印で教えていない3語
（こっぷ・かばん・ぼおる）が『noun=入力語 かつ pred=ないね』になった数」
（M6bでの修正：M6は noun非空かつpredに『ないね』を含む、で判定していたため
noun側が入力と無関係な語でも通ってしまっていた）。

モデルは走らせない（学習ステップを進めない）。

    .venv/Scripts/python.exe F/scripts/f87_message_static_probe.py [モデルパス]

既定モデルパス: F/logs/F2-87_M6_学習/model.pt
出力: <モデルと同じ場所>/静止測定_M6/gru.csv
      <モデルと同じ場所>/静止測定_M6/結果.md
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
from cerebral_cortex.message_layer import MessageLayer
from hearing import Vocabulary, normalize_kana
from language_hippocampus.language_hippocampus import LanguageHippocampus

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "F/logs/F2-87_M6_学習/model.pt"
OUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "静止測定_M6")

# 【仕様_M5_塊レベル層.md §9 と同じ8語+空の机】うち5語は教えた語、
#   「こっぷ」「かばん」「ぼおる」の3語は教えていない語（仕様_M6合否節）。
WORDS = ["くつ", "こっぷ", "おさら", "おわん", "かばん", "がおお", "ばす", "ぼおる"]
EMPTY = "ないね"
INPUTS = WORDS + [EMPTY]
UNTAUGHT = {"こっぷ", "かばん", "ぼおる"}

C1_TEXT = "おわんないね"
C1_SPEAKER = "self"
MAX_CHUNKS = 4        # 【仕様書§6と同じ】塊は最大4個で止める
TOPK = 20             # 【仕様書§3・Tier3】softmax上位候補数（trainer.pyと同じ恣意的定数）
MARKS = ("here", "gone")   # M6の合否は「ある」「消えた」の2印だけを見る（仕様書「合否」節）


class HearingVocabView:
    """blob["hearing_vocab"]（モーラの名簿）を chunk_vocab.chunk_string が要求する
    「.decode(list[int]) -> str」インタフェースに合わせるだけの薄いラッパー
    （f85と同一の複製）。
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
        char2idx = {v: int(k) for k, v in self.idx2char.items()}
        ids = []
        for ch in text:
            if ch not in char2idx:
                return None
            ids.append(char2idx[ch])
        return ids


class MsgTaro:
    """このテストに要る部分だけを持つ入れ物（f85のChunkTaroと同型）。"""
    pass


def build_taro_message(path):
    """チェックポイントから塊GRU・塊の名簿・言いたいことの層・視覚投射・
    言語海馬を復元する（f85のbuild_taro_chunkに言いたいことの層を足しただけ）。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "chunk_vocab" not in blob or "chunk_brain" not in blob:
        raise ValueError(
            f"チェックポイントに塊レベル層のキー（chunk_vocab/chunk_brain）が無い。"
            f"produce.chunk_level=true で訓練されたモデルが必要: {path}")
    if "message_layer" not in blob:
        raise ValueError(
            f"チェックポイントに言いたいことの層のキー（message_layer）が無い。"
            f"produce.message_level=true で訓練されたモデルが必要: {path}")

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

    ml = MessageLayer()
    ml.load_state_dict(blob["message_layer"])

    t = MsgTaro()
    t.chunk_vocab = cv
    t.chunk_brain = chunk_brain
    t._visual_projection = vp
    t.language_hippocampus = hippo
    t.message_layer = ml
    t.hearing_vocab = hearing_vocab
    t.blob = blob
    return t


def load_prototypes(blob):
    """語彙の表(lexicon.proto)から9種のプロトタイプベクトル(384次元)を読む（f85と同一）。"""
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
              f"（見つかった語: {sorted(word_to_vec)}）。無い語は0ベクトルで埋める。")
    dim = blob["lexicon"]["channels"]["vision"]["dim"]
    return {w: word_to_vec.get(w, np.zeros(dim, dtype=np.float32)) for w in INPUTS}


def project_key(t, vec384):
    vin = torch.tensor(list(vec384), dtype=torch.float32)
    with torch.no_grad():
        return t._visual_projection(vin)


def chunk_ids_for_text(t, text):
    """テキストをモーラid化し、塊の名簿にある塊で左から最長一致に切る
    （c1文脈用の簡易変換。f85のchunk_ids_for_textと同一の複製）。
    """
    ids = t.hearing_vocab.encode(text)
    if ids is None:
        return []
    cv = t.chunk_vocab
    out, i = [], 0
    while i < len(ids):
        best = None
        for j in range(len(ids), i, -1):
            cid = cv.chunk2idx.get(tuple(ids[i:j]))
            if cid is not None:
                best = (cid, j); break
        if best is None:
            print(f"[警告] c1文脈の位置{i}から名簿に一致する塊が無い。読み飛ばす"); i += 1; continue
        out.append(best[0]); i = best[1]
    return out


def build_context_h(t, key, mark, speaker, chunk_ids):
    """f85のbuild_context_hと同一の複製（塊レベルの文脈hidden作成）。"""
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


def generate_message(t, key, hidden, state):
    """trainer.py の gru_hippoブロック（message_level分岐）をそのまま写した生成
    （仕様_M6_言いたいことの層.md 後半§3）。

    戻り値: (word, conf, seq, roles)。
    """
    cv = t.chunk_vocab
    par_c = cv.specials.get("parent")
    gone_id_c = cv.specials.get("gone")
    here_id_c = cv.specials.get("here")
    with torch.no_grad():
        # 先頭の塊の分布だけ取る（<PARENT>直後、印は入れない）。
        out0, _hh0 = t.chunk_brain.forward_hidden(
            torch.tensor([[par_c]], dtype=torch.long), hidden=hidden, prefix_vec=key)
        logits0 = t.chunk_brain.perception_head(out0)[0, -1]
        probs0 = torch.softmax(logits0, dim=-1)
        topk = min(TOPK, probs0.shape[-1])
        top_vals, top_idx = torch.topk(probs0, topk)
    noun_candidates = [
        (int(i), float(p)) for p, i in zip(top_vals.tolist(), top_idx.tolist())
        if t.message_layer.role(int(i)) == "noun"]
    seq, conf, roles = t.message_layer.compose(state, noun_candidates)
    seq_clean = [tk for tk in seq if tk != gone_id_c and tk != here_id_c]
    word = "".join(cv.chunk_string(tk, t.hearing_vocab) for tk in seq_clean)
    noun_str, pred_str = "", ""
    for tk, r in zip(seq, roles):
        s = cv.chunk_string(tk, t.hearing_vocab)
        if r == "noun":
            noun_str = s
        elif r == "pred":
            pred_str = s
    return word, conf, seq_clean, roles, noun_str, pred_str


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1] モデル復元:", MODEL_PATH, flush=True)
    t = build_taro_message(MODEL_PATH)
    cv = t.chunk_vocab
    ml = t.message_layer
    print(f"  塊の名簿サイズ={cv.size}"
          f"  役割表(state_count)={len(ml.state_count)}件"
          f"  述語表(pred_count)={ {k: len(v) for k, v in ml.pred_count.items()} }"
          f"  順番表(role_bigram)={ml.role_bigram}", flush=True)

    print("[2] 語彙の表からプロトタイプを読む", flush=True)
    protos = load_prototypes(t.blob)

    print("[3] 机上確認（先頭1件）", flush=True)
    key0 = project_key(t, protos[WORDS[0]])
    word0, conf0, seq0, roles0, noun0, pred0 = generate_message(t, key0, hidden=None, state="here")
    print(f"  入力={WORDS[0]} 印=here c0 → 「{word0}」(noun={noun0} pred={pred0}) "
          f"conf={conf0:.4f} roles={roles0}", flush=True)

    print("[4] gru.csv 生成（8語×印(here/gone)×文脈(c0/c1)）", flush=True)
    rows = []
    keys = {w: project_key(t, protos[w]) for w in INPUTS}
    c1_chunk_ids = chunk_ids_for_text(t, C1_TEXT)
    for w in INPUTS:
        key = keys[w]
        h_c1 = build_context_h(t, key, mark="gone", speaker=C1_SPEAKER, chunk_ids=c1_chunk_ids)
        for mark in MARKS:
            for ctx_name, hidden in (("c0", None), ("c1", h_c1)):
                word, conf, seq, roles, noun_s, pred_s = generate_message(
                    t, key, hidden, state=mark)
                rows.append({
                    "input": w, "mark": mark, "context": ctx_name,
                    "noun": noun_s, "pred": pred_s, "word": word,
                    "conf": round(conf, 5), "roles": "|".join(roles),
                })
    gru_csv = os.path.join(OUT_DIR, "gru.csv")
    with io.open(gru_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(
            fp, fieldnames=["input", "mark", "context", "noun", "pred", "word", "conf", "roles"])
        w_.writeheader()
        w_.writerows(rows)
    print("  ->", gru_csv, f"({len(rows)}行)", flush=True)

    # 【M6b・2026-09-07・仕様_M6b判定修正】判定文を「noun が入力の語と一致
    #   かつ pred が『ないね』」に直す（M6の判定「noun非空かつpredにないねを
    #   含む」は、noun側が入力と無関係な語でも通ってしまっていたため）。
    print("[5] 判定文の集計", flush=True)
    hit = 0
    detail = []
    for r in rows:
        if r["input"] in UNTAUGHT and r["mark"] == "gone" and r["context"] == "c0":
            ok = (r["noun"] == r["input"]) and (r["pred"] == "ないね")
            detail.append((r["input"], r["word"], r["noun"], r["pred"], ok))
            if ok:
                hit += 1
    print(f"  消えたの印×教えていない3語（c0）：{hit}/3 が「noun=入力語 かつ pred=ないね」",
          flush=True)
    for d in detail:
        print("   ", d, flush=True)

    # 【M6b】役割の一覧（塊・観測数・vis_cons・役割）を出力する（仕様書「報告」節）。
    print("[5.5] 役割の一覧", flush=True)
    role_rows = []
    for cid, sc in sorted(ml.state_count.items()):
        n = sc["here"] + sc["gone"]
        vc = ml.vis_cons.get(cid)
        r = ml.role(cid)
        word = cv.chunk_string(cid, t.hearing_vocab)
        role_rows.append({
            "chunk_id": cid, "word": word, "n_obs": n,
            "vis_cons": (round(vc, 5) if vc is not None else ""),
            "role": (r if r is not None else ""),
        })
        print(f"   chunk={cid} word={word} n_obs={n}"
              f" vis_cons={vc if vc is not None else 'None'} role={r}", flush=True)
    role_csv = os.path.join(OUT_DIR, "役割一覧.csv")
    with io.open(role_csv, "w", encoding="utf-8", newline="") as fp:
        w_ = csv.DictWriter(fp, fieldnames=["chunk_id", "word", "n_obs", "vis_cons", "role"])
        w_.writeheader()
        w_.writerows(role_rows)
    print("  ->", role_csv, f"({len(role_rows)}行)", flush=True)

    print("[6] 結果.md 記録", flush=True)
    md = []
    md.append("# f87 言いたいことの層の静止測定：結果\n")
    md.append(f"モデル: `{MODEL_PATH}`\n")
    md.append(f"消えたの印×教えていない3語（こっぷ・かばん・ぼおる、c0文脈）の"
              f"「noun=入力語 かつ pred=ないね」率: {hit}/3\n")
    md.append("詳細（input, word, noun, pred, ok):\n")
    for d in detail:
        md.append(f"- {d}\n")
    md.append(f"\n役割の一覧: `{role_csv}`\n")
    md_path = os.path.join(OUT_DIR, "結果.md")
    with io.open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))
    print("  ->", md_path, flush=True)
    print("[OK] 完了", flush=True)


if __name__ == "__main__":
    main()
