# -*- coding: utf-8 -*-
"""F2-74 空机プローブ（2026-09-05・書き捨て診断・読み取り専用）。

F2-74（`F/logs/F2-74_M2_消失発話/`）で、机が空の時間帯（sim_sec ≈ 8.5・18.5・26.6・28.6）に
太郎が毎回「くつだね／くつだよ」と言った。物がある時間帯（4.5 おわん・14.5 がおー・24.6 かばん）
は正しく言えている。この書き捨てスクリプトは、
  (a) 空の机の見た目が、太郎の語彙8語のうち「くつ」に一番近いか（DINOv2の見た目コサイン類似度）
  (b) GRUの先頭音の分布が空の机でどうなるか（top-3確率）
を静止測定で確かめる。学習・保存・海馬書き込みは一切しない。

【材料】
  画像：F/logs/F2-74_M2_消失発話/検出フレーム/frame_<step>.png（224x224、
        obs["eye_left"]をPIL resizeしたもの＝run/plugins/common/object_files.py 151-153行）。
        空の机＝sim_sec 8.5/18.5/26.6/28.6付近の各1枚、物あり＝4.5/14.5/24.6付近の各1枚
        （目視で空/物ありを確認済み。下記FRAMESのコメント参照）。
  モデル：F/models/F2-50_r8_seed98_2026-09-03.pt（F2-74の開始モデル）。
        語彙プロトタイプの読み方は F/scripts/f68_word_expectation_probe.py に倣う。
  視覚の符号化：run/trainer.py `_vision_backend_encode`（249-294行、2026-09-05時点）を
        読んだ結果、F2-74の設定 lexicon_vision={"source":"wide", "eye":"left", "fovea_px":32}
        では、`source=="wide"` のとき `_wide=True` となり
          - 画像は eye_left_fovea ではなく eye_left（wide, 未クロップ）を使う
          - has_fovea_px_attr が True になり、encode() 呼び出しの間だけ
            backend.fovea_px を 10**9 に一時上書きする（fovea_crop側の
            `min(fovea_px, h)` により実質ノークロップになる）
        ＝ 仕様に書かれた「fovea_px=32で中心切り出し」は、source="wide"下では
        実際には**発動しない**（no-opに上書きされる）。これは仕様の記述と食い違うが、
        コードを読んだ実測に基づく（run/trainer.py 264-294行）。よってこのスクリプトは
        vision_backends.get_backend({"backend":"dinov2_vits14","fovea_px":10**9,"eye":"left"})
        をそのまま呼び、fovea_crop を掛けない（＝production呼び出しの実際の経路を再現）。
        近似が残る点：検出フレームPNGは既にPIL resizeで224x224化されたもの
        （object_files.py独自の処理）であり、trainer本体が使う生のeye_left画像
        （解像度不明・trainer内でtorch.bilinearにより224へ拡大）とは、リサイズの
        アルゴリズムが違う（PIL既定resample vs torch bilinear）。中身は同じ画像なので
        大きくは変わらないはずだが、バイト単位の完全一致ではない。

    .venv/Scripts/python.exe F/scripts/f75_empty_desk_probe.py
出力: F/logs/F2-74_M2_消失発話/空机プローブ/結果.json
      F/logs/F2-74_M2_消失発話/空机プローブ/表.csv
      F/logs/F2-74_M2_消失発話/空机プローブ/図_空机_類似度.png
      F/logs/F2-74_M2_消失発話/空机プローブ/図_使った画像.png
"""
import os, sys, io, json, csv, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("F/scripts"))

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from senses.vision_backends import get_backend
import f71_pattern_generalization_disambiguation as F71  # build_taro（そのまま流用）

MODEL_PATH = "F/models/F2-50_r8_seed98_2026-09-03.pt"
RUN_DIR = "F/logs/F2-74_M2_消失発話"
FRAMES_DIR = os.path.join(RUN_DIR, "検出フレーム")
OUT = os.path.join(RUN_DIR, "空机プローブ")

# 語彙8語（仕様指定の並び）
TARGET_WORDS = ["くつ", "コップ", "おさら", "おわん", "かばん", "がおー", "バス", "ボール"]

# 使用フレーム：物体ファイル.csv（step,sim_time）を突き合わせ、仕様のsim_sec付近で
# 目視確認（本スクリプト実装時にReadツールで7枚とも開いて確認）した結果を固定値で持つ。
#   空の机の目視結果：frame_00077(t=8.2s)=机と壁だけ／frame_00179(t=18.4s)=同／
#     frame_00267(t=27.2s)=同／frame_00278(t=28.3s)=同（いずれも物は写っていない）。
#   物ありの目視結果：frame_00045(t=5.0s)=白い椀（おわん）／frame_00137(t=14.2s)=
#     恐竜のおもちゃ（がおー）／frame_00245(t=25.0s)=青いかばん（かばん）。
#   注：仕様のsim_sec 26.6には最も近いstep256(t=26.1s)が物体ファイルCSV上でも
#   n_dets=2で存在したが、目視するとまだ「かばん」が写っており空ではなかった
#   （画像で確認：frame_00256.png）。そのため空の机が実際に成立している最も近い
#   フレームstep267(t=27.2s)を採用した（仕様の想定時刻と最大0.6秒ずれるが、
#   「空である」ことを優先した）。
FRAMES = [
    {"step": 77,  "sim_time": 8.2,  "target_sim_sec": 8.5,  "label": "空",           "expect": None},
    {"step": 179, "sim_time": 18.4, "target_sim_sec": 18.5, "label": "空",           "expect": None},
    {"step": 267, "sim_time": 27.2, "target_sim_sec": 26.6, "label": "空",           "expect": None},
    {"step": 278, "sim_time": 28.3, "target_sim_sec": 28.6, "label": "空",           "expect": None},
    {"step": 45,  "sim_time": 5.0,  "target_sim_sec": 4.5,  "label": "物あり(おわん)", "expect": "おわん"},
    {"step": 137, "sim_time": 14.2, "target_sim_sec": 14.5, "label": "物あり(がおー)", "expect": "がおー"},
    {"step": 245, "sim_time": 25.0, "target_sim_sec": 24.6, "label": "物あり(かばん)", "expect": "かばん"},
]


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float((a / na) @ (b / nb))


def load_word_prototypes(blob):
    """語→プロトタイプ(384,) の辞書を作る（f68_word_expectation_probe.py に倣う）。"""
    idx2char = blob["hearing_vocab"]["idx2char"]

    def _decode(chunk):
        return "".join(idx2char.get(i, "?") for i in chunk)

    raw_protos = blob["lexicon"]["proto"]
    protos = {}
    for w in TARGET_WORDS:
        w_norm = w.replace("ー", "")  # 長音は耳の入口で別表記（例：がおー→がおお）
        cands = [(chunk, v) for chunk, v in raw_protos.items() if _decode(chunk).startswith(w_norm)]
        if not cands:
            print("[警告] 語→プロトタイプ 見つからず:", w)
            continue
        chunk, v = min(cands, key=lambda cv: len(cv[0]))
        protos[w] = np.asarray(v, dtype=np.float64)
        print("語→プロトタイプ:", w, "<-", _decode(chunk))
    missing = [w for w in TARGET_WORDS if w not in protos]
    if missing:
        raise ValueError(f"プロトタイプが見つからない語がある: {missing}")
    return protos


def generate_with_top3(t, vec, max_length=8):
    """generate_probe（f71由来）と同一ロジック＋先頭音のtop-3確率を追加で取る。

    出典：F/scripts/f71_pattern_generalization_disambiguation.py generate_probe()
    （run/trainer.py `_apply_word_production` の `_wc == "gru_hippo"` ブロック本体側の再現）。
    「先頭音」＝<PARENT>トークン＋視覚prefixを与えた直後、最初に出す文字のsoftmax分布。
    """
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    dev = t.brain._device()
    vin = torch.tensor(list(vec), dtype=torch.float32, device=dev)
    with torch.no_grad():
        key = t._visual_projection(vin)
        out, hh = t.brain.forward_hidden(
            torch.tensor([[par]], dtype=torch.long, device=dev),
            hidden=None, prefix_vec=key)
        logits = t.brain.perception_head(out)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top3_idx = torch.topk(probs, min(3, probs.shape[-1])).indices.tolist()
        top3 = [(pv.idx2char.get(i, "?"), round(float(probs[i]), 5)) for i in top3_idx]
        seq = []
        for _ in range(max_length):
            tk = int(torch.argmax(logits))
            if tk == 2:  # <EOS>
                break
            seq.append(tk)
            out, hh = t.brain.forward_hidden(
                torch.tensor([[tk]], dtype=torch.long, device=dev), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
    word = pv.decode(seq) if seq else ""
    return word, top3


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    print("[1] チェックポイント読み込み:", MODEL_PATH)
    blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    protos = load_word_prototypes(blob)

    print("[2] 視覚バックエンド構築（dinov2_vits14, eye=left, fovea_px=10**9=ノークロップ）")
    # 【注】fovea_px=10**9 は「クロップしない」という指示であり、production側が
    #   source="wide" のとき encode() 呼び出し中に一時的に行っている上書きと同じ値
    #   （run/trainer.py 282行 `backend.fovea_px = 10 ** 9`）。
    backend = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9, "eye": "left"})

    print("[3] 画像読み込み・符号化・語彙8語とのコサイン類似度")
    rows = []
    images_for_grid = []
    for fr in FRAMES:
        path = os.path.join(FRAMES_DIR, "frame_%05d.png" % fr["step"])
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = np.array(Image.open(path).convert("RGB"))
        vec = np.asarray(backend.encode(img, img), dtype=np.float64)
        sims = {w: _cos(vec, protos[w]) for w in TARGET_WORDS}
        nearest = max(sims.items(), key=lambda kv: kv[1])
        rows.append({"frame": fr, "vec": vec, "sims": sims, "nearest": nearest})
        images_for_grid.append((fr, img))
        print("  step=%-4d t=%5.1fs %-14s 最近傍=%-6s(%.3f)  くつ=%.3f"
              % (fr["step"], fr["sim_time"], fr["label"], nearest[0], nearest[1], sims["くつ"]))

    print("\n[4] GRU生成（本体・海馬なし）と先頭音top-3")
    print("    モデル読み込み(f71.build_taro)…")
    t = F71.build_taro(MODEL_PATH)
    gen_results = []
    for r in rows:
        fr = r["frame"]
        word, top3 = generate_with_top3(t, r["vec"])
        gen_results.append({"frame": fr, "word": word, "top3": top3})
        top3_str = " / ".join("%s:%.3f" % (c, p) for c, p in top3)
        print("  step=%-4d %-14s 生成=%-10r 先頭音top3= %s" % (fr["step"], fr["label"], word, top3_str))

    print("\n[5] 出力: JSON・CSV")
    result = {
        "model": MODEL_PATH,
        "target_words": TARGET_WORDS,
        "frames": [],
    }
    csv_rows = []
    for r, g in zip(rows, gen_results):
        fr = r["frame"]
        entry = {
            "step": fr["step"], "sim_time": fr["sim_time"], "target_sim_sec": fr["target_sim_sec"],
            "label": fr["label"], "expect": fr["expect"],
            "sims": {w: round(v, 5) for w, v in r["sims"].items()},
            "nearest_word": r["nearest"][0], "nearest_sim": round(r["nearest"][1], 5),
            "kutsu_sim": round(r["sims"]["くつ"], 5),
            "generated": g["word"], "top3_first_char": g["top3"],
        }
        result["frames"].append(entry)
        row = {"step": fr["step"], "sim_time": fr["sim_time"], "label": fr["label"],
               "expect": fr["expect"], "nearest_word": entry["nearest_word"],
               "nearest_sim": entry["nearest_sim"], "kutsu_sim": entry["kutsu_sim"],
               "generated": entry["generated"],
               "top3_first_char": " / ".join("%s:%.3f" % (c, p) for c, p in g["top3"])}
        for w in TARGET_WORDS:
            row["sim_" + w] = entry["sims"][w]
        csv_rows.append(row)

    jpath = os.path.join(OUT, "結果.json")
    with io.open(jpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("生データ:", jpath)

    cpath = os.path.join(OUT, "表.csv")
    fieldnames = ["step", "sim_time", "label", "expect", "nearest_word", "nearest_sim",
                  "kutsu_sim", "generated", "top3_first_char"] + ["sim_" + w for w in TARGET_WORDS]
    with io.open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerows(csv_rows)
    print("表:", cpath)

    print("\n[6] 図: ヒートマップ")
    labels = ["%s\nstep%d t=%.1fs" % (fr["label"], fr["step"], fr["sim_time"]) for fr in FRAMES]
    mat = np.array([[r["sims"][w] for w in TARGET_WORDS] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TARGET_WORDS))); ax.set_xticklabels(TARGET_WORDS, fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="コサイン類似度")
    ax.set_title("空の机 / 物ありフレーム × 語彙8語 のコサイン類似度（F2-74開始モデル）", fontsize=11)
    fig.tight_layout()
    hpath = os.path.join(OUT, "図_空机_類似度.png")
    fig.savefig(hpath, dpi=120)
    print("図:", hpath)

    print("[7] 図: 使った画像7枚")
    fig2, axes = plt.subplots(1, len(FRAMES), figsize=(3.0 * len(FRAMES), 3.4))
    for ax, (fr, img) in zip(axes, images_for_grid):
        ax.imshow(img)
        ax.set_title("%s\nstep%d t=%.1fs" % (fr["label"], fr["step"], fr["sim_time"]), fontsize=9)
        ax.axis("off")
    fig2.suptitle("空机プローブに使った7枚（検出器に渡した224x224画像そのまま）", fontsize=11)
    fig2.tight_layout()
    gpath = os.path.join(OUT, "図_使った画像.png")
    fig2.savefig(gpath, dpi=120)
    print("図:", gpath)

    print("\n[OK] 完了")


if __name__ == "__main__":
    main()
