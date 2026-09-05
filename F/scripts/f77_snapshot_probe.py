# -*- coding: utf-8 -*-
"""F2-74c 発話瞬間プローブ（2026-09-05・書き捨て診断・読み取り専用）。

F2-74（`F/logs/F2-74_M2_消失発話/`）で、机が空の時間帯に太郎が毎回「くつだね」と
言う件。f75（1秒刻みの検出フレーム＝object_filesが別途スキャンして作る俯瞰
224x224画像）・f76（発話文脈の再生、同じ検出フレームを使用）のどちらでも
「くつ」は再現しなかった。残る候補は「発話の瞬間に太郎の目に実際に写っていた
画像」が検出フレームと違うこと。

このスクリプトは、F2-74cで新設した produce_snapshot プラグイン
（run/plugins/common/produce_snapshot.py）が保存した画像（＝
`ctx.last["obs_out"]["eye_left"]`、trainer._vision_backend_encode と同じ選び方）
に対して、f75と同じ符号化（切り出し無し・DINOv2, fovea_px=10**9）と
語彙最近傍・GRU自力生成（hidden=None）を行う。

【重要な訂正・produce_snapshot自体の保存画像は使わない】
  produce_snapshot.py の主出力（out_dir配下の「NNNN_語_simX.XXX.png」、
  および視覚ベクトル.npz の vec）は、**F2-74c の設定
  (lexicon_vision.source="wide", fovea_camera=false) では、実際に語の選択に
  使われた画像と一致しない**ことが分かった。理由：
    - trainer._vision_backend_encode（run/trainer.py 265-294行）は
      lexicon_vision.source=="wide" のとき、fovea_camera の有無に関わらず
      backend.fovea_px を encode() 呼び出しの間だけ 10**9（無クロップ）に
      一時的に上書きする（279行 `has_fovea_px_attr = (has_fovea or _wide) and ...`）。
      F2-74cはこの分岐＝**周辺60度の画像まるごと・無クロップ**が使われた。
    - 一方 produce_snapshot.py は `has_fovea = "eye_left_fovea" in obs and
      "eye_right_fovea" in obs` だけで分岐しており、`_wide`（lexicon_vision.source）
      を見ていない（2026-08-28に書かれた設計で、2026-08-31の"wide"機構より前の
      想定が残ったまま。produce_snapshot.py 90-93行のコメント「周辺カメラ(eye_left)
      から32px切り出すのは fovea_camera=False のシーンだけ」は、source="wide"が
      無かった時点では正しかったが、現在のtrainer.pyの分岐とは食い違う）。
      その結果 produce_snapshot は has_fovea=False の経路に入り、
      `crop = self._crop(img, fovea_px)` で **backend.fovea_px の「そのときの
      値」＝32px（lexicon_vision.fovea_px=32・trainer側の一時上書きは既に
      finally で戻った後）を使って中央32pxへ切り出す**。npzのvecも同じ32px
      クロップ由来（②の else 分岐は encode(obs["eye_left"], obs["eye_right"]) を
      呼ぶが、その内部でも同じ backend.fovea_px=32 が使われるため実質同じ）。
    - 実測で確認済み：保存PNG（例 F/logs/F2-74c_M2_発話瞬間画像/発話瞬間スナップ/
      0005_くつだね_sim0.783.png）を画素値で調べると、7px周期のブロック状
      （242px÷7≈34.5px、fovea_px=32に対応）が現れ、32px画像を最近傍補間で
      拡大した絵であることが分かる（このスクリプト実装時にBashで検算済み）。
  ⇒ out_dir の画像・npzベクトルは「trainer が実際に語の選択に使った画像」
    ではなく「32px中央だけを切り出した別の画像」。これをそのまま使うと
    f75と同じ誤りを繰り返す（見せる絵が実際の入力と違う）。

  **代わりに full_dir（発話瞬間スナップ_全体）の4枚組パネルのうち、
  左上「左目・周辺(60度)」パネル（＝`wl = obs["eye_left"]`そのもの、無クロップ）
  を使う。** produce_snapshot.py 140-166行を読むと、has_fovea=Falseのときも
  `wl = np.asarray(obs["eye_left"])`（クロップ前の生画像）がそのままこのパネルに
  描かれる（134-166行）。これが trainer が実際に符号化に使った画像と一致する
  （trainer は source="wide" のとき fovea_px=10**9でクロップを無効化した
  obs["eye_left"]をそのままDINOv2へ渡すので、パネルの絵と同一）。
  ただし、このパネル画像はmatplotlibが4枚組の1枚として再描画したPNGから
  該当区画をピクセル単位で切り出したものであり、trainer本体が使う生の配列
  そのものではない（imshow(interpolation="nearest")で拡大描画されているため、
  多少のブロック状拡大＋JPEG非可逆のないPNG保存だが、リサイズアルゴリズムは
  近似）。f75・f76が既に抱えている「検出フレームPNGは元の生配列と同一ではない」
  という近似と同じ種類・同程度の近似として扱う。

【対象フレーム】F2-74c走行（seed=91,300ステップ）の太郎の発話.csv・
  物体ファイル.csv を突き合わせ、空の机（n_dets=0、直近フレーム）で発話した
  全件（4件：8.5/18.5/26.6/28.6s、いずれも「くつ」）＋物ありで発話した3件
  （4.5s おわん／12.5s がおー／24.6s かばん）を対象にする。

    .venv/Scripts/python.exe F/scripts/f77_snapshot_probe.py
出力: F/logs/F2-74c_M2_発話瞬間画像/発話瞬間プローブ/結果.json
      F/logs/F2-74c_M2_発話瞬間画像/発話瞬間プローブ/表.csv
      F/logs/F2-74c_M2_発話瞬間画像/発話瞬間プローブ/図_類似度.png
      F/logs/F2-74c_M2_発話瞬間画像/発話瞬間プローブ/図_使った画像.png
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
import f71_pattern_generalization_disambiguation as F71   # build_taro をそのまま流用
import f75_empty_desk_probe as F75                        # _cos / load_word_prototypes / generate_with_top3 を流用

MODEL_PATH = "F/models/F2-50_r8_seed98_2026-09-03.pt"
RUN_DIR = "F/logs/F2-74c_M2_発話瞬間画像"
FULL_DIR = os.path.join(RUN_DIR, "発話瞬間スナップ_全体")
OUT = os.path.join(RUN_DIR, "発話瞬間プローブ")

TARGET_WORDS = ["くつ", "コップ", "おさら", "おわん", "かばん", "がおー", "バス", "ボール"]

# 太郎の発話.csv（F2-74c走行）の通し番号（produce_snapshotの発話番号と一致：
# 両者とも「ctx.last_produce が立った回数」を1から数えるカウンタなので同じ順序）。
# 空の机＝物体ファイル.csvでn_dets=0の直近ステップに一致する発話（4件、全件）。
# 物あり＝3件（4.5s おわん／12.5s がおー／24.6s かばん）。
FRAMES = [
    {"idx": 5,  "step": 85,  "sim_time": 8.5,  "label": "空",           "expect": None,   "word_csv": "くつだね"},
    {"idx": 10, "step": 185, "sim_time": 18.5, "label": "空",           "expect": None,   "word_csv": "くつだね"},
    {"idx": 14, "step": 266, "sim_time": 26.6, "label": "空",           "expect": None,   "word_csv": "くつだよ"},
    {"idx": 15, "step": 286, "sim_time": 28.6, "label": "空",           "expect": None,   "word_csv": "くつだね"},
    {"idx": 3,  "step": 45,  "sim_time": 4.5,  "label": "物あり(おわん)", "expect": "おわん", "word_csv": "おわんだよ"},
    {"idx": 7,  "step": 125, "sim_time": 12.5, "label": "物あり(がおー)", "expect": "がおー", "word_csv": "がおお"},
    {"idx": 13, "step": 246, "sim_time": 24.6, "label": "物あり(かばん)", "expect": "かばん", "word_csv": "かばん"},
]


def extract_wide_left_panel(png_path):
    """produce_snapshotのfull_dir 4枚組PNGから、左上「左目・周辺(60度)」区画
    （＝trainerが実際に符号化に使ったobs["eye_left"]そのもの、無クロップ）を
    ピクセル単位で切り出す。境界は白背景(RGB>248)の行/列ギャップを実測して
    自動検出する（画像ごとにタイトル文字列の長さが違っても頑健なように、
    固定座標ではなく都度検出する）。
    """
    im = np.array(Image.open(png_path).convert("RGB"))
    white = (im[:, :, 0] > 248) & (im[:, :, 1] > 248) & (im[:, :, 2] > 248)
    rowfrac = white.mean(axis=1)
    colfrac = white.mean(axis=0)

    def _content_blocks(frac, min_len=20):
        """frac<0.9（非白＝内容あり）が続く区間を検出し、最初の連続ブロックを返す。"""
        mask = frac < 0.9
        idx = np.where(mask)[0]
        if len(idx) == 0:
            raise ValueError("内容行/列が見つからない: " + png_path)
        blocks = []
        cur = [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                blocks.append(cur)
                cur = [i]
        blocks.append(cur)
        blocks = [b for b in blocks if len(b) >= min_len]
        return blocks

    row_blocks = _content_blocks(rowfrac)
    col_blocks = _content_blocks(colfrac)
    # 1行目（上段パネル）＝最初のrowブロック、1列目（左パネル）＝最初のcolブロック
    r0, r1 = row_blocks[0][0], row_blocks[0][-1]
    c0, c1 = col_blocks[0][0], col_blocks[0][-1]
    return im[r0:r1 + 1, c0:c1 + 1, :]


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    print("[1] チェックポイント読み込み:", MODEL_PATH)
    blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    protos = F75.load_word_prototypes(blob)

    print("[2] 視覚バックエンド構築（dinov2_vits14, eye=left, fovea_px=10**9=ノークロップ、f75と同じ）")
    backend = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9, "eye": "left"})

    print("[3] 発話瞬間の実画像（produce_snapshot full_dir・左目周辺60度パネル）を読み込み・符号化")
    rows = []
    images_for_grid = []
    for fr in FRAMES:
        png_path = os.path.join(FULL_DIR, "%04d_%s_sim" % (fr["idx"], fr["word_csv"]))
        # ファイル名末尾のsim値は発話ごとに違うのでglobで拾う
        import glob
        cands = glob.glob(png_path + "*.png")
        if not cands:
            raise FileNotFoundError(png_path + "*.png")
        img = extract_wide_left_panel(cands[0])
        vec = np.asarray(backend.encode(img, img), dtype=np.float64)
        sims = {w: F75._cos(vec, protos[w]) for w in TARGET_WORDS}
        nearest = max(sims.items(), key=lambda kv: kv[1])
        rows.append({"frame": fr, "vec": vec, "sims": sims, "nearest": nearest, "png": cands[0]})
        images_for_grid.append((fr, img))
        print("  idx=%-3d t=%5.1fs %-16s 最近傍=%-6s(%.3f)  くつ=%.3f"
              % (fr["idx"], fr["sim_time"], fr["label"], nearest[0], nearest[1], sims["くつ"]))

    print("\n[4] GRU生成（本体・海馬なし・hidden=None）と先頭音top-3")
    t = F71.build_taro(MODEL_PATH)
    gen_results = []
    for r in rows:
        fr = r["frame"]
        word, top3 = F75.generate_with_top3(t, r["vec"])
        gen_results.append({"frame": fr, "word": word, "top3": top3})
        top3_str = " / ".join("%s:%.3f" % (c, p) for c, p in top3)
        print("  idx=%-3d %-16s 生成=%-10r 先頭音top3= %s" % (fr["idx"], fr["label"], word, top3_str))

    print("\n[5] 出力: JSON・CSV")
    result = {"model": MODEL_PATH, "target_words": TARGET_WORDS, "frames": []}
    csv_rows = []
    for r, g in zip(rows, gen_results):
        fr = r["frame"]
        entry = {
            "idx": fr["idx"], "step": fr["step"], "sim_time": fr["sim_time"],
            "label": fr["label"], "expect": fr["expect"], "word_csv（走行時の実際の発話）": fr["word_csv"],
            "sims": {w: round(v, 5) for w, v in r["sims"].items()},
            "nearest_word": r["nearest"][0], "nearest_sim": round(r["nearest"][1], 5),
            "kutsu_sim": round(r["sims"]["くつ"], 5),
            "generated": g["word"], "top3_first_char": g["top3"],
            "元画像": r["png"],
        }
        result["frames"].append(entry)
        row = {"idx": fr["idx"], "step": fr["step"], "sim_time": fr["sim_time"], "label": fr["label"],
               "expect": fr["expect"], "word_csv": fr["word_csv"],
               "nearest_word": entry["nearest_word"], "nearest_sim": entry["nearest_sim"],
               "kutsu_sim": entry["kutsu_sim"], "generated": entry["generated"],
               "top3_first_char": " / ".join("%s:%.3f" % (c, p) for c, p in g["top3"])}
        for w in TARGET_WORDS:
            row["sim_" + w] = entry["sims"][w]
        csv_rows.append(row)

    jpath = os.path.join(OUT, "結果.json")
    with io.open(jpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("生データ:", jpath)

    cpath = os.path.join(OUT, "表.csv")
    fieldnames = ["idx", "step", "sim_time", "label", "expect", "word_csv", "nearest_word",
                  "nearest_sim", "kutsu_sim", "generated", "top3_first_char"] + ["sim_" + w for w in TARGET_WORDS]
    with io.open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        wtr.writerows(csv_rows)
    print("表:", cpath)

    print("\n[6] 図: ヒートマップ")
    labels = ["%s\nidx%d t=%.1fs\n実際=%s" % (fr["label"], fr["idx"], fr["sim_time"], fr["word_csv"]) for fr in FRAMES]
    mat = np.array([[r["sims"][w] for w in TARGET_WORDS] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 6.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(TARGET_WORDS))); ax.set_xticklabels(TARGET_WORDS, fontsize=10)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "%.2f" % mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="コサイン類似度")
    ax.set_title("発話瞬間の実画像（produce_snapshot・周辺60度・無クロップ）× 語彙8語 の類似度", fontsize=11)
    fig.tight_layout()
    hpath = os.path.join(OUT, "図_類似度.png")
    fig.savefig(hpath, dpi=120)
    print("図:", hpath)

    print("[7] 図: 使った画像7枚")
    fig2, axes = plt.subplots(1, len(FRAMES), figsize=(3.0 * len(FRAMES), 3.4))
    for ax, (fr, img) in zip(axes, images_for_grid):
        ax.imshow(img)
        ax.set_title("%s\nidx%d t=%.1fs\n実際=%s" % (fr["label"], fr["idx"], fr["sim_time"], fr["word_csv"]), fontsize=9)
        ax.axis("off")
    fig2.suptitle("発話瞬間プローブに使った7枚（trainerが実際に符号化した画像＝周辺60度・無クロップ）", fontsize=11)
    fig2.tight_layout()
    gpath = os.path.join(OUT, "図_使った画像.png")
    fig2.savefig(gpath, dpi=120)
    print("図:", gpath)

    print("\n[8] 文脈再生（条件A相当・8.5秒窓）を、この走行(F2-74c)自身のイベントで再構成し、"
          "『今この瞬間の視覚』だけをproduce_snapshotの実画像（無クロップ）に差し替えて1回試す。"
          "文脈側（親発話に紐づく過去の視覚）はf76と同じ近似＝検出フレームを使う"
          "（過去分は物体ファイル.csvのsim_time最近傍で選ぶしかなく、produce_snapshotは"
          "『発話イベント（自分の発話）』にしか紐付いていないため、親発話の視覚は取れない）。")
    import f76_context_replay_probe as F76
    F76.RUN_DIR = RUN_DIR
    F76.FRAMES_DIR = os.path.join(RUN_DIR, "検出フレーム")
    all_events = F76.load_events()
    frame_index = F76.load_frame_index()
    before = [e for e in all_events if e["sim_sec"] < 8.5]
    hidden_A = F76.build_context(t, backend, frame_index, before)
    empty_vec_real = rows[0]["vec"]  # idx=5, t=8.5s, produce_snapshotの実画像で符号化したベクトル
    word_A, top3_A = F76.generate_from_context(t, empty_vec_real, hidden_A)
    top3_A_str = " / ".join("%s:%.3f" % (c, p) for c, p in top3_A)
    print("  window=8.5s 条件A（文脈=検出フレーム近似, 視覚=produce_snapshot実画像）"
          " 生成=%r top3= %s" % (word_A, top3_A_str))
    result["context_replay_A_window8.5"] = {
        "note": "文脈（親発話の視覚キー）はf76と同じ検出フレーム近似。今この瞬間の視覚だけが"
                "produce_snapshotの実画像（周辺60度・無クロップ）に置き換わっている。",
        "n_ctx_events": len(before), "generated": word_A, "top3_first_char": top3_A,
    }
    with io.open(jpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n[OK] 完了")


if __name__ == "__main__":
    main()
