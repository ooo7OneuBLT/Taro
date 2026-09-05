# -*- coding: utf-8 -*-
"""F2-74 文脈プローブ（2026-09-05・書き捨て診断・読み取り専用）。

F2-74（`F/logs/F2-74_M2_消失発話/`）で、机が空の時間帯（8.5s/18.5s/26.6s/28.6s）に
太郎が毎回「くつだね」と言った。f75（空机プローブ）では文脈なし（hidden=None）＋
空の机の視覚だけを与えると「おさらだよ」になり「くつ」は再現しなかった。
走行は produce.context=true（GRUの隠れ状態が発話をまたいで持続。親の発話と
太郎自身の発話を run/trainer.py `_context_feed` で流し込む）。

このスクリプトは、8.5秒窓・18.5秒窓の2つについて、**走行と同じ文脈（親＋太郎の
発話イベントを実際の順序・話者印・視覚キーで流し込んだGRUの隠れ状態）を
再生**し、そこから「本体（GRU自力）」で発話を生成させ、「くつだね」が
再現するかを条件A〜Gで比べる。学習・保存・海馬書き込みは一切しない。

【前提・出典】
  _context_feed の再現：run/trainer.py 336〜420行（2026-09-05時点）を読んだ結果、
  F2-74の設定（produce.listen_learn=false・produce.listen_eos=true）では
    - 聞く学習ブロック（376〜415行）は listen_learn=false のため一歩も通らない
      （＝本体への勾配更新も海馬書き込みも起きない。このスクリプトが再現するのは
      最後の no_grad ブロック（416〜420行）だけでよい）
    - ids = [speaker_id] + encode(text)[:cap] に、speaker!="self" のとき
      （listen_eos=true なので）EOS(id=2) が足される（358〜359行）
    - 視覚（vision_vec）は _visual_projection で64次元に変換して prefix_vec として
      forward_hidden に渡す（V1・367〜369・417行）
  文脈のリセット：run/trainer.py 546〜554行（_hear_parent_utterance内）。
  親の発話のtarget（見せている物）が直前の親発話と変わったら
  t._context_hidden=None にしてから feed する（「場面ごと」決定1）。
  自分の発話（_context_feed("self", ...)）はリセット判定に関与しない
  （target比較は親発話でしか行われないコード＝636行 self._context_feed("self", generated)
  はvision_vecを渡していない。よってこのスクリプトも自己発話にはvision_vecを渡さない）。

  発話時の生成（本体・GRU自力）：run/trainer.py 699〜722行
  （_apply_word_production の word_choice=="gru_hippo" ブロックの本体側）を
  そのまま再現。実走行では hidden=t._context_hidden から生成が始まる
  （2026-09-04配線・693〜698行のコメント参照）。f75の generate_with_top3 は
  常に hidden=None だった（f75はF2-74の走行時の文脈を再現していなかった）。

【近似・限界（f75から引き継ぎ）】
  - 視覚エンコードは source="wide" なので fovea_px は無効化される
    （run/trainer.py 264-294行。f75と同じ扱い）。
  - 検出フレームPNG（224x224、PIL resize済み）を使う。trainer本体が使う生の
    eye_left画像とはリサイズアルゴリズムが違う可能性がある（f75と同じ近似）。
  - backend.encode(img_l, img_r) は本来左右目の画像を別々に渡すが、
    検出フレームは左目1枚しか保存されていないため img, img で代用する
    （f75と同じ近似）。
  - 親発話の視覚キーに使うフレームは、発話イベント.csvのsim_secと
    物体ファイル.csvのsim_time列が最も近い検出フレームを機械的に選ぶ
    （両CSVの「step」列は時計が異なる＝f75で発覚済みのため、step同士を
    直接比較せずsim_time同士で最近傍を取る）。
  - 太郎自身の発話は、実際にGRUが生成した生のトークンID列ではなく、
    太郎の発話.csvのgenerated_word列（デコード後の文字列）を
    produce_vocab.encode()で再エンコードしたものを使う（encode/decodeが
    往復可能という前提の近似）。

    .venv/Scripts/python.exe F/scripts/f76_context_replay_probe.py
出力: F/logs/F2-74_M2_消失発話/文脈プローブ/結果.json
      F/logs/F2-74_M2_消失発話/文脈プローブ/表.csv
      F/logs/F2-74_M2_消失発話/文脈プローブ/図_条件別_先頭音.png
"""
import os, sys, io, csv, json, warnings
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
from senses.hearing import expand_long_vowel
import f71_pattern_generalization_disambiguation as F71   # build_taro をそのまま流用

MODEL_PATH = "F/models/F2-50_r8_seed98_2026-09-03.pt"
RUN_DIR = "F/logs/F2-74_M2_消失発話"
FRAMES_DIR = os.path.join(RUN_DIR, "検出フレーム")
OUT = os.path.join(RUN_DIR, "文脈プローブ")

# 【近似・f75から引き継ぎ】f75で目視確認済みの「空の机」フレーム。
#   sim_time最近傍でも同じstepが選ばれることを下のnearest_frame_step()呼び出し後に検証する。
EMPTY_FRAME_BY_WINDOW = {8.5: 77, 18.5: 179}
# 【近似】f75で目視確認済みの「直前の物」フレーム（条件Bで使う）。
PREV_OBJECT_FRAME_BY_WINDOW = {8.5: 45, 18.5: 137}   # おわん(4.5-5.0s) / がおー(14.2s)

PROBE_WINDOWS = [8.5, 18.5]


# ============================================================================
# ① イベント読み込み（親の発話イベント.csv＋太郎の発話.csvを時刻順にマージ）
# ============================================================================
def load_events():
    events = []
    with io.open(os.path.join(RUN_DIR, "太郎の発話.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            events.append({
                "step": int(row["step"]), "sim_sec": float(row["sim_sec"]),
                "speaker": "self", "text": row["generated_word"],
                "target": None, "cause": None,
            })
    with io.open(os.path.join(RUN_DIR, "発話イベント.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            events.append({
                "step": int(row["step"]), "sim_sec": float(row["sim_sec"]),
                "speaker": "parent", "text": row["text"],
                "target": row["target"] or None, "cause": row["cause"] or None,
            })
    events.sort(key=lambda e: e["step"])
    return events


def load_frame_index():
    """物体ファイル.csv の (step, sim_time) 一覧（重複stepは1つに畳む）。"""
    rows = {}
    with io.open(os.path.join(RUN_DIR, "物体ファイル.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s = int(row["step"])
            rows.setdefault(s, float(row["sim_time"]))
    return sorted(rows.items())


def nearest_frame_step(frame_index, sim_sec):
    """sim_time が一番近いフレームのstepを返す（同点はstepが小さい方）。"""
    return min(frame_index, key=lambda st: (abs(st[1] - sim_sec), st[0]))[0]


# ============================================================================
# ② 視覚エンコード（f75と同じ経路：source="wide"相当＝fovea_px=10**9でno-op）
# ============================================================================
_frame_cache = {}

def encode_frame(backend, step):
    if step in _frame_cache:
        return _frame_cache[step]
    path = os.path.join(FRAMES_DIR, "frame_%05d.png" % step)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = np.array(Image.open(path).convert("RGB"))
    vec = np.asarray(backend.encode(img, img), dtype=np.float64)
    _frame_cache[step] = vec
    return vec


# ============================================================================
# ③ _context_feed の再現（run/trainer.py 336-420行。listen_learn=false のため
#    聞く学習ブロックは無く、最後のno_gradブロックだけを再現すればよい）
# ============================================================================
def context_feed(t, speaker, token_ids, vision_vec=None):
    cap = t.brain.embedding.num_embeddings
    spk_id = t._context_speaker_ids[speaker]
    ids = [spk_id] + [int(i) for i in token_ids if int(i) < cap]
    # 【F2-74の設定固定・listen_eos=true】speaker!="self"なら親発話にEOSを足す
    #   （run/trainer.py 358-359行）。このスクリプトはF2-74の設定だけを再現する
    #   ので listen_eos の分岐は持たず常にこの通りにする。
    if speaker != "self":
        ids = ids + [2]
    if spk_id is None or spk_id >= cap:
        return
    dev = t.brain._device()
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    vin = None
    if vision_vec is not None:
        vin = torch.tensor(list(vision_vec), dtype=torch.float32, device=dev)
    with torch.no_grad():
        pfx = t._visual_projection(vin) if vin is not None else None
        _, h = t.brain.forward_hidden(x, hidden=t._context_hidden, prefix_vec=pfx)
    t._context_hidden = h.detach()


def build_context(t, backend, frame_index, events):
    """eventsを時刻順に_context_feedへ流し、最終的なt._context_hiddenを返す。

    親発話でtargetが直前の親発話と変わったらリセット（run/trainer.py 546-550行）。
    自分の発話はリセット判定に関与しない・視覚も渡さない（835行）。
    """
    t._context_hidden = None
    last_target = "__NONE__"
    first_parent = True
    for ev in events:
        if ev["speaker"] == "parent":
            if not first_parent and ev["target"] != last_target:
                t._context_hidden = None
            last_target = ev["target"]
            first_parent = False
            fstep = nearest_frame_step(frame_index, ev["sim_sec"])
            vv = encode_frame(backend, fstep)
            ids = t.produce_vocab.encode(expand_long_vowel(ev["text"]))
            context_feed(t, "parent", ids, vision_vec=vv)
        else:
            ids = t.produce_vocab.encode(expand_long_vowel(ev["text"]))
            context_feed(t, "self", ids, vision_vec=None)
    return t._context_hidden


# ============================================================================
# ④ 生成（run/trainer.py 699-722行の本体側をそのまま再現。hiddenを外から渡せる
#    ようにした点だけがf71 generate_probe（常にhidden=None）との違い）
# ============================================================================
def generate_from_context(t, vec, hidden, max_length=8):
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    dev = t.brain._device()
    vin = torch.tensor(list(vec), dtype=torch.float32, device=dev)
    with torch.no_grad():
        key = t._visual_projection(vin)
        out, hh = t.brain.forward_hidden(
            torch.tensor([[par]], dtype=torch.long, device=dev),
            hidden=hidden, prefix_vec=key)
        logits = t.brain.perception_head(out)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top3_idx = torch.topk(probs, min(3, probs.shape[-1])).indices.tolist()
        top3 = [(pv.idx2char.get(i, "?"), round(float(probs[i]), 5)) for i in top3_idx]
        seq = []
        for _ in range(max_length):
            tk = int(torch.argmax(logits))
            if tk == 2:
                break
            seq.append(tk)
            out, hh = t.brain.forward_hidden(
                torch.tensor([[tk]], dtype=torch.long, device=dev), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
    word = pv.decode(seq) if seq else ""
    return word, top3


# ============================================================================
# ⑤ 条件A〜Gの組み立て
# ============================================================================
def build_conditions(all_events, window):
    """windowより前の全イベント（時刻順）から、条件A〜Gそれぞれの
    「context_feedへ流すイベント列」を返す。条件Bは条件Aと同じhiddenを使う
    （生成時のvecだけ差し替える）ので、ここではA用のイベント列だけを返す。
    """
    before = [e for e in all_events if e["sim_sec"] < window]
    parent_only = [e for e in before if e["speaker"] == "parent"]
    self_only = [e for e in before if e["speaker"] == "self"]
    last_only = before[-1:]
    drop_vanish = before
    note_g = None
    if before and before[-1]["speaker"] == "parent" and before[-1]["cause"] == "vanish":
        drop_vanish = before[:-1]
    else:
        note_g = "直前イベントが「ないね」の親発話ではなかった（想定外）"
    return {
        "A": before, "D": parent_only, "E": self_only, "F": last_only, "G": drop_vanish,
    }, note_g, before


def main():
    os.makedirs(OUT, exist_ok=True)
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()

    print("[1] チェックポイント読み込み:", MODEL_PATH)
    t = F71.build_taro(MODEL_PATH)
    print("    語彙サイズ", t.produce_vocab.size, " <PARENT>=", t._context_speaker_ids["parent"],
          " <SELF>=", t._context_speaker_ids["self"])

    print("[2] 視覚バックエンド構築（dinov2_vits14, fovea_px=10**9=ノークロップ, f75と同じ）")
    backend = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9, "eye": "left"})

    print("[3] イベント読み込み")
    all_events = load_events()
    frame_index = load_frame_index()
    for e in all_events:
        print("  step=%-4d t=%5.1fs %-6s %-10s target=%s cause=%s"
              % (e["step"], e["sim_sec"], e["speaker"], e["text"], e["target"], e["cause"]))

    print("\n[4] 条件A〜G構築・生成")
    result = {"model": MODEL_PATH, "windows": []}
    csv_rows = []
    plot_data = {}   # (window, cond) -> top3

    for window in PROBE_WINDOWS:
        empty_step = EMPTY_FRAME_BY_WINDOW[window]
        prev_step = PREV_OBJECT_FRAME_BY_WINDOW[window]
        # f75で使った空机フレームが、sim_time最近傍選定でも選ばれるかを確認
        chk = nearest_frame_step(frame_index, window)
        note_frame = None
        if chk != empty_step:
            note_frame = ("sim_time最近傍で選んだstepはEMPTY_FRAME_BY_WINDOW固定値と食い違う: "
                           "nearest=%d, fixed=%d" % (chk, empty_step))
            print("  [警告]", note_frame)

        conds_events, note_g, before = build_conditions(all_events, window)
        empty_vec = encode_frame(backend, empty_step)
        prev_vec = encode_frame(backend, prev_step)

        window_entry = {"window_sec": window, "note_frame": note_frame,
                         "note_g": note_g, "n_events_before": len(before),
                         "conditions": {}}

        # A: 走行と同じ順で全発話 + 空の机
        hidden_A = build_context(t, backend, frame_index, conds_events["A"])
        word_A, top3_A = generate_from_context(t, empty_vec, hidden_A)
        window_entry["conditions"]["A"] = {"generated": word_A, "top3": top3_A,
                                            "n_events": len(conds_events["A"])}

        # B: Aと同じ文脈 + 直前の物の視覚
        word_B, top3_B = generate_from_context(t, prev_vec, hidden_A)
        window_entry["conditions"]["B"] = {"generated": word_B, "top3": top3_B,
                                            "n_events": len(conds_events["A"])}

        # C: 文脈なし + 空の机
        word_C, top3_C = generate_from_context(t, empty_vec, None)
        window_entry["conditions"]["C"] = {"generated": word_C, "top3": top3_C, "n_events": 0}

        # D: 親の発話だけ + 空の机
        hidden_D = build_context(t, backend, frame_index, conds_events["D"])
        word_D, top3_D = generate_from_context(t, empty_vec, hidden_D)
        window_entry["conditions"]["D"] = {"generated": word_D, "top3": top3_D,
                                            "n_events": len(conds_events["D"])}

        # E: 太郎の発話だけ + 空の机
        hidden_E = build_context(t, backend, frame_index, conds_events["E"])
        word_E, top3_E = generate_from_context(t, empty_vec, hidden_E)
        window_entry["conditions"]["E"] = {"generated": word_E, "top3": top3_E,
                                            "n_events": len(conds_events["E"])}

        # F: 直前1発話だけ + 空の机
        hidden_F = build_context(t, backend, frame_index, conds_events["F"])
        word_F, top3_F = generate_from_context(t, empty_vec, hidden_F)
        window_entry["conditions"]["F"] = {"generated": word_F, "top3": top3_F,
                                            "n_events": len(conds_events["F"]),
                                            "last_event": conds_events["F"]}

        # G: Aから最後の「○○ないね」を除いた文脈 + 空の机
        hidden_G = build_context(t, backend, frame_index, conds_events["G"])
        word_G, top3_G = generate_from_context(t, empty_vec, hidden_G)
        window_entry["conditions"]["G"] = {"generated": word_G, "top3": top3_G,
                                            "n_events": len(conds_events["G"])}

        result["windows"].append(window_entry)

        for cond in ["A", "B", "C", "D", "E", "F", "G"]:
            entry = window_entry["conditions"][cond]
            top3_str = " / ".join("%s:%.3f" % (c, p) for c, p in entry["top3"])
            print("  window=%5.1fs cond=%s n_ctx_events=%-3d 生成=%-10r top3= %s"
                  % (window, cond, entry["n_events"], entry["generated"], top3_str))
            csv_rows.append({
                "window_sec": window, "condition": cond,
                "n_ctx_events": entry["n_events"], "generated": entry["generated"],
                "top3_first_char": top3_str,
                "contains_kutsu": ("くつ" in entry["generated"]),
            })
            plot_data[(window, cond)] = entry["top3"]

    print("\n[5] 出力: JSON・CSV")
    jpath = os.path.join(OUT, "結果.json")
    with io.open(jpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("生データ:", jpath)

    cpath = os.path.join(OUT, "表.csv")
    with io.open(cpath, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["window_sec", "condition", "n_ctx_events",
                                             "generated", "top3_first_char", "contains_kutsu"])
        wtr.writeheader()
        wtr.writerows(csv_rows)
    print("表:", cpath)

    print("[6] 図: 条件別の先頭音top-3")
    conds = ["A", "B", "C", "D", "E", "F", "G"]
    fig, axes = plt.subplots(1, len(PROBE_WINDOWS), figsize=(7.5 * len(PROBE_WINDOWS), 5.0),
                              sharey=True)
    if len(PROBE_WINDOWS) == 1:
        axes = [axes]
    colors = ["#2b6cb0", "#63b3ed", "#bee3f8"]
    for ax, window in zip(axes, PROBE_WINDOWS):
        x = np.arange(len(conds))
        width = 0.25
        for rank in range(3):
            heights, labels = [], []
            for cond in conds:
                top3 = plot_data[(window, cond)]
                if rank < len(top3):
                    heights.append(top3[rank][1])
                    labels.append(top3[rank][0])
                else:
                    heights.append(0.0)
                    labels.append("")
            bars = ax.bar(x + (rank - 1) * width, heights, width,
                           color=colors[rank], label="top%d" % (rank + 1))
            for bar, lab in zip(bars, labels):
                if lab:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            lab, ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n(%s)" % (c, plot_data[(window, c)][0][0]
                                           if plot_data[(window, c)] else "-") for c in conds])
        ax.set_ylim(0, 1.05)
        ax.set_title("窓=%.1fs 各条件の先頭音top3" % window, fontsize=12)
        ax.set_ylabel("確率")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("F2-74 文脈プローブ：条件A〜G × 先頭音top3（F2-50開始モデル）", fontsize=13)
    fig.tight_layout()
    gpath = os.path.join(OUT, "図_条件別_先頭音.png")
    fig.savefig(gpath, dpi=120)
    print("図:", gpath)

    print("\n[OK] 完了")


if __name__ == "__main__":
    main()
