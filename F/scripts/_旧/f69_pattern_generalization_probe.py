# -*- coding: utf-8 -*-
"""F2-69 型の応用テスト（2026-09-04夜）。

仕様：F/docs/二語文/仕様_型の応用テスト_2026-09-04.md

問い：太郎の発話の「本体」（GRU自力、run/trainer.py の `_wc == "gru_hippo"` ブロック）は、
「くつ ない」「バス ない」「コップ ない」を listen_learn（既定OFF）で聞かせたあと、
一度も聞かせていない「おわん」にも自分で「ない」を当てはめて言おうとするか（型の応用）。

このスクリプトは走行ループ（run/main.py）を使わない。脳のオブジェクト
（TaroBrain・VisualProjection・語彙）をチェックポイントから直接組み立てて操作する。
触ってよいのはこのファイルだけ：run/trainer.py・lexicon.py・run/config.py・
recurrent_core.py・既存プラグインは変更しない（「本体」生成ロジックと listen_learn
学習ロジックは run/trainer.py からこのファイルへコピーして再現している。出典は各関数の
docstringに明記）。

    .venv/Scripts/python.exe F/scripts/f69_pattern_generalization_probe.py
    .venv/Scripts/python.exe F/scripts/f69_pattern_generalization_probe.py --repeats 10
出力: F/logs/F2-69_型の応用テスト/結果.json
      F/logs/F2-69_型の応用テスト/図_訓練前後比較.png
"""
import os, sys, io, json, random, argparse, warnings, itertools
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "run")
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("F/scripts"))

import numpy as np
import torch
import torch.nn.functional as Fnn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 【手本・仕様書「後半」節】実物8語の描画・角度は f49_test.render_stimuli() をそのまま使う
#   （太郎の左目・本番と同じ提示角度・複数個体・複数角度。ファイル自体は変更しない）。
import f49_test as T
from cerebral_cortex.recurrent_core import TaroBrain
from cerebral_cortex.visual_projection import VisualProjection
from hearing import Vocabulary, expand_long_vowel
from senses.vision_backends import get_backend

MODEL_PATH = "F/models/F2-49c_r3_seed93_2026-09-03.pt"
OUT = "F/logs/F2-69_型の応用テスト"

TEST_WORD = "おわん"          # 一度も「◯◯ない」を聞かせない対象（型の応用の的）
POS_CONTROL = "くつ"          # 正の対照：訓練にも使う語
NEG_CONTROL = "がおー"        # 負の対照：訓練にもテストにも使わない語
TRAIN_WORDS = ["くつ", "バス", "コップ"]     # 「◯◯ない」を聞かせる3語
MEASURE_WORDS = [TEST_WORD, POS_CONTROL, NEG_CONTROL]

DEFAULT_REPEATS = 20
LISTEN_LR = 0.001            # チェックポイントの訓練時と同じ値（F2-49c_r3実験ファイル参照）
LISTEN_GRAD_CLIP = 1.0       # taro_setup.py の既定値と同じ
SEED = 20260904


# ============================================================================
# ① 脳のオブジェクトを直接組み立てる（run/main.py は使わない）
# ============================================================================
class Taro69:
    """太郎の脳のうち、このテストに要る部分だけを持つ入れ物。
    run/taro_setup.py の Taro クラス相当だが、mujoco環境なしで作れる最小構成
    （仕様書「実行環境の構築」節：TaroBrain・visual_projection・produce_vocab等を直接生成）。
    """
    pass


def build_taro(path):
    """チェックポイント(F2-49c系)から本体（TaroBrain）・視覚投射・語彙を復元する。

    読み込みの手順は F/scripts/f49_test.py の load_model()・
    F/scripts/f68_word_expectation_probe.py と同じ（手本）。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pv = Vocabulary()
    pv.char2idx = dict(blob["brain_vocab"]["char2idx"])
    pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}
    pv.size = max(pv.idx2char) + 1

    brain = TaroBrain(vocab_size=3)
    brain.resize_embedding(blob["brain"]["embedding.weight"].shape[0])
    # 形が合う層だけ読み込む（f49_test.load_modelと同じ、strict=False）
    brain.load_state_dict(
        {k: v for k, v in blob["brain"].items()
         if k in brain.state_dict() and brain.state_dict()[k].shape == v.shape},
        strict=False)
    brain.set_vocab_mapping(pv.char2idx)

    vp = VisualProjection(384, brain.embedding.embedding_dim)
    vp.load_state_dict(blob["visual_projection"])

    t = Taro69()
    t.brain = brain
    t.produce_vocab = pv
    t._visual_projection = vp
    t._context_hidden = None
    t._context_last_target = None
    if "<PARENT>" not in pv.char2idx:
        raise ValueError(
            "チェックポイントに<PARENT>トークンが無い（produce.context=falseで訓練された"
            "モデルの可能性。F2-49c系はcontext=true・listen_learn=trueで訓練済みのはず）。")
    t._context_speaker_ids = {"parent": pv.char2idx["<PARENT>"],
                               "self": pv.char2idx.get("<SELF>")}
    # 【仕様書「実行環境の構築」手順2】listen_learnは既定Falseから明示的にONへ変える。
    t._listen_learn = False
    t._listen_optimizer = None
    t._listen_grad_clip = LISTEN_GRAD_CLIP
    return t


def enable_listen_learn(t, lr=LISTEN_LR):
    """listen_learn を使うためのoptimizerを用意する（仕様書「実行環境の構築」手順3）。

    出典：run/taro_setup.py の `_setup_produce` 内、`if taro._listen_learn:` ブロック
    （539〜552行付近、2026-09-04時点。学習対象は embedding・GRU・知覚ヘッド・視覚投射で
    trainer.py と同一）。このファイルはtaro_setup.pyを変更せず、ここで再現するだけ。
    """
    t._listen_learn = True
    extra = list(t._visual_projection.parameters())
    params = itertools.chain(t.brain.embedding.parameters(), t.brain.gru.parameters(),
                              t.brain.perception_head.parameters(), extra)
    t._listen_optimizer = torch.optim.Adam(params, lr=lr)


def _ensure_brain_capacity(t):
    """脳の名簿(produce_vocab)が入力口(embedding)を追い越していたら伸ばす。

    出典：run/trainer.py の `_ensure_brain_capacity`（コピー・再現。このテストで使う
    全文字は事前確認済みでチェックポイントの語彙(85語)に収まっているため、実際には
    一度も発火しない想定の安全網）。
    """
    pv = t.produce_vocab
    if pv.size <= t.brain.embedding.num_embeddings:
        return
    t.brain.resize_embedding(pv.size)
    t.brain.set_vocab_mapping(pv.char2idx)
    if getattr(t, "_listen_optimizer", None) is not None:
        extra = (list(t._visual_projection.parameters())
                 if getattr(t, "_visual_projection", None) is not None else [])
        params = itertools.chain(t.brain.embedding.parameters(), t.brain.gru.parameters(),
                                  t.brain.perception_head.parameters(), extra)
        lr = t._listen_optimizer.param_groups[0]["lr"]
        t._listen_optimizer = torch.optim.Adam(params, lr=lr)


def listen_learn_step(t, text, vision_vec):
    """聞く学習の1ステップ（出典：run/trainer.py `_context_feed` 内、
    `getattr(t, "_listen_learn", False)` ブロックのコピー・再現。
    cross_entropy・backward・optimizer.step のみ。海馬(language_hippocampus)への
    書き込みは対象外＝仕様書前半「太郎の発話の仕組みのうち…」節、本体（GRU）だけを見る）。

    speaker は "parent" 固定（親が「くつ ない」等を言い聞かせる場面）。
    """
    dev = t.brain._device()
    ids = t.produce_vocab.encode(expand_long_vowel(text))
    _ensure_brain_capacity(t)
    cap = t.brain.embedding.num_embeddings
    full_ids = [t._context_speaker_ids["parent"]] + [int(i) for i in ids if int(i) < cap]
    if full_ids[0] >= cap or len(full_ids) < 2:
        return None
    vin = (torch.tensor(list(vision_vec), dtype=torch.float32, device=dev)
           if vision_vec is not None else None)
    xin = torch.tensor([full_ids[:-1]], dtype=torch.long, device=dev)
    tgt = torch.tensor([full_ids[1:]], dtype=torch.long, device=dev)
    pfx = t._visual_projection(vin) if vin is not None else None
    out, _ = t.brain.forward_hidden(xin, hidden=t._context_hidden, prefix_vec=pfx)
    if pfx is not None:
        out = out[:, 1:, :]          # 先頭＝視覚トークン位置の出力は損失に使わない
    loss = Fnn.cross_entropy(t.brain.perception_head(out)[0], tgt[0])
    t._listen_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for g in t._listen_optimizer.param_groups for p in g["params"]),
        t._listen_grad_clip)
    t._listen_optimizer.step()
    # 文脈hiddenの更新（no_grad、_context_feedと同じ）
    with torch.no_grad():
        x = torch.tensor([full_ids], dtype=torch.long, device=dev)
        pfx2 = t._visual_projection(vin) if vin is not None else None
        _, h = t.brain.forward_hidden(x, hidden=t._context_hidden, prefix_vec=pfx2)
    t._context_hidden = h.detach()
    return float(loss.item())


def generate_probe(t, vec, max_length=8):
    """「本体（GRU自力）」生成（出典：run/trainer.py `_apply_word_production` の
    `_wc == "gru_hippo"` ブロックのうち、本体側だけをコピー・再現。海馬の想起との
    比較は対象外＝仕様書前半の対象外規定）。hidden=Noneで毎回独立に測る
    （F/scripts/f49_test.py の greedy() と同じ流儀＝手本）。

    生成文字列に加え、生成過程の各ステップのsoftmaxで「な」「い」がそれぞれ
    獲得した確率の最大値（ピーク）を返す。
    """
    pv = t.produce_vocab
    par = pv.char2idx["<PARENT>"]
    dev = t.brain._device()
    idx_na = pv.char2idx.get("な")
    idx_i = pv.char2idx.get("い")
    max_na, max_i = 0.0, 0.0

    def track(logits):
        nonlocal max_na, max_i
        p = torch.softmax(logits, dim=-1)
        if idx_na is not None:
            max_na = max(max_na, float(p[idx_na]))
        if idx_i is not None:
            max_i = max(max_i, float(p[idx_i]))

    vin = torch.tensor(list(vec), dtype=torch.float32, device=dev)
    with torch.no_grad():
        key = t._visual_projection(vin)
        out, hh = t.brain.forward_hidden(
            torch.tensor([[par]], dtype=torch.long, device=dev),
            hidden=None, prefix_vec=key)
        logits = t.brain.perception_head(out)[0, -1]
        track(logits)
        seq = []
        for _ in range(max_length):
            tk = int(torch.argmax(logits))
            if tk == 2:
                break
            seq.append(tk)
            out, hh = t.brain.forward_hidden(
                torch.tensor([[tk]], dtype=torch.long, device=dev), hidden=hh)
            logits = t.brain.perception_head(out)[0, -1]
            track(logits)
    word = pv.decode(seq) if seq else ""
    return word, max_na, max_i


# ============================================================================
# ② 刺激（画像・視覚ベクトル）の用意
# ============================================================================
def build_stimuli():
    """8語の刺激画像を作り（f49_test.render_stimuli()の流用）、要る5語
    （おわん・くつ・がおー・バス・コップ）だけを視覚ベクトル化して返す。

    戻り値: {語: [(vec(384,), meta_dict), ...]}  各語6枚（2個体×3角度）
    """
    need = set(MEASURE_WORDS) | set(TRAIN_WORDS)
    print("[刺激] render_stimuli() 実行中（本番と同じ提示角度・太郎の左目224px）…", flush=True)
    stim = T.render_stimuli()
    stim = [s for s in stim if s[0] in need]
    print(f"[刺激] {len(stim)}枚（対象語: {sorted(need)}）", flush=True)
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    # 【視覚キーの確認・仕様書「検証・止まる条件」節】lexicon_vision.source="wide"の
    #   ときの本番経路（run/trainer.py _vision_backend_encode）は fovea_px を一時的に
    #   10**9へ上書きして eye_left/eye_right をそのまま渡す＝ここと同じ手順。
    #   F2-49c_r3のlexicon_vision.eyeは"left"だが、単眼画像しか無いのでencode(img,img)
    #   （左右に同じ画像）にする。eye="both"（既定）で平均しても同じ画像同士の平均は
    #   自分自身と一致するため、結果はeye="left"と数値的に同一。
    out = {}
    for word, ind, unseen, yaw, img, name in stim:
        vec = np.asarray(be.encode(img, img), dtype=np.float32)
        out.setdefault(word, []).append(
            (vec, {"individual": ind, "unseen": unseen, "yaw": yaw, "material": name}))
    print("[視覚キー確認] 例（くつ・個体%d・横%d度）: 先頭5次元 %s  ノルム %.4f"
          % (out[POS_CONTROL][0][1]["individual"], out[POS_CONTROL][0][1]["yaw"],
             np.round(out[POS_CONTROL][0][0][:5], 4).tolist(),
             float(np.linalg.norm(out[POS_CONTROL][0][0]))), flush=True)
    for w in need:
        if w not in out:
            raise ValueError(f"語「{w}」の刺激画像が render_stimuli() から得られなかった")
    return out


# ============================================================================
# ③ 測定
# ============================================================================
def measure(t, stimuli, words, phase):
    records = []
    for w in words:
        for vec, meta in stimuli[w]:
            said, na, ii = generate_probe(t, vec)
            records.append({"phase": phase, "word": w, "generated": said,
                             "na_max": round(na, 5), "i_max": round(ii, 5), **meta})
    return records


def summarize(records):
    """語×段階 で な最大・い最大の平均と、生成文字列の内訳を作る。"""
    summary = {}
    for r in records:
        key = (r["word"], r["phase"])
        summary.setdefault(key, {"na": [], "i": [], "said": []})
        summary[key]["na"].append(r["na_max"])
        summary[key]["i"].append(r["i_max"])
        summary[key]["said"].append(r["generated"])
    out = {}
    for (w, phase), d in summary.items():
        from collections import Counter
        out[(w, phase)] = {
            "na_mean": float(np.mean(d["na"])), "na_max": float(np.max(d["na"])),
            "i_mean": float(np.mean(d["i"])), "i_max": float(np.max(d["i"])),
            "said_counts": dict(Counter(d["said"])), "n": len(d["na"])}
    return out


# ============================================================================
# ④ 訓練（listen_learn）
# ============================================================================
def train_pattern(t, stimuli, train_words, n_repeats, rng):
    """「◯◯ ない」を対応する画像を見せながら listen_learn で聞かせる
    （仕様書「手順」節②。3語をランダムな順で混ぜる＝1語だけ集中的にやらない）。
    """
    order = []
    for w in train_words:
        order += [w] * n_repeats
    rng.shuffle(order)
    losses = []
    last_target = None
    for i, w in enumerate(order):
        # 【文脈・場面の切り替わり】出典：run/trainer.py _hear_parent_utterance
        #   「target が変わったら場面の切り替わりとみなして文脈をリセット」と同じ規則。
        if w != last_target:
            t._context_hidden = None
            last_target = w
        vec, meta = rng.choice(stimuli[w])     # 個体・角度は毎回ランダムに変える
        loss = listen_learn_step(t, w + "ない", vec)
        losses.append({"step": i, "word": w, "loss": loss,
                        "individual": meta["individual"], "yaw": meta["yaw"]})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [訓練] {i + 1}/{len(order)}  語={w}  loss={loss:.4f}", flush=True)
    return losses


# ============================================================================
# ⑤ 出力
# ============================================================================
def _setup_font():
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()


def make_plot(summary_before, summary_after, words, path):
    _setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    x = np.arange(len(words))
    width = 0.35
    for ax, key, title in ((axes[0], "na_mean", "「な」の最大確率（softmaxピークの平均）"),
                            (axes[1], "i_mean", "「い」の最大確率（softmaxピークの平均）")):
        before = [summary_before[w][key] for w in words]
        after = [summary_after[w][key] for w in words]
        ax.bar(x - width / 2, before, width, label="訓練前", color="#a0aec0")
        ax.bar(x + width / 2, after, width, label="訓練後", color="#2b6cb0")
        ax.set_xticks(x); ax.set_xticklabels(words, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.0); ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle("F2-69 型の応用テスト：「◯◯ ない」を聞いた後、"
                  "一度も聞かせていない「おわん」にも「ない」を当てはめるか", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("図:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=str, default=OUT,
                     help="既定は本番出力先(結果.json等)。追試(10回/40回)はサブフォルダを指定して"
                          "本番結果を上書きしない。")
    args = ap.parse_args()
    out_dir = args.outdir

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 74)
    print(f" F2-69 型の応用テスト  repeats={args.repeats}  seed={args.seed}")
    print("=" * 74)

    print("[1] チェックポイント読み込み:", MODEL_PATH)
    t = build_taro(MODEL_PATH)
    print(f"    語彙サイズ {t.produce_vocab.size}  <PARENT>={t._context_speaker_ids['parent']}"
          f"  な={t.produce_vocab.char2idx.get('な')}  い={t.produce_vocab.char2idx.get('い')}")

    print("[2] 刺激（画像→視覚ベクトル）を用意")
    stimuli = build_stimuli()

    print("[3] 訓練前の測定（おわん・くつ・がおー）")
    t._context_hidden = None
    before = measure(t, stimuli, MEASURE_WORDS, "訓練前")

    print(f"[4] listen_learn を有効化して訓練（各{args.repeats}回、3語混ぜてランダム順）")
    enable_listen_learn(t, lr=LISTEN_LR)
    losses = train_pattern(t, stimuli, TRAIN_WORDS, args.repeats, rng)

    print("[5] 訓練後の測定（おわん・くつ・がおー）")
    t._context_hidden = None
    after = measure(t, stimuli, MEASURE_WORDS, "訓練後")

    print("[6] 集計・出力")
    summary_all = summarize(before + after)
    summary_before = {w: summary_all[(w, "訓練前")] for w in MEASURE_WORDS}
    summary_after = {w: summary_all[(w, "訓練後")] for w in MEASURE_WORDS}

    print("\n%-6s %8s %10s %10s %10s %10s" % ("語", "段階", "な最大平均", "い最大平均", "件数", "生成例"))
    for w in MEASURE_WORDS:
        for phase, s in (("訓練前", summary_before[w]), ("訓練後", summary_after[w])):
            top_said = sorted(s["said_counts"].items(), key=lambda kv: -kv[1])[0][0]
            print("%-6s %8s %10.4f %10.4f %10d   %r(%d/%d)"
                  % (w, phase, s["na_mean"], s["i_mean"], s["n"], top_said,
                     s["said_counts"][top_said], s["n"]))

    result = {
        "model": MODEL_PATH, "repeats": args.repeats, "seed": args.seed,
        "train_words": TRAIN_WORDS, "test_word": TEST_WORD,
        "pos_control": POS_CONTROL, "neg_control": NEG_CONTROL,
        "records_before": before, "records_after": after,
        "summary_before": {w: {k: v for k, v in summary_before[w].items()} for w in MEASURE_WORDS},
        "summary_after": {w: {k: v for k, v in summary_after[w].items()} for w in MEASURE_WORDS},
        "train_losses": losses,
    }
    jpath = os.path.join(out_dir, "結果.json")
    with io.open(jpath, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
    print("生データ:", jpath)

    ppath = os.path.join(out_dir, "図_訓練前後比較.png")
    make_plot(summary_before, summary_after, MEASURE_WORDS, ppath)

    print("\n[判定用の数字（実装側では判断しない）]")
    for w in MEASURE_WORDS:
        d = summary_after[w]["na_mean"] - summary_before[w]["na_mean"]
        print(f"  {w}: な最大平均 訓練前{summary_before[w]['na_mean']:.4f} → "
              f"訓練後{summary_after[w]['na_mean']:.4f}（差 {d:+.4f}）")
    print("[OK] 完了")


if __name__ == "__main__":
    main()
