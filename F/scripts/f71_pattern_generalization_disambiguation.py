# -*- coding: utf-8 -*-
"""F2-71 型の応用テストの切り分け（見た目の混同を除いた再検証、2026-09-05）。

仕様：F/docs/二語文/仕様_型の応用テスト_切り分け_2026-09-04.md
前回：F/scripts/f70_pattern_generalization_via_replay.py（対象＝おわん。型の応用らしき
兆候はあったが、名詞を訓練語「コップ」に取り違えた。静止測定で、おわんの見た目が
元々コップに近いこと＝コサイン類似度0.29（訓練したくつ0.16・バス0.07より高い）を確認済み。
これが混同の原因かもしれないという疑いが残った）。

前回との違いは**教える語・テスト対象・正負の対照だけ**（静止測定で互いに視覚的にはっきり
離れていることを確認済みの組み合わせに差し替え：がおー vs バス＝0.12、がおー vs かばん＝0.12、
がおー vs ボール＝0.17。参考：問題を起こしたおわん vs コップ＝0.29）。
モデル復元・視覚エンコード・海馬書き込み・睡眠リプレイ・測定方法（generate_probe/measure/
summarize/build_stimuli/hippo_write_step/write_new_words/sleep_replay）は f70 から
**そのまま流用**（コピー。各関数のdocstringの出典表記もf70のまま維持する）。

    .venv/Scripts/python.exe F/scripts/f71_pattern_generalization_disambiguation.py
    .venv/Scripts/python.exe F/scripts/f71_pattern_generalization_disambiguation.py --replay_passes 10 --outdir F/logs/F2-71_型の応用_切り分け/追試_w1r10
    .venv/Scripts/python.exe F/scripts/f71_pattern_generalization_disambiguation.py --write_repeats 5 --replay_passes 10 --outdir F/logs/F2-71_型の応用_切り分け/追試_w5r10
出力: F/logs/F2-71_型の応用_切り分け/結果.json
      F/logs/F2-71_型の応用_切り分け/図_訓練前後比較.png
      F/logs/F2-71_型の応用_切り分け/比較_f70とf71.json
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

# 【手本・仕様書「手本」節】実物8語の描画・角度は f49_test.render_stimuli() をそのまま
#   使う（f70と同じ）。この import が taro_core/src/brain 配下を sys.path に足す副作用を
#   持つ（scene.py 経由で taro_setup.py が実行される）ため、cerebral_cortex・
#   language_hippocampus の import より先に置く（f70と同じ順）。
import f49_test as T
from cerebral_cortex.recurrent_core import TaroBrain
from cerebral_cortex.visual_projection import VisualProjection
from hearing import Vocabulary, expand_long_vowel
from senses.vision_backends import get_backend
from language_hippocampus.language_hippocampus import LanguageHippocampus

MODEL_PATH = "F/models/F2-49c_r3_seed93_2026-09-03.pt"
OUT = "F/logs/F2-71_型の応用_切り分け"
F70_RESULT = "F/logs/F2-70_型の応用テスト_海馬経由/結果.json"

# ここだけがf70と異なる（仕様書「変更点」節）：視覚的にはっきり離れた組み合わせに差し替え。
TEST_WORD = "がおー"          # 一度も「◯◯ない」を聞かせない対象（型の応用の的）。f70は「おわん」
POS_CONTROL = "バス"          # 正の対照：訓練にも使う語（教えた3語のうち1つ）。f70は「くつ」
NEG_CONTROL = "くつ"          # 負の対照：訓練にもテストにも使わない語。f70は「がおー」
TRAIN_WORDS = ["バス", "かばん", "ボール"]   # 「◯◯ない」を海馬に書き込む3語。f70は["くつ","バス","コップ"]
MEASURE_WORDS = [TEST_WORD, POS_CONTROL, NEG_CONTROL]

LISTEN_LR = 0.001            # チェックポイントの訓練時と同じ値（F2-49c_r3実験ファイル参照）。
                              # 睡眠リプレイ中は後述の通り hippo.sleep_lr に一時的に差し替わる。
LISTEN_GRAD_CLIP = 1.0       # taro_setup.py の既定値と同じ
DEFAULT_WRITE_REPEATS = 1    # 仕様書「訓練条件」：まず各1回で試す
DEFAULT_REPLAY_PASSES = None  # Noneならチェックポイントの既定値（現状1）をそのまま使う
SEED = 20260905


# ============================================================================
# ① 脳のオブジェクトを直接組み立てる（f70 build_taro と完全に同一。そのままコピー）
# ============================================================================
class Taro71:
    """太郎の脳のうち、このテストに要る部分だけを持つ入れ物（f70のTaro70と同型）。"""
    pass


def build_taro(path):
    """チェックポイント(F2-49c系)から本体（TaroBrain）・視覚投射・語彙・言語海馬を復元する。

    f70 `build_taro()` と完全に同一（そのままコピー。出典：run/taro_setup.py
    556〜567行の流儀と同じ）。
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pv = Vocabulary()
    pv.char2idx = dict(blob["brain_vocab"]["char2idx"])
    pv.idx2char = {int(i): c for c, i in pv.char2idx.items()}
    pv.size = max(pv.idx2char) + 1

    brain = TaroBrain(vocab_size=3)
    brain.resize_embedding(blob["brain"]["embedding.weight"].shape[0])
    brain.load_state_dict(
        {k: v for k, v in blob["brain"].items()
         if k in brain.state_dict() and brain.state_dict()[k].shape == v.shape},
        strict=False)
    brain.set_vocab_mapping(pv.char2idx)

    vp = VisualProjection(384, brain.embedding.embedding_dim)
    vp.load_state_dict(blob["visual_projection"])

    if "language_hippocampus" not in blob:
        raise ValueError(
            "チェックポイントに language_hippocampus が無い（産出設定 produce.hippocampus"
            "が有効なモデルが必要。F2-49c_r3はhippocampus有効のはず）。")
    hippo = LanguageHippocampus()
    hippo.load_state_dict(blob["language_hippocampus"])

    t = Taro71()
    t.brain = brain
    t.produce_vocab = pv
    t._visual_projection = vp
    t.language_hippocampus = hippo
    t._context_hidden = None
    t._context_last_target = None
    if "<PARENT>" not in pv.char2idx:
        raise ValueError(
            "チェックポイントに<PARENT>トークンが無い（produce.context=falseで訓練された"
            "モデルの可能性。F2-49c系はcontext=true・listen_learn=trueで訓練済みのはず）。")
    t._context_speaker_ids = {"parent": pv.char2idx["<PARENT>"],
                               "self": pv.char2idx.get("<SELF>")}
    t._listen_learn = False
    t._listen_optimizer = None
    t._listen_grad_clip = LISTEN_GRAD_CLIP
    return t


def enable_listen_learn(t, lr=LISTEN_LR):
    """睡眠リプレイに使う optimizer を用意する（f70 `enable_listen_learn` と同一）。

    出典：run/taro_setup.py の `_setup_produce` 内、`if taro._listen_learn:` ブロック
    （539〜552行付近。学習対象は embedding・GRU・知覚ヘッド・視覚投射で trainer.py と同一）。
    このoptimizerの初期lrは睡眠リプレイ中は hippo.sleep_lr に一時的に差し替えられる
    （sleep_replay関数）ため、ここでのlr自体は使われない瞬間の方が長い。
    """
    t._listen_learn = True
    extra = list(t._visual_projection.parameters())
    params = itertools.chain(t.brain.embedding.parameters(), t.brain.gru.parameters(),
                              t.brain.perception_head.parameters(), extra)
    t._listen_optimizer = torch.optim.Adam(params, lr=lr)


def generate_probe(t, vec, max_length=8):
    """「本体（GRU自力）」生成。f70 `generate_probe` と完全に同一（そのままコピー）。

    出典：run/trainer.py `_apply_word_production` の `_wc == "gru_hippo"` ブロックのうち
    本体側だけを再現。海馬の想起との競合は対象外（仕様書前半の対象外規定）。
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
# ② 刺激（画像・視覚ベクトル）の用意（f70 build_stimuli と完全に同一。そのままコピー）
# ============================================================================
def build_stimuli():
    """8語の刺激画像を作り（f49_test.render_stimuli()の流用）、要る5語
    （がおー・バス・かばん・ボール・くつ）だけを視覚ベクトル化して返す。

    戻り値: {語: [(vec(384,), meta_dict), ...]}  各語6枚（2個体×3角度）
    """
    need = set(MEASURE_WORDS) | set(TRAIN_WORDS)
    print("[刺激] render_stimuli() 実行中（本番と同じ提示角度・太郎の左目224px）…", flush=True)
    stim = T.render_stimuli()
    stim = [s for s in stim if s[0] in need]
    print(f"[刺激] {len(stim)}枚（対象語: {sorted(need)}）", flush=True)
    be = get_backend({"backend": "dinov2_vits14", "fovea_px": 10 ** 9})
    out = {}
    for word, ind, unseen, yaw, img, name in stim:
        vec = np.asarray(be.encode(img, img), dtype=np.float32)
        out.setdefault(word, []).append(
            (vec, {"individual": ind, "unseen": unseen, "yaw": yaw, "material": name}))
    print("[視覚キー確認] 例（%s・個体%d・横%d度）: 先頭5次元 %s  ノルム %.4f"
          % (POS_CONTROL, out[POS_CONTROL][0][1]["individual"], out[POS_CONTROL][0][1]["yaw"],
             np.round(out[POS_CONTROL][0][0][:5], 4).tolist(),
             float(np.linalg.norm(out[POS_CONTROL][0][0]))), flush=True)
    for w in need:
        if w not in out:
            raise ValueError(f"語「{w}」の刺激画像が render_stimuli() から得られなかった")
    return out


# ============================================================================
# ③ 測定（f70 measure/summarize と完全に同一。そのままコピー）
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
    from collections import Counter
    summary = {}
    for r in records:
        key = (r["word"], r["phase"])
        summary.setdefault(key, {"na": [], "i": [], "said": []})
        summary[key]["na"].append(r["na_max"])
        summary[key]["i"].append(r["i_max"])
        summary[key]["said"].append(r["generated"])
    out = {}
    for (w, phase), d in summary.items():
        out[(w, phase)] = {
            "na_mean": float(np.mean(d["na"])), "na_max": float(np.max(d["na"])),
            "i_mean": float(np.mean(d["i"])), "i_max": float(np.max(d["i"])),
            "said_counts": dict(Counter(d["said"])), "n": len(d["na"])}
    return out


# ============================================================================
# ④ 海馬への書き込み（f70 と完全に同一。そのままコピー）
# ============================================================================
def hippo_write_step(t, text, vision_vec, step):
    """1件を海馬へ焼き付けるだけ（本体への勾配更新はしない）。

    出典：run/trainer.py 406〜415行（`_context_feed`内、`language_hippocampus.write`
    呼び出し部分）のうち write() 呼び出しだけを再現。ids[0]=speaker(<PARENT>)・
    ids[1:]=text全体（"バスない"等）のエンコード結果という対応関係も同じにする
    （trainer.py 336〜414行：_context_feedのidsの作り方＝[speaker]+encode(text)）。
    直接のcross_entropy/backward/optimizer.stepは行わない
    （仕様書前半：「海馬のwrite()で書き込む」。本体の学習は睡眠リプレイ側でやる）。
    """
    dev = t.brain._device()
    ids = t.produce_vocab.encode(expand_long_vowel(text))
    cap = t.brain.embedding.num_embeddings
    tokens = [int(i) for i in ids if int(i) < cap]
    speaker = t._context_speaker_ids["parent"]
    vin = torch.tensor(list(vision_vec), dtype=torch.float32, device=dev)
    with torch.no_grad():
        key_vis = t._visual_projection(vin).cpu().numpy()
    t.language_hippocampus.write(key_vis, speaker, tokens, step)
    return tokens


def write_new_words(t, stimuli, words, repeats, rng):
    """3つの新しい発話（バスない・かばんない・ボールない）を、対応する物を見せた
    視覚つきで海馬に書き込む（仕様書「手順」。1語ずつrepeats回、順序はランダムに
    混ぜる。個体・角度も毎回ランダムに変える＝f70 write_new_words と同じ流儀）。
    """
    hippo = t.language_hippocampus
    start_step = max((ep["written_at"] for ep in hippo.episodes), default=-1) + 1
    order = []
    for w in words:
        order += [w] * repeats
    rng.shuffle(order)
    records = []
    step = start_step
    for w in order:
        vec, meta = rng.choice(stimuli[w])
        text = w + "ない"
        tokens = hippo_write_step(t, text, vec, step)
        records.append({"word": w, "text": text, "tokens": tokens, "written_at": step,
                         "individual": meta["individual"], "yaw": meta["yaw"]})
        step += 1
    return records


# ============================================================================
# ⑤ 睡眠リプレイ（f70 sleep_replay と完全に同一。そのままコピー）
# ============================================================================
def sleep_replay(t, replay_passes=None):
    """言語海馬のリプレイ。出典：run/trainer.py `_consolidate_language`
    （1096〜1140行、2026-09-04時点。grep -n "_consolidate_language" run/trainer.py で
    行番号を再確認済み）。ロジックをそのままコピー・再現する：
      hippo.sample(rng, len(hippo))で全件取得→torch.randpermでシャッフル→
      sleep_lrへ一時差し替え→replay_passes回、各エピソードでforward_hidden→
      perception_head→cross_entropy→backward→clip_grad_norm_→optimizer.step()→
      lrを元に戻す→最後にhippo.decay()。

    replay_passes を明示指定したときはチェックポイントの既定値
    （hippo.replay_passes、現状1）の代わりにそちらを使う（まず1で試し、5・10でも追試）。
    乱数は torch.default_generator のみ使用（設計「決定性の注意」と同じ流儀）。
    """
    hippo = t.language_hippocampus
    passes = int(replay_passes) if replay_passes is not None else hippo.replay_passes
    dev = t.brain._device()
    opt = t._listen_optimizer
    orig_lrs = [g["lr"] for g in opt.param_groups]
    for g in opt.param_groups:
        g["lr"] = hippo.sleep_lr
    all_losses = []
    try:
        for p in range(passes):
            eps = hippo.sample(torch.default_generator, len(hippo))
            perm = torch.randperm(len(eps)).tolist()
            eps = [eps[i] for i in perm]
            pass_losses = []
            for ep in eps:
                ids = [ep["speaker"]] + list(ep["tokens"])
                if len(ids) < 2:
                    continue
                xin = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
                tgt = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
                pfx = torch.tensor(ep["key_vis"], dtype=torch.float32, device=dev)
                out, _ = t.brain.forward_hidden(xin, hidden=None, prefix_vec=pfx)
                out = out[:, 1:, :]      # 先頭＝視覚トークン位置は損失に使わない
                loss = Fnn.cross_entropy(t.brain.perception_head(out)[0], tgt[0])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p_ for grp in opt.param_groups for p_ in grp["params"]),
                    t._listen_grad_clip)
                opt.step()
                pass_losses.append(float(loss.item()))
            all_losses.append(pass_losses)
            print(f"  [睡眠リプレイ] pass {p + 1}/{passes}  件数={len(pass_losses)}  "
                  f"平均loss={np.mean(pass_losses):.4f}", flush=True)
    finally:
        for g, lr in zip(opt.param_groups, orig_lrs):
            g["lr"] = lr
    hippo.decay()
    return all_losses


# ============================================================================
# ⑥ 出力（f70 make_plot と同じ見た目＋f70とのf71比較表を追加）
# ============================================================================
def _setup_font():
    fp = "C:/Windows/Fonts/meiryo.ttc"
    font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()


def make_plot(summary_before, summary_after, words, path, title_suffix=""):
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
    fig.suptitle("F2-71 型の応用テストの切り分け（見た目の混同を除いた再検証）：「◯◯ ない」を"
                 f"海馬に書き込み睡眠で復習させた後、一度も聞かせていない"
                 f"「{TEST_WORD}」にも「ない」を当てはめるか{title_suffix}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("図:", path)


def make_comparison_table(summary_before, summary_after, out_path):
    """「がおー対象（f71）」と「おわん対象（f70）」の比較表を1つ作る（仕様書「出力」節）。

    f70とf71では教える語・テスト対象・正負の対照の語そのものが違う（今回の変更点）ため、
    語名では対応しない。役割（テスト対象／正の対照／負の対照）で対応させて並べる。
    f70の結果.json（F70_RESULT）が無ければスキップする（比較できないだけで、f71本体の
    結果には影響しない）。
    """
    if not os.path.exists(F70_RESULT):
        print(f"[比較] {F70_RESULT} が無いためf70比較はスキップ")
        return None
    with io.open(F70_RESULT, encoding="utf-8") as fp:
        f70 = json.load(fp)

    def top_said(said_counts):
        return sorted(said_counts.items(), key=lambda kv: -kv[1])[0][0]

    roles = [
        ("テスト対象（型の応用の的）", f70["test_word"], TEST_WORD),
        ("正の対照（訓練語）", f70["pos_control"], POS_CONTROL),
        ("負の対照（無関係語）", f70["neg_control"], NEG_CONTROL),
    ]
    rows = []
    for role, w70, w71 in roles:
        f70_before = f70["summary_before"][w70]["na_mean"]
        f70_after = f70["summary_after"][w70]["na_mean"]
        f71_before = summary_before[w71]["na_mean"]
        f71_after = summary_after[w71]["na_mean"]
        rows.append({
            "role": role,
            "f70_word": w70, "f70_write_repeats": f70["write_repeats"],
            "f70_replay_passes": f70["replay_passes"],
            "f70_na_mean_訓練前": round(f70_before, 5),
            "f70_na_mean_訓練後": round(f70_after, 5),
            "f70_差": round(f70_after - f70_before, 5),
            "f70_生成例_訓練後": top_said(f70["summary_after"][w70]["said_counts"]),
            "f71_word": w71,
            "f71_na_mean_訓練前": round(f71_before, 5),
            "f71_na_mean_訓練後": round(f71_after, 5),
            "f71_差": round(f71_after - f71_before, 5),
            "f71_生成例_訓練後": top_said(summary_after[w71]["said_counts"]),
        })
    with io.open(out_path, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=1)
    print("\n[比較] f70（対象=おわん） vs f71（対象=がおー）")
    print("%-20s %8s %8s %14s %14s %14s %14s"
          % ("役割", "f70語", "f71語", "f70差(な)", "f71差(な)", "f70訓練後生成", "f71訓練後生成"))
    for r in rows:
        print("%-20s %8s %8s %14.4f %14.4f %14s %14s"
              % (r["role"], r["f70_word"], r["f71_word"], r["f70_差"], r["f71_差"],
                 r["f70_生成例_訓練後"], r["f71_生成例_訓練後"]))
    print("比較表:", out_path)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_repeats", type=int, default=DEFAULT_WRITE_REPEATS,
                     help="新3語（バスない・かばんない・ボールない）を海馬へ書き込む回数"
                          "（各語につきこの回数。既定1。効果が小さければ3〜5で追試）")
    ap.add_argument("--replay_passes", type=int, default=DEFAULT_REPLAY_PASSES,
                     help="睡眠リプレイの周回数。省略時はチェックポイントの既定値"
                          "（現状1）を使う。効果が小さければ5・10で追試")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=str, default=OUT,
                     help="既定は本番出力先(結果.json等)。追試はサブフォルダを指定して"
                          "本番結果を上書きしない。")
    args = ap.parse_args()
    out_dir = args.outdir

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 74)
    print(f" F2-71 型の応用テストの切り分け  write_repeats={args.write_repeats}"
          f"  replay_passes={args.replay_passes}  seed={args.seed}")
    print(f" 教える語={TRAIN_WORDS}  テスト対象={TEST_WORD}"
          f"  正の対照={POS_CONTROL}  負の対照={NEG_CONTROL}")
    print("=" * 74)

    print("[1] チェックポイント読み込み:", MODEL_PATH)
    t = build_taro(MODEL_PATH)
    print(f"    語彙サイズ {t.produce_vocab.size}  <PARENT>={t._context_speaker_ids['parent']}"
          f"  な={t.produce_vocab.char2idx.get('な')}  い={t.produce_vocab.char2idx.get('い')}")
    print(f"    言語海馬: {len(t.language_hippocampus)}件"
          f"（capacity={t.language_hippocampus.capacity}"
          f" decay={t.language_hippocampus.decay_rate}"
          f" replay_passes(既定)={t.language_hippocampus.replay_passes}"
          f" sleep_lr={t.language_hippocampus.sleep_lr}）")

    print("[2] 刺激（画像→視覚ベクトル）を用意")
    stimuli = build_stimuli()

    print(f"[3] 訓練前の測定（{'・'.join(MEASURE_WORDS)}）")
    t._context_hidden = None
    before = measure(t, stimuli, MEASURE_WORDS, "訓練前")

    print(f"[4] 海馬へ新規3語を書き込み（各{args.write_repeats}回、3語混ぜてランダム順）")
    enable_listen_learn(t, lr=LISTEN_LR)   # 睡眠リプレイ用のoptimizerをここで用意
    write_records = write_new_words(t, stimuli, TRAIN_WORDS, args.write_repeats, rng)
    print(f"    書き込み後の言語海馬: {len(t.language_hippocampus)}件"
          f"（訓練前は{len(t.language_hippocampus) - len(write_records)}件だったはず）")
    for r in write_records[:6]:
        print(f"      written_at={r['written_at']} 語={r['word']} text={r['text']!r}"
              f" tokens={r['tokens']}")
    if len(write_records) > 6:
        print(f"      …ほか{len(write_records) - 6}件")

    print(f"[5] 睡眠リプレイ実行（replay_passes="
          f"{args.replay_passes if args.replay_passes is not None else t.language_hippocampus.replay_passes}）")
    replay_losses = sleep_replay(t, replay_passes=args.replay_passes)

    print(f"[6] 訓練後の測定（{'・'.join(MEASURE_WORDS)}）")
    t._context_hidden = None
    after = measure(t, stimuli, MEASURE_WORDS, "訓練後")

    print("[7] 集計・出力")
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
        "model": MODEL_PATH, "write_repeats": args.write_repeats,
        "replay_passes": (args.replay_passes if args.replay_passes is not None
                           else t.language_hippocampus.replay_passes),
        "seed": args.seed,
        "hippo_len_before_write": len(t.language_hippocampus) - len(write_records),
        "hippo_len_after_write": len(t.language_hippocampus),
        "train_words": TRAIN_WORDS, "test_word": TEST_WORD,
        "pos_control": POS_CONTROL, "neg_control": NEG_CONTROL,
        "records_before": before, "records_after": after,
        "summary_before": {w: summary_before[w] for w in MEASURE_WORDS},
        "summary_after": {w: summary_after[w] for w in MEASURE_WORDS},
        "hippo_write_records": write_records,
        "replay_losses_per_pass": [
            {"pass": i + 1, "n": len(pl), "mean_loss": float(np.mean(pl)) if pl else None}
            for i, pl in enumerate(replay_losses)
        ],
    }
    jpath = os.path.join(out_dir, "結果.json")
    with io.open(jpath, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
    print("生データ:", jpath)

    ppath = os.path.join(out_dir, "図_訓練前後比較.png")
    make_plot(summary_before, summary_after, MEASURE_WORDS, ppath,
              title_suffix=f"\n(write_repeats={args.write_repeats}, "
                            f"replay_passes={result['replay_passes']})")

    cpath = os.path.join(out_dir, "比較_f70とf71.json")
    make_comparison_table(summary_before, summary_after, cpath)

    print("\n[判定用の数字（実装側では判断しない）]")
    for w in MEASURE_WORDS:
        d = summary_after[w]["na_mean"] - summary_before[w]["na_mean"]
        print(f"  {w}: な最大平均 訓練前{summary_before[w]['na_mean']:.4f} → "
              f"訓練後{summary_after[w]['na_mean']:.4f}（差 {d:+.4f}）")
    print("[OK] 完了")


if __name__ == "__main__":
    main()
