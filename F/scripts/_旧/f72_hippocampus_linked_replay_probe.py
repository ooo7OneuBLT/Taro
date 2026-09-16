# -*- coding: utf-8 -*-
"""F2-72 睡眠リプレイの隣接エピソード連結テスト（がおーへの型の応用、再挑戦）。

仕様：F/docs/二語文/仕様_睡眠リプレイの隣接エピソード連結テスト_2026-09-05.md

背景（仕様書前半）：F2-69〜71で「がおー（一度も『〜ない』を教えていない語）」への
型の応用を試したが証拠が得られなかった。原因調査で、太郎の睡眠リプレイが記憶を
1件ずつ完全に独立して（GRUの隠れ状態を毎回リセットして）学習させていることが
分かった（運動性海馬の実装・人間の睡眠リプレイの実測・R2D2の zero start state
問題、の3方向で裏付け）。本スクリプトは、海馬の中で時間的に近い記憶
（`written_at`45〜53：バス×6・がおー×3が交互に並ぶ既存クラスタ）だけを
隠れ状態をつなげたまま連続処理し、独立処理版と比較する。

方法（仕様書後半）：
- モデル復元・視覚エンコード・`generate_probe()`（測定方法）は
  `F/scripts/f71_pattern_generalization_disambiguation.py`をモジュールとして
  import しそのまま使う（変更しない）。
- 連結リプレイ（`linked_sleep_replay()`）は本ファイルで新規に書く（f71のコピー
  ではない）。クラスタ9件を元の順で連結処理（前エントリの最終隠れ状態を次の
  開始隠れ状態にそのまま渡す。detachしない）。9件ぶんのcross_entropyを合計して
  1回だけbackward/step。クラスタ外66件は`_consolidate_language`と同じ独立処理
  （hidden=None固定・個別step）のまま（今回はクラスタ内だけ変える最小の変更）。
- 独立処理版（従来）の比較対象は f71 の `sleep_replay()`（`_consolidate_language`
  をそのまま再現したもの）を無変更で使う。
- 「連結版」「独立処理版」を同じ乱数シードで両方実行し、がおー・バス・くつの
  訓練前後の「ない」最大確率と実際の生成文字列を比較する。

  .venv/Scripts/python.exe F/scripts/f72_hippocampus_linked_replay_probe.py
出力: F/logs/F2-72_睡眠リプレイ連結テスト/結果.json
      F/logs/F2-72_睡眠リプレイ連結テスト/図_連結版と独立版の比較.png
"""
import os, sys, io, json, argparse, warnings
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

# f71をモジュールとしてそのまま流用する（build_taro/generate_probe/build_stimuli/
# measure/summarize/enable_listen_learn/sleep_replay/_setup_font 等。import時に
# f71側の `import f49_test as T` が実行され、scene.py 経由で taro_core/src/brain
# 配下がsys.pathに追加される副作用がある＝f71と同じ順序で読み込む必要がある）。
import f71_pattern_generalization_disambiguation as f71lib

MODEL_PATH = f71lib.MODEL_PATH
OUT = "F/logs/F2-72_睡眠リプレイ連結テスト"
SEED = f71lib.SEED  # 20260905（f71と同じ。仕様書「比較」節：同じ乱数シードで両方実行）
CLUSTER_LO, CLUSTER_HI = 45, 53
DEFAULT_REPLAY_PASSES = 10


# ============================================================================
# 連結リプレイ（新規ロジック。f71 sleep_replay のコピーではない）
# ============================================================================
def extract_cluster(hippo, lo=CLUSTER_LO, hi=CLUSTER_HI):
    """written_atがlo〜hiの範囲のエピソードをwritten_at昇順で取り出す。

    出典：仕様書「後半」1節。前回調査でwritten_at45〜53にバス×6・がおー×3が
    元の順で交互に並んでいることを確認済み（本スクリプト作成時に再確認：
    バスだね/バスだよ/バスだね/がおおだね/がおおだよ/がおおいるね/バスだよ/
    バスだね/バス、の9件）。
    """
    cluster = [ep for ep in hippo.episodes if lo <= ep["written_at"] <= hi]
    cluster.sort(key=lambda e: e["written_at"])
    return cluster


def _step_one_episode(t, ep, hidden, dev):
    """1エピソードぶんのforward。戻り値: (loss, 更新後hidden)。

    出典：run/trainer.py `_consolidate_language`（1121〜1136行）と同じ損失の
    取り方（forward_hidden→先頭視覚トークン位置を除く→cross_entropy）。
    hiddenだけ外から渡せるようにして連結・独立の両方から使えるようにした。
    """
    ids = [ep["speaker"]] + list(ep["tokens"])
    if len(ids) < 2:
        return None, hidden
    xin = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
    tgt = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
    pfx = torch.tensor(ep["key_vis"], dtype=torch.float32, device=dev)
    out, hh = t.brain.forward_hidden(xin, hidden=hidden, prefix_vec=pfx)
    out = out[:, 1:, :]  # 先頭＝視覚トークン位置は損失に使わない
    loss = Fnn.cross_entropy(t.brain.perception_head(out)[0], tgt[0])
    return loss, hh


def linked_sleep_replay(t, replay_passes, cluster_lo=CLUSTER_LO, cluster_hi=CLUSTER_HI):
    """クラスタ内だけ隠れ状態を連結し、クラスタ外は今まで通り独立処理する睡眠リプレイ。

    仕様書「後半」2節の実装。f71 sleep_replay とは別に今回新規で書く：
    - クラスタ（written_at cluster_lo〜cluster_hi、元の順）：1件目はhidden=None、
      以降は前エントリの最終隠れ状態をそのまま次エントリへ渡す（detachしない）。
      9件ぶんのcross_entropyを合計し、1回だけzero_grad→backward→clip→stepする。
    - クラスタ外：`_consolidate_language`と同じ独立処理（hidden=None固定・
      hippo.sample()による直近偏重抽出→シャッフル→エピソードごとにzero_grad/
      backward/step）。hippo.sample()はクラスタ外のリストにだけ一時的に絞って
      呼ぶ（hippo.episodesを一時的に差し替えて呼び、直後に元へ戻す。
      sample()自体は無変更で流用）。
    """
    hippo = t.language_hippocampus
    dev = t.brain._device()
    opt = t._listen_optimizer
    grad_clip = t._listen_grad_clip

    cluster = extract_cluster(hippo, cluster_lo, cluster_hi)
    cluster_id_set = set(id(ep) for ep in cluster)
    others = [ep for ep in hippo.episodes if id(ep) not in cluster_id_set]
    print(f"  [連結リプレイ] クラスタ{len(cluster)}件"
          f"（written_at {cluster[0]['written_at']}〜{cluster[-1]['written_at']}）"
          f"  クラスタ外{len(others)}件  合計{len(hippo)}件")
    for ep in cluster:
        print(f"    written_at={ep['written_at']:3d}  {t.produce_vocab.decode(ep['tokens'])!r}")

    orig_lrs = [g["lr"] for g in opt.param_groups]
    for g in opt.param_groups:
        g["lr"] = hippo.sleep_lr

    cluster_pass_losses, other_pass_losses = [], []
    try:
        for p in range(replay_passes):
            # --- クラスタ内：連結処理（隠れ状態をつなげたまま。1回だけbackward） ---
            opt.zero_grad()
            hidden = None
            losses = []
            for ep in cluster:
                loss, hidden = _step_one_episode(t, ep, hidden, dev)
                if loss is not None:
                    losses.append(loss)
            total_loss = torch.stack(losses).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (pp for grp in opt.param_groups for pp in grp["params"]), grad_clip)
            opt.step()
            cluster_loss_vals = [float(l.item()) for l in losses]
            cluster_pass_losses.append(cluster_loss_vals)
            hidden = None  # backward後、計算グラフは既に解放済み。次passへは持ち越さない。

            # --- クラスタ外：独立処理（_consolidate_language相当。変更しない） ---
            orig_eps = hippo.episodes
            hippo.episodes = others
            eps = hippo.sample(torch.default_generator, len(others))
            hippo.episodes = orig_eps
            perm = torch.randperm(len(eps)).tolist()
            eps = [eps[i] for i in perm]
            o_losses = []
            for ep in eps:
                loss, _ = _step_one_episode(t, ep, None, dev)
                if loss is None:
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (pp for grp in opt.param_groups for pp in grp["params"]), grad_clip)
                opt.step()
                o_losses.append(float(loss.item()))
            other_pass_losses.append(o_losses)
            print(f"  [連結リプレイ] pass {p + 1}/{replay_passes}  "
                  f"クラスタ平均loss={np.mean(cluster_loss_vals):.4f}  "
                  f"クラスタ外平均loss={np.mean(o_losses):.4f}", flush=True)
    finally:
        for g, lr in zip(opt.param_groups, orig_lrs):
            g["lr"] = lr
    hippo.decay()
    return {
        "cluster_written_at": [ep["written_at"] for ep in cluster],
        "cluster_texts": [t.produce_vocab.decode(ep["tokens"]) for ep in cluster],
        "n_cluster": len(cluster), "n_others": len(others),
        "cluster_pass_losses": cluster_pass_losses,
        "other_pass_losses": other_pass_losses,
    }


# ============================================================================
# 机上確認（仕様書「検証・止まる条件」節：1周がエラーなく通ることを先に確認）
# ============================================================================
def sanity_check():
    print("=" * 74)
    print(" [机上確認] クラスタ抽出→連結リプレイ1周が通るかを先に確認")
    print("=" * 74)
    torch.manual_seed(SEED)
    t = f71lib.build_taro(MODEL_PATH)
    f71lib.enable_listen_learn(t, lr=f71lib.LISTEN_LR)
    hippo = t.language_hippocampus
    cluster = extract_cluster(hippo)
    assert len(cluster) == 9, f"クラスタ件数が9件でない: {len(cluster)}件"
    texts = [t.produce_vocab.decode(ep["tokens"]) for ep in cluster]
    n_bus = sum(1 for s in texts if s.startswith("バス"))
    n_gao = sum(1 for s in texts if s.startswith("がお"))
    assert n_bus == 6 and n_gao == 3, f"内訳が想定と違う（バス{n_bus}・がおー{n_gao}）: {texts}"
    out = linked_sleep_replay(t, replay_passes=1)
    assert len(out["cluster_pass_losses"]) == 1
    assert len(out["cluster_pass_losses"][0]) == 9
    print(f"  [机上確認] OK：クラスタ内訳=バス{n_bus}・がおー{n_gao}件、"
          f"1周のクラスタ平均loss={np.mean(out['cluster_pass_losses'][0]):.4f}、"
          f"クラスタ外平均loss={np.mean(out['other_pass_losses'][0]):.4f}")
    print()


# ============================================================================
# 実験本体（連結版・独立処理版を同じ乱数シードで実行）
# ============================================================================
def run_one(label, sleep_fn, seed, stimuli, replay_passes, inject_nai=False):
    """inject_nai=True のとき、仕様書には無い補足検証として、f71と同じ手順
    （write_new_words）で「バスない・かばんない・ボールない」を新規に海馬へ
    書き込んでから睡眠リプレイする（「訓練前」測定の直後・リプレイの直前、
    f71 main()の[3]→[4]→[5]と同じ順序）。理由：チェックポイント
    F2-49c_r3の言語海馬75件には「ない」を含む発話が1件も無いことを実測で
    確認した（本スクリプト作成時のチェック。かばん/がおー/バス等はすべて
    「～だね/だよ/いるね」のみ）。そのため仕様書「やること」節を文字通り
    （書き込みなし・既存75件のみ）実装すると、な確率は訓練前後で常にほぼ0の
    ままになり、「型の応用」仮説の検証にならない可能性が高い。この関数は
    その懸念を確かめるための追加実験（既定では実行しない。要相談点として
    報告に記載）。
    """
    print("=" * 74)
    print(f" [{label}]  replay_passes={replay_passes}  seed={seed}  inject_nai={inject_nai}")
    print("=" * 74)
    torch.manual_seed(seed)
    t = f71lib.build_taro(MODEL_PATH)
    print(f"  言語海馬: {len(t.language_hippocampus)}件"
          f"（sleep_lr={t.language_hippocampus.sleep_lr}）")
    f71lib.enable_listen_learn(t, lr=f71lib.LISTEN_LR)

    t._context_hidden = None
    before = f71lib.measure(t, stimuli, f71lib.MEASURE_WORDS, "訓練前")

    write_records = None
    if inject_nai:
        import random
        rng = random.Random(seed)
        write_records = f71lib.write_new_words(
            t, stimuli, f71lib.TRAIN_WORDS, repeats=1, rng=rng)
        print(f"  [ない注入] {len(write_records)}件書き込み（written_at="
              f"{write_records[0]['written_at']}〜{write_records[-1]['written_at']}）")

    extra = sleep_fn(t, replay_passes)

    t._context_hidden = None
    after = f71lib.measure(t, stimuli, f71lib.MEASURE_WORDS, "訓練後")

    summary_all = f71lib.summarize(before + after)
    summary_before = {w: summary_all[(w, "訓練前")] for w in f71lib.MEASURE_WORDS}
    summary_after = {w: summary_all[(w, "訓練後")] for w in f71lib.MEASURE_WORDS}

    print(f"\n  [{label}] 結果")
    print("  %-6s %8s %10s %10s %10s   %s" % ("語", "段階", "な最大平均", "い最大平均", "件数", "生成例(最頻)"))
    for w in f71lib.MEASURE_WORDS:
        for phase, s in (("訓練前", summary_before[w]), ("訓練後", summary_after[w])):
            top = sorted(s["said_counts"].items(), key=lambda kv: -kv[1])[0][0]
            print("  %-6s %8s %10.4f %10.4f %10d   %r(%d/%d)"
                  % (w, phase, s["na_mean"], s["i_mean"], s["n"], top,
                     s["said_counts"][top], s["n"]))
    print()
    return {
        "label": label, "seed": seed, "replay_passes": replay_passes,
        "inject_nai": inject_nai, "write_records": write_records,
        "hippo_len": len(t.language_hippocampus),
        "records_before": before, "records_after": after,
        "summary_before": summary_before, "summary_after": summary_after,
        "extra": extra,
    }


def make_plot(result_linked, result_indep, words, path):
    f71lib._setup_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(len(words))
    width = 0.2
    series = [
        ("連結-訓練前", result_linked["summary_before"], "#a0aec0", -1.5 * width),
        ("連結-訓練後", result_linked["summary_after"], "#2b6cb0", -0.5 * width),
        ("独立-訓練前", result_indep["summary_before"], "#f6ad9a", 0.5 * width),
        ("独立-訓練後", result_indep["summary_after"], "#c53030", 1.5 * width),
    ]
    for ax, key, title in ((axes[0], "na_mean", "「な」の最大確率（平均）"),
                            (axes[1], "i_mean", "「い」の最大確率（平均）")):
        for label, summary, color, off in series:
            vals = [summary[w][key] for w in words]
            ax.bar(x + off, vals, width, label=label, color=color)
        ax.set_xticks(x); ax.set_xticklabels(words, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 1.0); ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("F2-72 睡眠リプレイの隣接エピソード連結テスト：\n"
                  "written_at45〜53（バス×6・がおー×3）を連結処理 vs 独立処理", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("図:", path)


def compare_and_save(result_linked, result_indep, words, outdir, tag, seed, replay_passes,
                      inject_nai):
    print("=" * 90)
    print(f" [比較・{tag}] 連結版 vs 独立処理版（同一シード, inject_nai={inject_nai}）")
    print("=" * 90)
    print("%-6s %8s | %10s %10s %14s | %10s %10s %14s"
          % ("語", "段階", "連結-な最大", "連結-い最大", "連結-生成例",
             "独立-な最大", "独立-い最大", "独立-生成例"))
    comparison_rows = []
    for w in words:
        for phase, sl, si in (("訓練前", result_linked["summary_before"], result_indep["summary_before"]),
                               ("訓練後", result_linked["summary_after"], result_indep["summary_after"])):
            top_l = sorted(sl[w]["said_counts"].items(), key=lambda kv: -kv[1])[0][0]
            top_i = sorted(si[w]["said_counts"].items(), key=lambda kv: -kv[1])[0][0]
            print("%-6s %8s | %10.4f %10.4f %14r | %10.4f %10.4f %14r"
                  % (w, phase, sl[w]["na_mean"], sl[w]["i_mean"], top_l,
                     si[w]["na_mean"], si[w]["i_mean"], top_i))
            comparison_rows.append({
                "word": w, "phase": phase,
                "linked_na_mean": sl[w]["na_mean"], "linked_i_mean": sl[w]["i_mean"],
                "linked_top_said": top_l,
                "independent_na_mean": si[w]["na_mean"], "independent_i_mean": si[w]["i_mean"],
                "independent_top_said": top_i,
            })

    result = {
        "tag": tag, "model": MODEL_PATH, "seed": seed, "replay_passes": replay_passes,
        "inject_nai": inject_nai,
        "cluster_range": [CLUSTER_LO, CLUSTER_HI],
        "measure_words": words,
        "linked": result_linked, "independent": result_indep,
        "comparison": comparison_rows,
    }
    jpath = os.path.join(outdir, f"結果_{tag}.json")
    with io.open(jpath, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
    print("\n生データ:", jpath)

    ppath = os.path.join(outdir, f"図_{tag}.png")
    make_plot(result_linked, result_indep, words, ppath)

    print("\n[判定用の数字（実装側では判断しない）]")
    for w in words:
        dl = result_linked["summary_after"][w]["na_mean"] - result_linked["summary_before"][w]["na_mean"]
        di = result_indep["summary_after"][w]["na_mean"] - result_indep["summary_before"][w]["na_mean"]
        print(f"  {w}: な最大平均差  連結版{dl:+.4f}  独立処理版{di:+.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_passes", type=int, default=DEFAULT_REPLAY_PASSES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", type=str, default=OUT)
    ap.add_argument("--skip_sanity", action="store_true",
                     help="机上確認（1周ドライラン）を飛ばす（再実行時の時短用）")
    ap.add_argument("--skip_inject_variant", action="store_true",
                     help="仕様書に無い補足実験（ない注入あり版）を飛ばし、"
                          "仕様書どおりの版（書き込みなし）だけ実行する")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not args.skip_sanity:
        sanity_check()

    print("[刺激] build_stimuli()（f71をそのまま流用。1回だけ計算し全実験で使い回す）")
    stimuli = f71lib.build_stimuli()
    words = f71lib.MEASURE_WORDS

    # --- 本編：仕様書どおり（海馬の書き込みなし。既存75件のみで連結/独立を比較） ---
    result_linked = run_one(
        "連結版（クラスタ内は隠れ状態をつなげる）", linked_sleep_replay,
        args.seed, stimuli, args.replay_passes)
    result_indep = run_one(
        "独立処理版（従来＝_consolidate_language相当、f71 sleep_replayをそのまま使用）",
        lambda t, rp: {"all_losses": f71lib.sleep_replay(t, replay_passes=rp)},
        args.seed, stimuli, args.replay_passes)
    compare_and_save(result_linked, result_indep, words, args.outdir,
                      "仕様書どおり_書き込みなし", args.seed, args.replay_passes,
                      inject_nai=False)

    if not args.skip_inject_variant:
        # --- 補足：f71と同じ「ない」新規書き込みを足した版（仕様書には無い。
        #   本チェックポイントの海馬に「ない」を含む発話が1件も無いため、
        #   書き込みなしのままでは「型の応用」の検証にならない懸念を確かめる） ---
        print("\n" + "#" * 90)
        print("# 補足実験：f71と同じ手順でバス/かばん/ボールに「ない」を新規書き込みしてから比較")
        print("# （仕様書には明記が無い。要相談点として報告する）")
        print("#" * 90 + "\n")
        result_linked2 = run_one(
            "連結版+ない注入", linked_sleep_replay,
            args.seed, stimuli, args.replay_passes, inject_nai=True)
        result_indep2 = run_one(
            "独立処理版+ない注入",
            lambda t, rp: {"all_losses": f71lib.sleep_replay(t, replay_passes=rp)},
            args.seed, stimuli, args.replay_passes, inject_nai=True)
        compare_and_save(result_linked2, result_indep2, words, args.outdir,
                          "補足_ない注入あり", args.seed, args.replay_passes,
                          inject_nai=True)

    print("[OK] 完了")


if __name__ == "__main__":
    main()
