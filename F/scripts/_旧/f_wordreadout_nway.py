# -*- coding: utf-8 -*-
"""N語版・直接読み出しテスト（2026-08-24 新規作成。作業B）。

F1-4/F1-6 の `f14_direct_readout.py` は語2つ・物体2つに決め打ちで書かれており、
4語（ぱぱ・ねこ・くつ・みず）以上のテストができない（F1-4h設計の技術付録が指摘）。
このファイルは f14_direct_readout.py を変更せず、同じ考え方（held-out視点を撮って、
語を聞いたときの「想像の見え」に一番近い物体の見えを当てる／両側引き算）を
**N語・N物体**へ一般化した新しいスクリプトである。

【流用（f14_direct_readout.pyから import。車輪の再発明はしない）】
  build_trainer, eye_addrs, set_eye_deg, solve_gaze（内部でcam_frame/angle_err_deg
  を使用）, get_vision, imagine_embedding, binom_threshold
  （いずれも語数・物体数に依存しない汎用関数。f14側の実装は一切変更していない）
  run.trainer._cosine_sim, vision_backends.fovea_crop も f14 と同じ流用元
  （f14はrun.trainerから、こちらは直接import）。

【新規実装（このファイル固有。f14にはN=2専用の同名処理しか無い）】
  - capture_views_nway：物体をN個に一般化した「1個ずつ提示」視点採取
    （f14.capture_views_isolated は toy1/toy2の2個に決め打ち。ここでは
    --word-map で明示された任意個の body 名をループする）
  - two_sided_scores_nway / score_nway：N択の両側引き算・argmax判定
    （f14の two_sided_scores/score_one は2択のif分岐で書かれており使えない）
  - 語どうしの分離度（新規の指標。f14には無い）：想像embedding（w'）どうしの
    ペア間コサイン類似度。値が小さい（0や負に近い）ほど語同士が離れている

書き込み先（指示された範囲のみ）：
  このファイル自身／scratchpad／
  F/logs/F_wordreadout_nway/ （新規）
"""
import argparse
import csv
import itertools
import json
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import cv2                                                       # noqa: E402
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                             # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mujoco                                                     # noqa: E402

from run.trainer import close_env, _cosine_sim                    # noqa: E402
from vision_backends import fovea_crop                            # noqa: E402

# f14_direct_readout は F/scripts 内の同居ファイル（このファイルと同じディレクトリ）
sys.path.insert(0, _HERE)
import f14_direct_readout as f14                                  # noqa: E402

# ============================================================================
# 設定
# ============================================================================
TARO_BASE = dict(
    actuation="muscle", age_months=6.0, orienting_reflex=True,
    hearing=True, lexicon_vision={"backend": "dinov2_vits14", "fovea_px": 32},
    word_attention={"active_sec": 3.0, "enabled": True}, lr=0.0,
)

_F_DIR_DEFAULT = os.path.join(_ROOT, "F", "logs", "F_wordreadout_nway")
_VIEWS_DIR_DEFAULT = os.path.join(_F_DIR_DEFAULT, "views")
_FIG_PATH_DEFAULT = os.path.join(_F_DIR_DEFAULT, "図1_場面別正誤マップ.png")
_SEP_FIG_DEFAULT = os.path.join(_F_DIR_DEFAULT, "図2_語の分離度.png")
_REPORT_PATH_DEFAULT = os.path.join(_F_DIR_DEFAULT, "レポート.md")


def _resolve(path, base=_ROOT):
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base, path)


def parse_args():
    ap = argparse.ArgumentParser(
        description="N語版 直接読み出しテスト（語数・物体数は実験の内容に合わせて指定する）")
    ap.add_argument("--word-map", nargs="+", required=True,
                     help="必須。'語=body名' の形をN個並べる（N=4なら4個）。"
                          "**語はhearing.hear()にそのまま渡す実際の発話テキスト**"
                          "（例: わんわん=test_object2 ぱぱ=test_object1 …）。"
                          "ローマ字等の別名は使わない（chunk化が変わり誤判定の原因になる。"
                          "実測で確認済み）。body名はシーンJSON内の物体bodyの名前")
    ap.add_argument("--scenes-dir", required=True,
                     help="この中の全JSONを1場面ずつ評価する。各シーンは"
                          "--word-mapで指定した全body名の物体を含んでいること")
    ap.add_argument("--models", nargs="+", required=True,
                     help="判定対象モデルを label=path の形で複数指定。"
                          "path省略時または'None'で白紙モデル扱い")
    ap.add_argument("--fovea-px", type=int, default=32,
                     help="視覚エンコードのフォビア窓[px]（学習時の設定と一致させること）")
    ap.add_argument("--backend", default="dinov2_vits14",
                     help="視覚バックエンド名。**学習時と必ず一致させること**"
                          "（学習と試験で違う目を使う事故がF1-8で実際に起きている）")
    ap.add_argument("--backend-seed", type=int, default=0,
                     help="vits14_untrained のランダム重みseed（学習時と一致させること）")
    ap.add_argument("--out-dir", default=None,
                     help=f"出力先ディレクトリ（既定: {_F_DIR_DEFAULT}）")
    ap.add_argument("--views-dir", default=None, help="視点PNGの出力先")
    ap.add_argument("--tag", default="", help="レポート見出しに付ける表示用ラベル")
    return ap.parse_args()


def parse_kv_list(items, what):
    out = []
    for spec in items:
        if "=" not in spec:
            raise ValueError(f"{what}: '{spec}' は 'key=value' 形式で指定してください")
        k, _, v = spec.partition("=")
        out.append((k, v))
    return out


# ============================================================================
# 視点採取：N物体版（f14.capture_views_isolated の一般化）
# ============================================================================
def capture_views_nway(scene_paths, toy_bodies, fovea_px, views_dir, say):
    """toy_bodies: [(toy_key, body_name), ...]（N個）。

    各シーンについて、toy_bodies を1個ずつ順に「見せる」（他は全てrgbaアルファ0で
    透明化）→中心へ視線を合わせて撮る。f14.capture_views_isolated の2物体決め打ち
    ループをN物体へ一般化しただけで、視点の撮り方（透明化→視線をNewton法で解く→
    フォビア切り出し）は完全に同一ロジック（f14.solve_gaze等をそのまま呼ぶ）。
    """
    views = {}          # (toy_key, scene_name) -> np.ndarray(384,)
    view_meta = []       # (toy_key, scene_name, png_path)
    solve_report = []

    for scene_path in scene_paths:
        scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        say(f"\n=== 視点採取(1個ずつ提示・N={len(toy_bodies)}): {scene_name} ===")
        tr = f14.build_trainer(scene_path, with_model=False)
        env, taro = tr.env, tr.taro
        model, data = env.unwrapped.model, env.unwrapped.data
        addrs = f14.eye_addrs(model)

        gadr = {}
        rgba_orig = {}
        for toy_key, body_name in toy_bodies:
            g = int(model.body(body_name).geomadr[0])
            gadr[toy_key] = g
            rgba_orig[toy_key] = model.geom_rgba[g].copy()

        for toy_key, body_name in toy_bodies:
            # 全物体をいったん元の見た目に戻してから、対象以外を透明化する
            for tk, _bn in toy_bodies:
                model.geom_rgba[gadr[tk]] = rgba_orig[tk]
            for tk, _bn in toy_bodies:
                if tk != toy_key:
                    model.geom_rgba[gadr[tk]][3] = 0.0
            mujoco.mj_forward(model, data)
            target = np.array(data.body(body_name).xpos, dtype=float).copy()
            h0, v0, err0 = f14.solve_gaze(model, data, addrs, target)
            say(f"  {toy_key}({body_name}) 中心解: h={h0:+.2f}° v={v0:+.2f}° "
                f"残差=({err0[0]:+.3f}°,{err0[1]:+.3f}°)")
            solve_report.append((scene_name, toy_key, h0, v0, float(np.max(np.abs(err0)))))
            f14.set_eye_deg(data, addrs, h0, v0)
            mujoco.mj_forward(model, data)
            imgs = f14.get_vision(env)
            vec = np.asarray(tr._vision_backend_encode(imgs), dtype=np.float64)
            views[(toy_key, scene_name)] = vec
            crop_src = imgs.get("eye_left_fovea", imgs["eye_left"])
            crop = fovea_crop(crop_src, fovea_px)
            png_path = os.path.join(views_dir, f"{scene_name}_{toy_key}.png")
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"png encode失敗: {png_path}")
            buf.tofile(png_path)
            view_meta.append((toy_key, scene_name, png_path))

        for tk, _bn in toy_bodies:
            model.geom_rgba[gadr[tk]] = rgba_orig[tk]
        close_env(env)
    return views, view_meta, solve_report


# ============================================================================
# 両側引き算・N択判定（f14.two_sided_scores/score_one の一般化）
# ============================================================================
def two_sided_scores_nway(imagine, views, view_keys, model_specs, words):
    v_all = np.stack([views[k] for k in view_keys], axis=0)
    v_mean = v_all.mean(axis=0)
    v_prime = {k: (views[k] - v_mean) for k in view_keys}

    w_prime = {}
    for model_name, _mp in model_specs:
        word_vecs = [imagine[(model_name, w)] for w in words
                     if imagine[(model_name, w)] is not None]
        w_mean = np.mean(np.stack(word_vecs, axis=0), axis=0) if word_vecs else None
        for w in words:
            vec = imagine[(model_name, w)]
            w_prime[(model_name, w)] = None if (vec is None or w_mean is None) \
                else (vec - w_mean)
    return v_prime, w_prime


def score_nway(w_prime_vec, v_prime_by_toy, correct_toy):
    """N択：全toyとのコサインを計算し、最大の物体を答えとする。

    戻り値: (sims辞書, answer_toy, correct(bool), diff=正解-次点)
    """
    if w_prime_vec is None:
        return {}, "N/A", False, float("nan")
    sims = {toy: _cosine_sim(w_prime_vec, v) for toy, v in v_prime_by_toy.items()}
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    answer_toy = ranked[0][0]
    correct = (answer_toy == correct_toy)
    s_correct = sims[correct_toy]
    others = [s for toy, s in sims.items() if toy != correct_toy]
    s_best_other = max(others) if others else float("nan")
    diff = s_correct - s_best_other
    return sims, answer_toy, correct, diff


# ============================================================================
# メイン
# ============================================================================
def main():
    args = parse_args()
    TARO_BASE["lexicon_vision"]["fovea_px"] = int(args.fovea_px)
    TARO_BASE["lexicon_vision"]["backend"] = str(args.backend)
    if str(args.backend) == "vits14_untrained":
        TARO_BASE["lexicon_vision"]["seed"] = int(args.backend_seed)
    # f14側の build_trainer が TARO_BASE を参照するので、こちら側の設定を反映させる
    f14.TARO_BASE = TARO_BASE
    print(f"[視覚バックエンド] {TARO_BASE['lexicon_vision']}")

    word_map = parse_kv_list(args.word_map, "--word-map")   # [(word_key, body_name), ...]
    words = [w for w, _ in word_map]
    n_words = len(words)
    if n_words < 2:
        raise ValueError("--word-map は2個以上指定してください")
    body_by_word = dict(word_map)
    # toy_key（表示・辞書キー用の内部名）は語そのものを使う（1語=1物体の1対1対応が前提）
    toy_bodies = [(w, body_by_word[w]) for w in words]
    answer = {w: w for w in words}     # 語wの正解toy_keyはw自身（1対1対応）

    # 【2026-08-24】表示ラベル=語そのもの（hear()に渡す実際の発話テキスト）。
    #   以前は --word-labels で別の表示名を付けられる設計にしていたが、
    #   語キーと表示ラベルを分けると「hear()に渡す語」と「表示だけの語」を
    #   取り違える事故が起きる（実測：ローマ字キーを使ってhear()に渡した結果、
    #   chunk化が変わり想像embeddingが常にNoneになった）。事故の芽を断つため
    #   表示ラベルは常に語キーと同一にする。
    label_of = {w: w for w in words}
    chance_level = 1.0 / n_words
    print(f"[語] {n_words}語: {list(zip(words, [body_by_word[w] for w in words]))}"
          f"  偶然の水準={chance_level:.1%}")

    e_dir = _resolve(args.out_dir) if args.out_dir else _F_DIR_DEFAULT
    views_dir = _resolve(args.views_dir) if args.views_dir else _VIEWS_DIR_DEFAULT
    os.makedirs(e_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)
    csv_path = os.path.join(e_dir, "明細.csv")
    npz_path = os.path.join(e_dir, "embeddings.npz")
    report_path = os.path.join(e_dir, "レポート.md")
    fig1_path = os.path.join(e_dir, "図1_場面別正誤マップ.png")
    fig2_path = os.path.join(e_dir, "図2_語の分離度.png")

    model_specs = []
    for label, path in parse_kv_list(args.models, "--models"):
        mp = None if path in ("", "None", "none") else path
        model_specs.append((label, mp))

    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say(f"判定対象モデル一覧: {model_specs}")
    say(f"出力先: e_dir={e_dir}  views_dir={views_dir}")

    scene_paths = sorted(
        os.path.join(args.scenes_dir, f) for f in os.listdir(args.scenes_dir)
        if f.endswith(".json"))
    say(f"scenes-dir: {args.scenes_dir}  場面数={len(scene_paths)}")
    if not scene_paths:
        raise ValueError(f"--scenes-dir にJSONが1つも無い: {args.scenes_dir}")

    # ---- 1) held-out視点の採取 -------------------------------------------
    views, view_meta, solve_report = capture_views_nway(
        scene_paths, toy_bodies, args.fovea_px, views_dir, say)
    view_keys = list(views.keys())
    scene_names = [os.path.splitext(os.path.basename(p))[0] for p in scene_paths]

    # ---- 2) 想像の見え埋め込み ---------------------------------------------
    say("\n=== 想像embeddingの取得 ===")
    imagine = {}
    _first_scene = scene_paths[0]
    for model_name, mp in model_specs:
        with_model = mp is not None
        tr = f14.build_trainer(_first_scene, with_model=with_model, model_path=mp, seed=0)
        taro = tr.taro
        say(f"  [{model_name}] hearing.vocab.size={taro.hearing.vocab.size} "
            f"lexicon語彙数={len(taro.lexicon.counts)}")
        for word in words:
            vec, chunk = f14.imagine_embedding(taro, word)
            imagine[(model_name, word)] = vec
            if vec is None:
                say(f"    {model_name}/{label_of[word]}: chunk={chunk} → 想像embeddingなし")
            else:
                say(f"    {model_name}/{label_of[word]}: chunk={chunk} → embedding取得"
                    f"(dim={vec.shape[0]}, norm={np.linalg.norm(vec):.4f})")
        close_env(tr.env)

    # ---- 3) 両側引き算 → N択判定 -------------------------------------------
    say("\n=== 両側引き算によるN択判定 ===")
    v_prime, w_prime = two_sided_scores_nway(imagine, views, view_keys, model_specs, words)

    rows = []
    sep_by_model = {}   # model_name -> {(word_i,word_j): cos(w'_i, w'_j)}
    for model_name, _mp in model_specs:
        for scene_name in scene_names:
            v_prime_by_toy = {w: v_prime[(w, scene_name)] for w in words}
            for word in words:
                wp = w_prime[(model_name, word)]
                sims, ans, correct, diff = score_nway(wp, v_prime_by_toy, answer[word])
                rows.append(dict(
                    model=model_name, scene=scene_name, word=label_of[word],
                    word_key=word, answer=label_of.get(ans, ans), correct=correct,
                    diff=diff,
                    **{f"sim_{label_of[t]}": sims.get(t, float("nan")) for t in words}))
        # 語どうしの分離度（想像embedding同士のペア間コサイン。シーン非依存＝1回でよい）
        pairs = {}
        for wi, wj in itertools.combinations(words, 2):
            a, b = w_prime[(model_name, wi)], w_prime[(model_name, wj)]
            pairs[(wi, wj)] = float("nan") if (a is None or b is None) else _cosine_sim(a, b)
        sep_by_model[model_name] = pairs

    fieldnames = ["model", "scene", "word", "word_key", "answer", "correct", "diff"] + \
        [f"sim_{label_of[t]}" for t in words]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    say(f"明細CSV: {csv_path} ({len(rows)}行)")

    # ---- 4) embeddings.npz -------------------------------------------------
    npz_data = {}
    for k, vec in views.items():
        key = "_".join(str(x) for x in k)
        npz_data[f"view_{key}"] = vec
    for (model_name, word), vec in imagine.items():
        npz_data[f"imagine_{model_name}_{word}"] = (
            vec if vec is not None else np.full(384, np.nan))
    np.savez(npz_path, **npz_data)
    say(f"embeddings.npz: {npz_path} ({len(npz_data)}キー)")

    # ---- 5) 集計（語ごとの正誤・正答数・偶然の水準を必ず併記） -----------------
    say(f"\n=== 正答数（{n_words}語・偶然の水準={chance_level:.1%}） ===")
    n_scenes = len(scene_names)
    summary = {}
    per_word_summary = {}
    for model_name, _mp in model_specs:
        model_rows = [r for r in rows if r["model"] == model_name]
        c = sum(1 for r in model_rows if r["correct"])
        n = len(model_rows)
        summary[model_name] = (c, n)
        say(f"  {model_name}: 全体 {c}/{n} ({c / n:.1%})  偶然の水準={chance_level:.1%}")
        per_word = {}
        for word in words:
            wr = [r for r in model_rows if r["word_key"] == word]
            cw = sum(1 for r in wr if r["correct"])
            per_word[word] = (cw, len(wr))
            pct = f" ({cw / len(wr):.1%})" if len(wr) else ""
            say(f"    {label_of[word]}: {cw}/{len(wr)}{pct}")
        per_word_summary[model_name] = per_word

        say(f"    語どうしの分離度（cos、|値|が小さいほど分離良好）:")
        for (wi, wj), c_ij in sep_by_model[model_name].items():
            say(f"      {label_of[wi]}-{label_of[wj]}: {c_ij:+.4f}")

    if n_scenes > 0:
        k5 = f14.binom_threshold(n_scenes, p=chance_level)
        say(f"\n（参考・合否は判断しない）片側二項検定p<0.05の最小正答数: "
            f"{k5}/{n_scenes}（n={n_scenes}, chance={chance_level:.1%}）")

    # ---- 6) 図1：場面×モデルの正誤マップ ------------------------------------
    say("\n=== 図1：場面×モデルの正誤マップ ===")
    model_names = [m for m, _ in model_specs]
    grid = np.zeros((len(model_names), len(scene_names)))
    for i, model_name in enumerate(model_names):
        pr = {r["scene"]: r["correct"] for r in rows if r["model"] == model_name}
        # 複数語が同じsceneに載る場合があるので、その場面の平均正答率で塗る
        for j, sn in enumerate(scene_names):
            vals = [r["correct"] for r in rows
                    if r["model"] == model_name and r["scene"] == sn]
            grid[i, j] = (sum(vals) / len(vals)) if vals else np.nan
    fig, ax = plt.subplots(figsize=(max(8, len(scene_names) * 0.3), 3))
    ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(f"場面（{len(scene_names)}場面）")
    ax.set_title(f"N語版 場面別正答率マップ（{n_words}語・偶然={chance_level:.0%}）")
    if len(scene_names) <= 40:
        ax.set_xticks(range(len(scene_names)))
        ax.set_xticklabels(scene_names, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    say(f"図1: {fig1_path}")

    # ---- 7) 図2：語どうしの分離度（ヒートマップ） ---------------------------
    say("\n=== 図2：語どうしの分離度 ===")
    fig, axes = plt.subplots(1, len(model_names), figsize=(4 * len(model_names), 3.6))
    if len(model_names) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, model_names):
        mat = np.eye(n_words)
        for (wi, wj), c_ij in sep_by_model[model_name].items():
            i, j = words.index(wi), words.index(wj)
            mat[i, j] = mat[j, i] = c_ij
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(n_words))
        ax.set_xticklabels([label_of[w] for w in words], rotation=45, fontsize=8)
        ax.set_yticks(range(n_words))
        ax.set_yticklabels([label_of[w] for w in words], fontsize=8)
        ax.set_title(model_name, fontsize=9)
    fig.suptitle("語どうしの分離度（想像embeddingのcos、対角=1.0）")
    fig.colorbar(im, ax=axes, shrink=0.7)
    os.makedirs(os.path.dirname(fig2_path), exist_ok=True)
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    say(f"図2: {fig2_path}")

    # ---- 8) レポート --------------------------------------------------------
    lines = []
    lines.append(f"# N語版 直接読み出しテスト 結果{f'（{args.tag}）' if args.tag else ''}\n")
    lines.append("モデル: " + "、".join(
        f"{mn}=`{mp}`" if mp else f"{mn}=白紙" for mn, mp in model_specs) + "\n")
    lines.append(f"語（{n_words}語・偶然の水準={chance_level:.1%}）: " +
                 "、".join(f"{label_of[w]}({body_by_word[w]})" for w in words) + "\n")
    lines.append(f"場面数: {n_scenes}\n")
    lines.append("## 正答数（全体・語ごと）\n")
    lines.append("| モデル | 全体 | " + " | ".join(label_of[w] for w in words) + " |")
    lines.append("|---|---|" + "---|" * n_words)
    for model_name, mp in model_specs:
        c, n = summary[model_name]
        row_cells = [f"{cw}/{nw}" for cw, nw in
                     (per_word_summary[model_name][w] for w in words)]
        lines.append(f"| {model_name} | {c}/{n} | " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("## 作ったファイル\n")
    lines.append(f"- 明細CSV: `{csv_path}`")
    lines.append(f"- 埋め込みnpz: `{npz_path}`")
    lines.append(f"- 図1（場面別正誤マップ）: `{fig1_path}`")
    lines.append(f"- 図2（語の分離度）: `{fig2_path}`")
    lines.append(f"- 視点画像: `{views_dir}\\` 配下 {len(view_meta)}枚")
    lines.append("")
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    say(f"\nレポート: {report_path}")

    return summary


if __name__ == "__main__":
    main()
