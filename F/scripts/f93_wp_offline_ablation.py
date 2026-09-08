# -*- coding: utf-8 -*-
"""切り分け実験：世界の予測器（ポート型）の入力記録をオフラインで変種に流して比べる。

【仕様】F/docs/二語文/仕様_M7b-1改3_切り分け実験_入力記録とオフライン再生_2026-09-10.md
「後半」2節。

【使い方】
    .venv/Scripts/python.exe F/scripts/f93_wp_offline_ablation.py <pkl> \
        [--variants V0,V1,V2,V3,V4,V5] [--out <dir>]

【読むだけ】<pkl>（run/plugins/common/world_predictor_record.py が書いた
  world_pred_inputsのtick列。1要素={"step","t_sec","vision","hearing","body",
  "attended_id","visible","vanished"}）。

【触らないこと】PortWorldPredictor本体（taro_core/src/brain/cerebral_cortex/
  world_predictor.py）は無改修。変種はこのスクリプトの中だけで、構築済みの
  インスタンスへ types.MethodType でメソッドを差し替えるか、構築引数を
  変えるだけで作る（本体のソースコードは1行も書き換えない）。

【出力】--out（既定＝pklと同じフォルダ）に
    結果_切り分け.json
    世界の予測器_切り分け.png
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, os.pardir))
_CORE_BRAIN = os.path.join(_ROOT, "taro_core", "src", "brain")
if _CORE_BRAIN not in sys.path:
    sys.path.insert(0, _CORE_BRAIN)

from cerebral_cortex.world_predictor import PortWorldPredictor, _PortMem  # noqa: E402

AGE_BIN_EDGES = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, None)]
AGE_BIN_LABELS = ["0-2", "2-4", "4-6", "6-8", "8-12", "12+"]
ALL_VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5"]


def _bin_label(age_sec):
    for (lo, hi), label in zip(AGE_BIN_EDGES, AGE_BIN_LABELS):
        if hi is None:
            if age_sec >= lo:
                return label
        elif lo <= age_sec < hi:
            return label
    return None


def load_records(pkl_path):
    import pickle
    with open(pkl_path, "rb") as fp:
        return pickle.load(fp)


def compute_variance_stats(records):
    """記録全体（全変種で共通）から、相対値の分母を1回だけ計算する。

    仕様「後半」2節：「err_vecを『記録中のobj_vecの要素分散の平均』で割ったもの、
    err_stateを『obj_stateの要素分散の平均』で割ったもの」。
    """
    vecs = []
    states = []
    for rec in records:
        for _fid, d in rec.get("vision", {}).items():
            if d.get("obj_vec") is not None:
                vecs.append(np.asarray(d["obj_vec"], dtype=np.float64))
            if d.get("obj_state") is not None:
                states.append(np.asarray(d["obj_state"], dtype=np.float64))
    var_vec_mean = float(np.var(np.stack(vecs), axis=0).mean()) if vecs else 1.0
    var_state_mean = float(np.var(np.stack(states), axis=0).mean()) if states else 1.0
    if var_vec_mean <= 0.0:
        var_vec_mean = 1.0
    if var_state_mean <= 0.0:
        var_state_mean = 1.0
    return var_vec_mean, var_state_mean


def act_dim_and_n_chunks(records):
    act_dim = None
    max_chunk = 0
    for rec in records:
        act = (rec.get("body") or {}).get("act")
        if act_dim is None and act is not None:
            act_dim = len(act)
        cid = (rec.get("hearing") or {}).get("chunk_id_plus1")
        if cid is not None:
            max_chunk = max(max_chunk, int(cid))
    if act_dim is None:
        act_dim = 0
    # 【仕様「後半」2節】「記録中のchunk_id_plus1の最大値+1 か 20 の大きい方」
    n_chunks = max(max_chunk + 1, 20)
    return act_dim, n_chunks


# ---------------------------------------------------------------------------
# 変種の作り方（本体無改修。構築引数を変えるか、構築後にメソッドを差し替えるだけ）
# ---------------------------------------------------------------------------

def _patch_v1(wp):
    """V1：遅い層を切る（文脈を零）。_fast_stepの中の self._h_slow を
    torch.zeros_like に差し替えるだけ（仕様「後半」2節）。"""
    def _fast_step(self, kind, h_prev, x):
        if h_prev is None:
            h_prev = torch.zeros(1, self.h_port, device=self._device())
        zero_slow = torch.zeros_like(self._h_slow)
        f_in = torch.cat([x, zero_slow], dim=-1)
        h_gru = self.gru[kind](f_in, h_prev)
        h_new = ((1.0 - 1.0 / self.tau_fast) * h_prev
                  + (1.0 / self.tau_fast) * h_gru)
        return h_new
    wp._fast_step = types.MethodType(_fast_step, wp)


_V2_SHARED_KEY = "__shared__"


def _patch_v2(wp):
    """V2：視覚の記憶を1本にして物をまたいで引き継ぐ（旧版の形）。
    _get_visionがfile_idに関係なく同じ_PortMemを返し、dropは何もしない
    （仕様「後半」2節「drop で消さない」）。

    【実装判断・2026-09-10、机上確認で気づいた点】共有する_PortMemを
    self._vision_states の外の属性に置くと、observe_all内蔵のtbptt=1 detach処理
    （`for st in self._vision_states.values(): st.h = st.h.detach()`、本体
    無改修のため触れない）の対象から外れ、隠れ状態が毎tickdetachされずに
    計算グラフが積み上がって数tickで壊れる（実測：RuntimeError inplace
    modification、200tick中の早い段階で発生）。self._vision_states の中に
    固定キーで1個だけ入れておけば、本体の既存detachループがそのまま拾う。
    """
    wp._vision_states[_V2_SHARED_KEY] = _PortMem()

    def _get_vision(self, file_id):
        return self._vision_states[_V2_SHARED_KEY]

    def drop(self, file_id):
        return

    wp._get_vision = types.MethodType(_get_vision, wp)
    wp.drop = types.MethodType(drop, wp)


def _patch_v5(wp):
    """V5：新しい物の記憶を「直前に注意していた物の記憶」から始める（温かい立ち上がり）。
    無ければ零（＝Noneのまま、_fast_stepが零から始める）。「直前に注意していた物」＝
    このtickを処理する直前までの外部ループが持つ状態（wp._last_attended_idに
    リプレイ側が毎tick先頭でセットする。仕様「後半」2節に外側からの更新方法の
    明記が無いため、リプレイの外側ループが「1つ前のtickのattended_id」を渡す形に
    した[実装判断・2026-09-10、理由：新しい物が現れた"直前"は前tickの時点でしか
    定義できないため]。"""
    wp._last_attended_id = None

    def _get_vision(self, file_id):
        st = self._vision_states.get(file_id)
        if st is None:
            st = _PortMem()
            prev_id = getattr(self, "_last_attended_id", None)
            prev_st = (self._vision_states.get(prev_id)
                       if prev_id is not None else None)
            if prev_st is not None and prev_st.h is not None:
                st.h = prev_st.h.detach().clone()
            self._vision_states[file_id] = st
        return st

    wp._get_vision = types.MethodType(_get_vision, wp)


def build_wp(variant, act_dim, n_chunks):
    torch.manual_seed(0)
    tau_fast = 1 if variant == "V4" else 5
    wp = PortWorldPredictor(
        n_chunks=n_chunks, act_dim=act_dim, obj_vec_dim=384,
        h_port=64, h_slow=32, summary_dim=32, tau_fast=tau_fast, tau_slow=40,
        lr=1e-3, max_files=4, slow_window=40,
    )
    if variant == "V1":
        _patch_v1(wp)
    elif variant == "V2":
        _patch_v2(wp)
    elif variant == "V5":
        _patch_v5(wp)
    # V0・V3・V4は構築済みのwpをそのまま使う（V3は入力の前処理だけ、V4は構築引数だけ）
    return wp


def run_variant(variant, records, act_dim, n_chunks, var_vec_mean, var_state_mean):
    wp = build_wp(variant, act_dim, n_chunks)

    bins_acc = {label: {"err_vec_sum": 0.0, "err_vec_n": 0,
                          "err_state_sum": 0.0, "err_state_n": 0}
                for label in AGE_BIN_LABELS}
    first_seen = {}
    prev_attended = None

    for i, rec in enumerate(records):
        vision = rec.get("vision", {}) or {}
        cur_ids = set(vision.keys())

        # 【仕様「後半」2節】記録に無くなったfile_idはwp.drop(fid)
        for old_id in list(wp.active_ids()):
            if old_id not in cur_ids:
                wp.drop(old_id)

        if variant == "V5":
            wp._last_attended_id = prev_attended

        for fid in cur_ids:
            if fid not in first_seen:
                first_seen[fid] = i

        vision_io = {}
        for fid, d in vision.items():
            obj_state = d.get("obj_state")
            obj_vec = d.get("obj_vec")
            if variant == "V3" and obj_vec is not None:
                v = np.asarray(obj_vec, dtype=np.float32)
                norm = float(np.linalg.norm(v))
                if norm > 1e-8:
                    v = v / norm
                obj_vec = v
            vision_io[fid] = {"obj_state": obj_state, "obj_vec": obj_vec}

        hearing = rec.get("hearing", {}) or {}
        parent_spoke = hearing.get("parent_spoke", 0.0)
        chunk_id_plus1 = hearing.get("chunk_id_plus1")
        # 【実装判断・2026-09-10】記録には predict_all 用の hearing_input
        #   （chunk_id_plus1は常時。仕様「後半」1節）しか無いため、observe_all用の
        #   hearing_target（parent_chunk_id_plus1はparent_spoke時のみ）は
        #   run/trainer.py _world_predictor_step_ports と同じ規則
        #   （"parent_chunk_id_plus1": chunk_id_plus1 if parent_spoke else None）
        #   で組み直した。
        hearing_target = {"parent_spoke": parent_spoke,
                           "parent_chunk_id_plus1": chunk_id_plus1 if parent_spoke else None}
        body = rec.get("body", {}) or {}
        act = body.get("act")
        body_target = {"act": act}

        obs = wp.observe_all(vision_io, hearing_target, body_target)

        attended_id = rec.get("attended_id")
        if attended_id is not None and attended_id in obs["by_file"]:
            d = obs["by_file"][attended_id]
            if d.get("err_vec") is not None and d.get("err_state") is not None:
                age_sec = (i - first_seen[attended_id]) * 0.1
                label = _bin_label(age_sec)
                if label is not None:
                    acc = bins_acc[label]
                    acc["err_vec_sum"] += d["err_vec"]
                    acc["err_vec_n"] += 1
                    acc["err_state_sum"] += d["err_state"]
                    acc["err_state_n"] += 1

        hearing_input = {"parent_spoke": parent_spoke, "chunk_id_plus1": chunk_id_plus1,
                          "time_since_parent": hearing.get("time_since_parent", 1.0)}
        body_input = {"act": act}
        wp.predict_all(vision_io, hearing_input, body_input)

        prev_attended = attended_id

    result = {}
    for label in AGE_BIN_LABELS:
        acc = bins_acc[label]
        n_vec, n_state = acc["err_vec_n"], acc["err_state_n"]
        err_vec_mean = acc["err_vec_sum"] / n_vec if n_vec else None
        err_state_mean = acc["err_state_sum"] / n_state if n_state else None
        result[label] = {
            "n": n_vec,
            "err_vec_mean": err_vec_mean,
            "err_state_mean": err_state_mean,
            "err_vec_rel_mean": (err_vec_mean / var_vec_mean) if err_vec_mean is not None else None,
            "err_state_rel_mean": (err_state_mean / var_state_mean) if err_state_mean is not None else None,
        }
    return result


def _setup_font():
    """日本語が文字化けしないようにする（F/scripts/f89_world_predictor_plot.py
    のsetup_font()と同じ探し方）。"""
    from matplotlib import font_manager, rcParams
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ("BIZ UDGothic", "Yu Gothic", "Meiryo", "MS Gothic"):
        if c in names:
            rcParams["font.family"] = c
            return c
    return None


def make_plot(out_png, variants, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    x = list(range(len(AGE_BIN_LABELS)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for variant in variants:
        res = results[variant]
        y_vec_rel = [res[label]["err_vec_rel_mean"] for label in AGE_BIN_LABELS]
        y_state = [res[label]["err_state_mean"] for label in AGE_BIN_LABELS]
        axes[0].plot(x, y_vec_rel, marker="o", label=variant)
        axes[1].plot(x, y_state, marker="o", label=variant)
    axes[0].set_title("年齢別 err_vec（相対値）")
    axes[0].set_xlabel("注意中の物の年齢（秒）")
    axes[0].set_ylabel("err_vec / obj_vec要素分散平均")
    axes[1].set_title("年齢別 err_state")
    axes[1].set_xlabel("注意中の物の年齢（秒）")
    axes[1].set_ylabel("err_state")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_BIN_LABELS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pkl_path = os.path.abspath(args.pkl)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ALL_VARIANTS:
            raise ValueError(f"知らない変種: {v}（使える変種: {ALL_VARIANTS}）")

    out_dir = os.path.abspath(args.out) if args.out else os.path.dirname(pkl_path)
    os.makedirs(out_dir, exist_ok=True)

    records = load_records(pkl_path)
    var_vec_mean, var_state_mean = compute_variance_stats(records)
    act_dim, n_chunks = act_dim_and_n_chunks(records)

    results = {}
    timings = {}
    for variant in variants:
        t0 = time.time()
        results[variant] = run_variant(variant, records, act_dim, n_chunks,
                                        var_vec_mean, var_state_mean)
        elapsed = time.time() - t0
        timings[variant] = elapsed
        print(f"[f93] {variant}: {elapsed:.2f}秒（{len(records)}tick）")

    out_json = {
        "pkl": pkl_path,
        "n_ticks": len(records),
        "act_dim": act_dim,
        "n_chunks": n_chunks,
        "var_vec_mean": var_vec_mean,
        "var_state_mean": var_state_mean,
        "variants": results,
        "elapsed_sec": timings,
    }
    json_path = os.path.join(out_dir, "結果_切り分け.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(out_json, fp, ensure_ascii=False, indent=2)

    png_path = os.path.join(out_dir, "世界の予測器_切り分け.png")
    make_plot(png_path, variants, results)

    print(f"[f93] 出力: {json_path}")
    print(f"[f93] 出力: {png_path}")


if __name__ == "__main__":
    main()
