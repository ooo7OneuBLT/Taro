# -*- coding: utf-8 -*-
"""progress報酬のsurpriseボーナス（機構1、`cfg.progress_surprise_bonus`）の検証。

設計：作業記録（非公開）
      2026-08-04_progress報酬surpriseボーナス修正設計.md（担当の統合判断）
      同フォルダ\\作業中_案A/B/D_progress_surprise修正.md（各設計者の詳細）
仕様：実装担当からの2026-08-04付・しきい値方式（超過量型・ヒンジ）修正指示

注意：設計は「novelty」という語を使うが、太郎の実装が扱っている量は理論分類上
「surprise」（Barto, Mirolli & Baldassarre 2013）であり、self_touch_interest_bonusの
命名訂正と同じ理由で、この検証スクリプトでも一貫して「surprise」の語を使う。

確かめること：
  [1] バリデーション
      - progress_surprise_bonus=-1.0 → ValueError
      - progress_surprise_decay=1.5（bonus=1.0と同時） → ValueError
      - progress_surprise_var_tau=-0.1（bonus=1.0と同時） → ValueError
      - progress_surprise_threshold=-1.0（bonus=1.0と同時） → ValueError（新規）
      - 既定0.0でdecay/var_tau/thresholdが範囲外でも通ってよい（ゲートされているため）
      - 正の値(1.0)・decay=0.9・var_tau=0.99・threshold=4.0の正常系は通る
  [2] 既定OFF(0.0)で挙動が変わらないこと＋k=0.0で旧式と一致すること（新規）
      - 記録用プラグインでrew系列を記録し、同一設定・同一シードで2回走らせ完全一致を見る
      - LearningProgressを直接呼び出し、surprise_bonus=0.0のとき
        update()の返り値が現状の計算式(pe_slow_new-pe_fast_new)と一致することを確認する
      - surprise_threshold=0.0のとき、_surprise_traceが「旧式
        （max(dev,0)/(sqrt(var)+eps)をそのままEMAに通したもの）」と、resyncを呼ばない
        範囲で完全一致することを確認する（1000tick、Pure Pythonで旧式を再現したものと比較）
  [3] 正の値での挙動確認（contact_reward）＋しきい値方式の実データ検証（新規・最重要）
      - progress_surprise_bonus=0.0（基準）と1.0（有効時）で複数シード×4500stepを
        実行し、contact_rewardが出す自己接触tickの平均rewを比較する
      - _TraceRecorderで時系列順(step, category, z, frozen, trace)を記録し、
        シードごとにオフライン再計算してk候補[3.5, 4.0, 4.5, 5.0]それぞれの
        noneカテゴリtrace平均を出す。既定値(TARO_DEFAULTSのprogress_surprise_threshold)
        に対応する候補が0.05未満であることを合否判定に使う
  [3r] resync直後の確認（新規）
      - cfg.growsを使った短いrunでresyncイベントを複数回起こし、freeze中のtraceが
        厳密に0.0であること、freeze直後のz絶対値最大値が50を明確に下回ることを確認する
  [4] 目標B/C/Eの既存スクリプトのLearningProgress()呼び出しへの影響が無いことの確認
      （新規：surprise_thresholdへの言及チェックを追加）

使い方:
    .venv/Scripts/python.exe -u run/tools/check_progress_surprise.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config, TARO_DEFAULTS            # noqa: E402
from run.trainer import Trainer, close_env               # noqa: E402
from run.plugins.base import Plugin                       # noqa: E402
from learning_progress import LearningProgress             # noqa: E402

SCENE = "リーチング_リクライニング60度"

# 既定値はTARO_DEFAULTSから読む（ハードコードで重複させない）。
_DEFAULT_THRESHOLD = TARO_DEFAULTS["progress_surprise_threshold"][0]
_DEFAULT_DECAY = TARO_DEFAULTS["progress_surprise_decay"][0]


def _base_taro(**overrides):
    d = {"goal_babbling": True, "goal_space": "reach_self", "goal_trajectory": True,
         "reach_arm_side": "right", "touch": True, "somatosensory": True,
         "vision": False}
    d.update(overrides)
    return d


# ------------------------------------------------------------ [1] バリデーション
def check_validation():
    print("\n[1] 通ってはいけない条件でValueErrorになるか")
    ok = True

    def expect_error(label, taro_kwargs):
        nonlocal ok
        try:
            Config.from_spec({"scene": SCENE, "taro": _base_taro(**taro_kwargs),
                               "run": {"steps": 10, "seed": 0}})
            print(f"  {label} → エラーにならなかった（不合格）")
            ok = False
        except ValueError as e:
            print(f"  {label} → ValueError で止まった（合格）: {str(e).splitlines()[0]}")

    def expect_ok(label, taro_kwargs):
        nonlocal ok
        try:
            Config.from_spec({"scene": SCENE, "taro": _base_taro(**taro_kwargs),
                               "run": {"steps": 10, "seed": 0}})
            print(f"  {label} → エラーにならず通った（合格）")
        except ValueError as e:
            print(f"  {label} → 予期せずValueError（不合格）: {e}")
            ok = False

    expect_error("負の値(-1.0)",
                 {"progress_surprise_bonus": -1.0})
    expect_error("decay=1.5（bonus=1.0と同時）",
                 {"progress_surprise_bonus": 1.0, "progress_surprise_decay": 1.5})
    expect_error("var_tau=-0.1（bonus=1.0と同時）",
                 {"progress_surprise_bonus": 1.0, "progress_surprise_var_tau": -0.1})
    expect_error("しきい値が負(-1.0)（bonus=1.0と同時）",
                 {"progress_surprise_bonus": 1.0, "progress_surprise_threshold": -1.0})
    expect_ok("既定0.0でdecay/var_tau/thresholdが範囲外（ゲート対象外なので通ってよい）",
              {"progress_surprise_bonus": 0.0, "progress_surprise_decay": 1.5,
               "progress_surprise_var_tau": -0.1, "progress_surprise_threshold": -1.0})
    expect_ok("正の値(1.0)・decay=0.9・var_tau=0.99・threshold=4.0（正常系）",
              {"progress_surprise_bonus": 1.0, "progress_surprise_decay": 0.9,
               "progress_surprise_var_tau": 0.99, "progress_surprise_threshold": 4.0})
    expect_ok(f"しきい値の既定値({_DEFAULT_THRESHOLD})・bonus=1.0の正常系",
              {"progress_surprise_bonus": 1.0,
               "progress_surprise_threshold": _DEFAULT_THRESHOLD})

    print(f"  [1] 合否: {'合格' if ok else '不合格'}")
    return ok


# ---------------------------------------------------- [2] 既定0.0で挙動が変わらない
class _Recorder(Plugin):
    """rewを毎tickそのまま控えるだけの検証用プラグイン。"""
    name = "recorder"

    def __init__(self, config=None):
        super().__init__(config)
        self.rows = []

    def on_step_late(self, ctx):
        r = dict(getattr(ctx, "last_reward", {}) or {})
        r["step"] = ctx.step
        self.rows.append(r)


def _run_short(taro_overrides, steps, seed=0, checkpoint=None, extra_plugins=None):
    ckpt = checkpoint if checkpoint is not None else min(500, max(steps, 1))
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": _base_taro(**taro_overrides),
        "run": {"type": "train", "steps": steps, "seed": seed, "checkpoint": ckpt},
    })
    rec = _Recorder()
    plugins = [rec] + list(extra_plugins or [])
    tr = Trainer(cfg, plugins=plugins, verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        tr.run()
    finally:
        close_env(tr.env)
    return rec, tr


def _old_formula_trace(pe_series, decay=0.9, var_tau=0.99, gain=1.0, eps=1e-6,
                        freeze_ticks=100):
    """旧式（max(dev,0)/(sqrt(var)+eps)をそのままEMAに通したもの）をPure Pythonで
    再現する。k=0.0での新式との一致確認に使う（1節「k=0.0で旧式と一致する」の裏取り）。

    【2026-08-05追記】コンストラクタ直後のfreeze（learning_progress.py参照。
    _surprise_varが0.0から始まるため、1回目のupdate()でzが機械的に約10に固定される
    バグの修正）を、この参照実装にも反映した。反映しないと、この関数自体が
    「まだバグっていた頃」の旧式のままになり、k=0.0一致確認が常に不合格になる
    （バグを直したこと自体が原因の不一致であり、新式側の不具合ではない）。
    """
    pe_slow = 1.0
    var = 0.0
    trace = 0.0
    traces = []
    freeze_remaining = freeze_ticks
    for pe in pe_series:
        pe_slow_prev = pe_slow
        pe_slow = 0.99 * pe_slow + 0.01 * pe
        dev = pe - pe_slow_prev
        var = var_tau * var + (1.0 - var_tau) * dev ** 2
        if freeze_remaining > 0:
            freeze_remaining -= 1
            surprise = 0.0
        else:
            surprise = max(dev, 0.0) / (var ** 0.5 + eps)
        trace = decay * trace + gain * surprise
        traces.append(trace)
    return traces


def check_default_unchanged():
    print("\n[2] 既定0.0で既存実験の挙動が変わらないこと・k=0.0で旧式と一致すること")
    ok = True
    steps = 150

    rec1, _ = _run_short({}, steps)
    rec2, _ = _run_short({}, steps)
    rows1, rows2 = rec1.rows, rec2.rows
    same_len = len(rows1) == len(rows2)
    same_rew = same_len and all(
        abs(float(a.get("rew", 0.0)) - float(b.get("rew", 0.0))) < 1e-12
        for a, b in zip(rows1, rows2))
    print(f"  同一設定・同一シードで2回走らせ rew の系列が完全一致するか  "
          f"{'はい' if same_rew else 'いいえ'}（{len(rows1)}tick）")
    ok &= same_rew

    # LearningProgress単体で、surprise_bonus=0.0のときupdate()の返り値が
    #   現状の計算式(pe_slow_new-pe_fast_new)と一致することを確認する。
    lp = LearningProgress()   # surprise_bonus既定0.0
    import random
    rnd = random.Random(0)
    match_all = True
    pe_fast, pe_slow = lp.pe_fast, lp.pe_slow
    for _ in range(500):
        pe = rnd.uniform(0.0, 3.0)
        expected_fast = lp.tau_fast * pe_fast + lp.rate_fast * pe
        expected_slow = lp.tau_slow * pe_slow + lp.rate_slow * pe
        expected = expected_slow - expected_fast
        got = lp.update(pe)
        if abs(got - expected) > 1e-15:
            match_all = False
        pe_fast, pe_slow = expected_fast, expected_slow
    print(f"  surprise_bonus=0.0のupdate()が現状の計算式と500回すべて一致するか  "
          f"{'はい' if match_all else 'いいえ'}")
    ok &= match_all

    # k=0.0で旧式と厳密に一致するか（resyncを呼ばない範囲）。
    rnd2 = random.Random(42)
    pe_series = [1.0 + rnd2.uniform(-0.3, 0.3) for _ in range(1000)]
    old_traces = _old_formula_trace(pe_series)
    # 【なぜtraceを直接読むか】update()の戻り値はprogress(pe_slow-pe_fast+trace)なので、
    #   trace自体を比較するには_surprise_traceを直接読む必要がある。
    new_traces = []
    lp_new = LearningProgress(surprise_bonus=1.0, surprise_decay=0.9,
                               surprise_var_tau=0.99, surprise_threshold=0.0)
    for pe in pe_series:
        lp_new.update(pe)
        new_traces.append(lp_new._surprise_trace)
    max_diff = max(abs(a - b) for a, b in zip(old_traces, new_traces))
    k0_match = max_diff < 1e-9
    print(f"  k=0.0のとき、新式のtraceが旧式(1000tick)と一致するか（最大差={max_diff:.3e}）  "
          f"{'はい' if k0_match else 'いいえ'}")
    ok &= k0_match

    print(f"  [2] 合否: {'合格' if ok else '不合格'}")
    return ok


# ---------------------------------------------- [3] 正の値での挙動確認（contact_reward）
class _TraceRecorder(Plugin):
    """毎tick、実際の学習ループで動いている`ctx.taro.lp`の内部状態を、
    時系列順の1本のリストとして記録する検証用プラグイン。

    【なぜ時系列順の1本のリストか、2026-08-04】traceの漸化式（decayするEMA）は
    時間順序に依存するため、self/noneでリストを分けると後で正しく再計算できない
    （設計・案Aが明記した注意点）。カテゴリ分類は記録するが、リスト自体は
    分けない。オフライン再計算（_recompute_trace）を行うときに初めて
    カテゴリ別に集計する。

    【なぜ要るか、2026-08-04】従来の[3]「平常時にtraceがほぼゼロに保たれるか」は
    `LearningProgress`を単体で呼び出し、`baseline=1.0 + 一様ノイズ±0.02`という
    人工的なpe系列で作った"平常時"を判定に使っていた。だが実際の学習ループでは
    予測誤差(pe)がここまで穏やかではなく、`_surprise_trace`は実測で平均2.1〜3.5と
    大きく食い違っていた（逸脱リスト項⑫記録済み）。このクラスは、既に
    `_find_hitting_seed`が回しているrunに相乗りして、本物のpe系列から生まれた
    本物のzの値（しきい値に依存しない生データ）を記録する（実行を新たに増やさない）。
    """
    name = "trace_recorder"

    def __init__(self, config=None):
        super().__init__(config)
        self.rows = []   # 時系列順: (step, category, z, frozen, trace) のタプル

    def on_step_late(self, ctx):
        taro = getattr(ctx, "taro", None)
        lp = getattr(taro, "lp", None) if taro is not None else None
        z = getattr(lp, "_surprise_last_z", None) if lp is not None else None
        if z is None:
            return
        frozen = bool(getattr(lp, "_surprise_last_frozen", False))
        trace = float(getattr(lp, "_surprise_trace", 0.0))
        dt_info = getattr(ctx, "last_double_touch", None)
        self_hit = bool(dt_info["hit"]) if dt_info is not None else False
        self.rows.append((ctx.step, "self" if self_hit else "none", float(z), frozen, trace))


def _recompute_trace(rows, k, decay, gain):
    """rows: 1シード分の時系列順(step, category, z, frozen, trace)。
    しきい値kでtraceを再計算し、[(category, recomputed_trace), ...] を返す。

    注意：複数シードのrowsを1本の時系列に連結してはいけない（traceの漸化式は
    同一runの中でしか意味を持たない。シードをまたいで連結するとdecayが誤って
    別runの状態を引きずる）。この関数は必ずシードごとに個別に呼ぶこと。
    """
    trace = 0.0
    out = []
    for _, category, z, frozen, _ in rows:
        excess = 0.0 if frozen else max(z - k, 0.0)
        trace = decay * trace + gain * excess
        out.append((category, trace))
    return out


def _find_hitting_seed(bonus, steps, seed_candidates):
    """bonus有効の状態そのもので複数シードを振り、実際にヒットが起きる
    シードを実測して返す（0.0の実測をそのまま使い回さない。ノウハウ.md
    2026-08-03の教訓）。

    合わせて`_TraceRecorder`を相乗りさせ、本物のpe系列から生まれた
    zの実測値（時系列順のrows）も返す。decay/gain（実行時の設定）も
    呼び出し側が読めるようcfgを一緒に返す。
    """
    from run.plugins.common.contact_reward import ContactReward
    results = []
    for seed in seed_candidates:
        cr = ContactReward()
        trc = _TraceRecorder()
        # vision=Trueにする理由：check_self_touch_interest.py と同じ設定・同じシーンで
        #   自己接触が実際に起きることが2026-08-03に実測済み（seed=2で複数回ヒット）。
        _, tr = _run_short({"progress_surprise_bonus": bonus, "vision": True},
                            steps, seed=seed, extra_plugins=[cr, trc])
        n_self = cr.total_count.get("self", 0)
        rew_mean = (cr.total_rew_sum["self"] / n_self) if n_self else None
        results.append((seed, n_self, rew_mean, cr, trc, tr.cfg))
        print(f"    seed={seed} bonus={bonus} self接触tick数={n_self}"
              f"{'' if rew_mean is None else f' rew_mean={rew_mean:.5f}'}", flush=True)
    return results


def check_positive_bonus_behavior():
    print("\n[3] 正の値(1.0)での挙動確認（contact_reward）＋しきい値の実データ検証")
    ok = True
    steps = 4500
    seed_candidates = [0, 1, 2, 3]

    print("  bonus=0.0（基準）で複数シードを実測", flush=True)
    base_results = _find_hitting_seed(0.0, steps, seed_candidates)
    print("  bonus=1.0（有効時）で複数シードを実測"
          "（0.0の実測をそのまま使い回さない）", flush=True)
    bonus_results = _find_hitting_seed(1.0, steps, seed_candidates)

    base_hit = [(s, n, rm) for (s, n, rm, _, _, _) in base_results if n and n >= 3]
    bonus_hit = [(s, n, rm, cr) for (s, n, rm, cr, _, _) in bonus_results if n and n >= 3]

    if len(base_hit) < 3 or len(bonus_hit) < 3:
        print(f"  注意：ヒットが3シード分そろわなかった"
              f"（bonus=0.0: {len(base_hit)}件 / bonus=1.0: {len(bonus_hit)}件）。"
              "seed候補を増やす必要があるが、このrunでは判定を保留にする。")
    else:
        base_mean = sum(rm for (_, _, rm) in base_hit) / len(base_hit)
        bonus_mean = sum(rm for (_, _, rm, _) in bonus_hit) / len(bonus_hit)
        print(f"  bonus=0.0の自己接触tick平均rew（{len(base_hit)}シード平均）: "
              f"{base_mean:.5f}")
        print(f"  bonus=1.0の自己接触tick平均rew（{len(bonus_hit)}シード平均）: "
              f"{bonus_mean:.5f}")
        improved = bonus_mean > base_mean
        print(f"  bonus=1.0の方がbonus=0.0より0に近づく/改善しているか  "
              f"{'はい' if improved else 'いいえ'}")
        ok &= improved

    # ---- しきい値方式の実データ検証（新規・最重要）------------------------------
    #   複数のk候補で、シードごとにオフライン再計算し、noneカテゴリのtraceを集計する。
    print("  しきい値k候補ごとのnoneカテゴリtrace平均（4シード・シードごとに再計算してから集計）",
          flush=True)
    k_candidates = [3.5, 4.0, 4.5, 5.0]
    # decay/gainは実行時の設定と合わせる。ハードコードせず、実行に使ったcfgから読む。
    gain = 1.0   # bonus=1.0で回したので、trace更新式のgainはこの値
    decay = None
    per_k_none_values = {k: [] for k in k_candidates}
    per_seed_summaries = []
    for (seed, n_self, rew_mean, cr, trc, cfg_used) in bonus_results:
        if decay is None:
            decay = float(cfg_used.progress_surprise_decay)
        per_k_this_seed = {}
        for k in k_candidates:
            recomputed = _recompute_trace(trc.rows, k, decay, gain)   # シードごとに個別に再計算
            none_vals = [tr_ for (cat, tr_) in recomputed if cat == "none"]
            per_k_none_values[k].extend(none_vals)
            per_k_this_seed[k] = (sum(none_vals) / len(none_vals)) if none_vals else None
        per_seed_summaries.append((seed, per_k_this_seed))

    for seed, per_k_this_seed in per_seed_summaries:
        line = "  ".join(f"k={k}: {v:.5f}" if v is not None else f"k={k}: (noneなし)"
                          for k, v in per_k_this_seed.items())
        print(f"    seed={seed}  {line}", flush=True)

    k_means = {}
    for k in k_candidates:
        vals = per_k_none_values[k]
        k_means[k] = (sum(vals) / len(vals)) if vals else None
        if vals:
            print(f"  k={k}  noneカテゴリtrace平均（4シード合計{len(vals)}tick）: "
                  f"{k_means[k]:.6f}  {'合格(<0.05)' if abs(k_means[k]) < 0.05 else '不合格(>=0.05)'}")
        else:
            print(f"  k={k}  noneカテゴリのtickが0件（想定外）")

    default_mean = k_means.get(_DEFAULT_THRESHOLD)
    if default_mean is None:
        print(f"  注意：既定値k={_DEFAULT_THRESHOLD}がk候補に含まれていないか、"
              "noneデータが無かった（想定外）。判定を保留にする。")
    else:
        default_ok = abs(default_mean) < 0.05
        print(f"  既定値k={_DEFAULT_THRESHOLD}のnoneカテゴリtrace平均={default_mean:.6f}  "
              f"要件(0.05未満)を満たすか  {'はい' if default_ok else 'いいえ'}")
        ok &= default_ok

    # 平常時（bonusのrunから見た、しきい値方式ではない実行時そのままのtrace）も
    #   参考として表示する（実行時の既定k（TARO_DEFAULTSのprogress_surprise_threshold）で記録された実測trace列）。
    real_none_traces_runtime = []
    real_self_traces_runtime = []
    for (_, _, _, _, trc, _) in bonus_results:
        for (_, category, _, _, trace) in trc.rows:
            (real_none_traces_runtime if category == "none" else real_self_traces_runtime) \
                .append(trace)
    if real_none_traces_runtime:
        avg_runtime = sum(real_none_traces_runtime) / len(real_none_traces_runtime)
        print(f"  参考：実行時そのまま（既定k={_DEFAULT_THRESHOLD}）のnoneカテゴリtrace平均"
              f"（{len(real_none_traces_runtime)}tick）: {avg_runtime:.6f}")

    # 参考：人工的なpe系列（式の仕組み自体が動くかの参考のみ・総合判定には使わない）
    print("  参考（人工データ・式の仕組みの確認のみ。総合判定には使わない）：")
    lp = LearningProgress(surprise_bonus=1.0, surprise_decay=0.9, surprise_var_tau=0.99,
                           surprise_threshold=_DEFAULT_THRESHOLD)
    import random
    rnd = random.Random(1)
    baseline = 1.0
    traces = []
    for _ in range(2000):
        # 平常時＝ノイズだけの小さな変動（急上昇イベントなし）
        pe = baseline + rnd.uniform(-0.02, 0.02)
        lp.update(pe)
        traces.append(lp._surprise_trace)
    avg_trace_normal = sum(traces[-500:]) / 500
    small = abs(avg_trace_normal) < 0.05
    print(f"    人工データでの平常時2000tick後半500tickのtrace平均: "
          f"{avg_trace_normal:.6f}  "
          f"{'ほぼゼロ' if small else '大きい'}"
          "（参考値。合否判定には含めない）")

    # 突発的な急上昇イベントを1回混ぜると、traceが正に跳ね上がり、その後減衰することも確認
    lp2 = LearningProgress(surprise_bonus=1.0, surprise_decay=0.9, surprise_var_tau=0.99,
                            surprise_threshold=0.0)   # k=0.0：急上昇の検出感度を保つため
    rnd2 = random.Random(2)
    for _ in range(200):
        lp2.update(baseline + rnd2.uniform(-0.02, 0.02))
    trace_before_spike = lp2._surprise_trace
    lp2.update(baseline + 5.0)   # 急な驚き
    trace_at_spike = lp2._surprise_trace
    for _ in range(50):
        lp2.update(baseline + rnd2.uniform(-0.02, 0.02))
    trace_after_decay = lp2._surprise_trace
    spike_then_decay = (trace_at_spike > trace_before_spike) and \
                        (trace_after_decay < trace_at_spike)
    print(f"  急上昇イベントでtraceが跳ね上がり（{trace_before_spike:.5f}→"
          f"{trace_at_spike:.5f}）、その後減衰する（→{trace_after_decay:.5f}）か  "
          f"{'はい' if spike_then_decay else 'いいえ'}")
    ok &= spike_then_decay

    print(f"  [3] 合否: {'合格' if ok else '不合格'}")
    return ok, k_means


# ------------------------------------------------------- [3r] resync直後の確認
def check_resync_period():
    print("\n[3r] resync直後の確認（成長イベント直後の異常な過敏さが直っているか）")
    ok = True
    steps = 1600

    cr_unused = None  # contact_rewardはこの確認では使わない（growsとreach_selfの
                       # 組み合わせ自体は許可されている。self_touch_interest_bonusとは別機構）
    trc = _TraceRecorder()
    taro_overrides = {
        "progress_surprise_bonus": 1.0,
        "vision": True,
        # 【2026-08-04・実装担当が実行時に修正】age_ramp を指定し忘れており
        #   （既定0＝一瞬で切り替え）、age_start=0 だと"最初のcheckpoint(i=400)
        #   より前"にすでに目標月齢へ達しているため、regrow()の age 比較が
        #   常に「変化なし」になり resync が一度も起きなかった（初回実行で実測）。
        #   age_ramp=1000 にすると月齢が0→4ヶ月へ段階的に変わり、i=400/800/1200の
        #   3回のcheckpointそれぞれで異なる月齢になるので、3回resyncが起きる。
        "age_months": 0.0, "age_to": 4.0, "age_start": 0, "age_every": 400,
        "age_ramp": 1000,
    }
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": _base_taro(**taro_overrides),
        "run": {"type": "train", "steps": steps, "seed": 0, "checkpoint": steps},
    })
    rec = _Recorder()
    tr = Trainer(cfg, plugins=[rec, trc], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        tr.run()
    finally:
        close_env(tr.env)
    print(f"  実行完了：{len(trc.rows)}tick記録", flush=True)

    freeze_ticks = None
    # lp.surprise_freeze_ticksを読む（Trainer内のtaroから）。実行後もtr.taroが残るはず。
    lp = getattr(getattr(tr, "taro", None), "lp", None)
    if lp is not None:
        freeze_ticks = int(getattr(lp, "_surprise_freeze_ticks", 100))
    if freeze_ticks is None:
        freeze_ticks = 100
        print("  注意：lp._surprise_freeze_ticksを読めなかった。既定値100を使う（想定外）。")

    # resyncイベントの開始＝「frozen=Falseの直後にfrozen=Trueへ切り替わったtick」
    #   （先頭tickは除く）。
    events = []
    prev_frozen = None
    for i, (step, category, z, frozen, trace) in enumerate(trc.rows):
        if i == 0:
            prev_frozen = frozen
            continue
        if (not prev_frozen) and frozen:
            events.append(i)
        prev_frozen = frozen

    if not events:
        print("  想定外：resyncイベントが1件も検出できなかった。"
              "age_every等の設定を見直す必要がある。")
        return False, 0

    print(f"  検出したresyncイベント数: {len(events)}（tick位置: "
          f"{[trc.rows[i][0] for i in events]}）")

    max_z_after_all = []
    for ev_i in events:
        # freeze期間中（frozen=Trueの区間）の実測traceが厳密に0.0のままか
        freeze_end = ev_i
        while freeze_end < len(trc.rows) and trc.rows[freeze_end][3]:
            freeze_end += 1
        freeze_traces = [trc.rows[j][4] for j in range(ev_i, freeze_end)]
        freeze_all_zero = all(t == 0.0 for t in freeze_traces)
        print(f"    イベント@tick{trc.rows[ev_i][0]}: freeze区間長={len(freeze_traces)}  "
              f"freeze中traceが厳密に0.0か  "
              f"{'はい' if freeze_all_zero else 'いいえ（不合格）'}")
        ok &= freeze_all_zero

        # freeze直後、およそ100tick分の区間でのz絶対値の最大値
        window = trc.rows[freeze_end:freeze_end + 100]
        if not window:
            print(f"    イベント@tick{trc.rows[ev_i][0]}: freeze直後の区間データが無い"
                  "（runの終端に近い。想定外ではないが確認対象から除外）")
            continue
        max_z = max(abs(z) for (_, _, z, _, _) in window)
        max_z_after_all.append(max_z)
        under_bound = max_z < 50.0
        print(f"    イベント@tick{trc.rows[ev_i][0]}: freeze直後{len(window)}tickの"
              f"z絶対値最大値={max_z:.3f}  50を下回るか  "
              f"{'はい' if under_bound else 'いいえ（不合格）'}")
        ok &= under_bound

    print(f"  [3r] 合否: {'合格' if ok else '不合格'}")
    return ok, len(events)


# ------------------------------------------- [4] 既存呼び出しの確認
def check_existing_callers():
    print("\n[4] 目標B/C/Eの既存スクリプトのLearningProgress()呼び出し箇所の確認")
    ok = True
    targets = [
        (r"C\scripts\run_c_metrics_ac_lr.py", "LearningProgress"),
        (r"D\scripts\d0_selftouch.py", "LearningProgress"),
        (r"E\scripts\e_growth_train.py", "LearningProgress"),
    ]
    for rel, needle in targets:
        path = os.path.join(_R, rel)
        if not os.path.exists(path):
            print(f"  {rel} → ファイルが無い（想定外。要確認）")
            ok = False
            continue
        with open(path, encoding="utf-8") as fp:
            lines = fp.readlines()
        hit_lines = [(i + 1, ln.rstrip("\n")) for i, ln in enumerate(lines)
                     if needle in ln]
        if not hit_lines:
            print(f"  {rel} → LearningProgressへの言及が見つからない（想定外）")
            ok = False
            continue
        for lineno, text in hit_lines:
            touches_new_args = ("surprise_bonus" in text or "surprise_decay" in text
                                 or "surprise_var_tau" in text or "surprise_threshold" in text)
            print(f"  {rel}:{lineno}  {text.strip()}"
                  f"  {'（新引数に触れている！要確認）' if touches_new_args else '（新引数なし・影響なし）'}")
            ok &= not touches_new_args
    print(f"  [4] 合否: {'合格' if ok else '不合格'}")
    return ok


def _run_all():
    """全チェックを順に実行する（スクリプトとして直接実行されたときのみ）。

    、2026-08-04・実装担当が追加】if __name__=="__main__" で囲う前はモジュールとして
    importしただけで全チェック（[3]の重いTrainer実行を含む）が自動実行されてしまい、
    [3r]の修正確認など一部のチェックだけを再実行したいときにも毎回[3]を
    数十分かけて回さざるを得なかった（実際に2026-08-04の検証でこの問題に当たった）。
    """
    ok1 = check_validation()
    ok2 = check_default_unchanged()
    ok3, k_means = check_positive_bonus_behavior()
    ok3r, n_resync_events = check_resync_period()
    ok4 = check_existing_callers()

    print("\n" + "=" * 78)
    print(" 総合判定")
    print("=" * 78)
    allok = ok1 and ok2 and ok3 and ok3r and ok4
    print(f"  [1]バリデーション {'合格' if ok1 else '不合格'}  "
          f"[2]既定不変+k=0一致 {'合格' if ok2 else '不合格'}  "
          f"[3]正の値の挙動 {'合格' if ok3 else '不合格'}  "
          f"[3r]resync直後({n_resync_events}件) {'合格' if ok3r else '不合格'}  "
          f"[4]既存呼び出し {'合格' if ok4 else '不合格'}")
    print(f"  k候補ごとのnnoneカテゴリ个trace平均: "
          + "  ".join(f"k={k}: {v:.5f}" if v is not None else f"k={k}: (データなし)"
                      for k, v in (k_means or {}).items()))
    print(f"  総合: {'合格' if allok else '不合格'}")
    if not allok:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
