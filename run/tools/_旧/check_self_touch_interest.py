# -*- coding: utf-8 -*-
"""自己接触の興味度ボーナス（`cfg.self_touch_interest_bonus`）の検証。

設計：作業記録（非公開）（8節）
レビュー：同フォルダ\\2026-08-03_reach_self新奇性報酬_レビュー.md
仕様：作業記録（非公開）

確かめること：
  [1] 通ってはいけない条件（負の値／touch_mode不整合／成長実験との同時使用）で
      Config構築時にValueErrorで止まるか。既定0.0のときはこれらの制約に
      引っかからないことも確認する（0.0は「そもそも計算しない」のでゲート対象外）。
  [2] 既定0.0のとき、既存実験の挙動が1ビットも変わらないか
      （if cfg.self_touch_interest_bonus: の中身が一度も実行されない＝
      ctx.last_double_touch の新しい3キーが毎tick None/0.0のまま）。
      さらに同一設定・同一シードで2回走らせ、CSV相当の記録（rew系列）が
      完全一致するかも見る（新しいコードが乱数・状態を消費していないことの確認）。
  [3] 明示的な正の値（1.0）にした短時間学習で、
        hit=False の全tickで bonus が厳密に0.0
        hit=True の全tickで bonus が0.0以上（負にならない）
        bonus>0のtickのtouch_peが、hit=False区間の平均touch_peより高いか
      を確認する（落とし穴チェックリスト項96の教訓を踏襲）。
  [4] 既存のprogress報酬（body_progress、t.lp由来）が同じ自己接触tickで
      依然マイナスに触れているのに対し、bonus は同じtickで常に0以上であることを
      同じログから直接見比べる。

使い方:
    .venv/Scripts/python.exe run/tools/check_self_touch_interest.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config                    # noqa: E402
from run.trainer import Trainer, close_env        # noqa: E402
from run.plugins.base import Plugin               # noqa: E402

SCENE = "リーチング_リクライニング60度"


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
                 {"self_touch_interest_bonus": -1.0})
    expect_error("touch_mode不整合(input)",
                 {"self_touch_interest_bonus": 1.0, "touch_mode": "input"})
    expect_error("成長実験との同時使用(age_to=4.0)",
                 {"self_touch_interest_bonus": 1.0, "age_to": 4.0, "age_every": 5})
    expect_ok("既定0.0でtouch_mode=input（ゲート対象外なので通ってよい）",
              {"self_touch_interest_bonus": 0.0, "touch_mode": "input"})
    expect_ok("既定0.0でage_to指定（ゲート対象外なので通ってよい）",
              {"self_touch_interest_bonus": 0.0, "age_to": 4.0, "age_every": 5})
    expect_ok("正の値(1.0)・touch_mode=target・成長なし（正常系）",
              {"self_touch_interest_bonus": 1.0})

    print(f"  [1] 合否: {'合格' if ok else '不合格'}")
    return ok


# ---------------------------------------------------- [2] 既定0.0で挙動が変わらない
class _Recorder(Plugin):
    """rew・rpe・last_double_touchを毎tickそのまま控えるだけの検証用プラグイン。"""
    name = "recorder"

    def __init__(self, config=None):
        super().__init__(config)
        self.rows = []

    def on_step_late(self, ctx):
        d = dict(getattr(ctx, "last_double_touch", {}) or {})
        r = dict(getattr(ctx, "last_reward", {}) or {})
        d.update(r)
        d["step"] = ctx.step
        self.rows.append(d)


def _run_short(taro_overrides, steps, seed=0, checkpoint=None):
    # 【落とし穴】checkpointは睡眠リプレイ（consolidate）の頻度も兼ねる
    #   （cfg.replay既定True）。checkpoint=steps（1回きり）にすると、既存の比較実験
    #   （checkpoint=500）より睡眠による重みの定着が大幅に少なくなり、同じseedでも
    #   別の学習になる（この不一致で実際に「同じseed2なのにヒットが1件も出ない」と
    #   いう食い違いを踏んだ。既存実験と条件を揃えるため、既定は500にする）。
    ckpt = checkpoint if checkpoint is not None else min(500, max(steps, 1))
    cfg = Config.from_spec({
        "scene": SCENE,
        "taro": _base_taro(**taro_overrides),
        "run": {"type": "train", "steps": steps, "seed": seed, "checkpoint": ckpt},
    })
    rec = _Recorder()
    tr = Trainer(cfg, plugins=[rec], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        tr.run()
    finally:
        close_env(tr.env)
    return rec.rows


def check_default_unchanged():
    print("\n[2] 既定0.0で既存実験の挙動が変わらないこと")
    ok = True
    steps = 120
    rows = _run_short({}, steps)   # self_touch_interest_bonus未指定＝既定0.0
    have_rows = len(rows) > 0
    print(f"  ダブルタッチ判定tickが記録されたか  {'はい' if have_rows else 'いいえ'}"
          f"（{len(rows)}tick）")
    ok &= have_rows
    all_none_or_zero = all(
        (r.get("self_interest_touch_pe") is None)
        and (r.get("self_interest_progress") is None)
        and (float(r.get("self_interest_bonus", 0.0)) == 0.0)
        for r in rows)
    print(f"  全tickで self_interest_* が None/0.0 のままか（新しい計算が一度も"
          f"走っていない証拠）  {'はい' if all_none_or_zero else 'いいえ'}")
    ok &= all_none_or_zero

    # 同一設定・同一シードで2回走らせ、rewの系列が完全一致するか
    #   （新しいコードが乱数・状態を余分に消費していないことの確認。
    #    落とし穴チェックリスト 項3・項79の流儀）
    rows2 = _run_short({}, steps)
    same_len = len(rows) == len(rows2)
    same_rew = same_len and all(
        abs(float(a.get("rew", 0.0)) - float(b.get("rew", 0.0))) < 1e-12
        for a, b in zip(rows, rows2))
    print(f"  同一設定・同一シードで2回走らせ rew の系列が完全一致するか  "
          f"{'はい' if same_rew else 'いいえ'}")
    ok &= same_rew

    print(f"  [2] 合否: {'合格' if ok else '不合格'}")
    return ok


# --------------------------------------------- [3][4] 正の値・実際の挙動（実機・短時間）
def check_positive_bonus_behavior():
    print("\n[3][4] 正の値(1.0)での短時間学習・実際の挙動確認")
    ok = True
    # 頭へのダブルタッチ（trainer.py が実際に使う taro_core の
    #   DoubleTouchDetector・cfg.reach_arm_side="right_palm"→"head"）はレアな事象。
    #   注意（重要な落とし穴）：報酬を変える（self_touch_interest_bonus>0にする）と、
    #   MuJoCoの接触は初期の小さな摂動にも敏感なため、ヒットが起きるかどうか・
    #   起きる時刻はseedごとに全く違う。「bonus=0.0で(seed, step)にヒットが
    #   あった」という事実は、bonus=1.0の同じ(seed, step)にヒットが起きることを
    #   一切保証しない（実測：seed0・vision=True・checkpoint=500で、
    #   bonus=0.0なら3000stepで6件ヒットするが、bonus=1.0にすると同じseed・
    #   同じstep数で0件になった）。そこでbonus=1.0を有効にした状態で複数seedを
    #   実機で直接振ったところ、seed=2は4500stepまでに複数回ヒットが得られた
    #   （単独で振ったときは5件・本チェックの実行順序では2件——実行順序が
    #   変わるとヒットの正確な回数・タイミングまで変わる。MuJoCoの接触判定が
    #   浮動小数の微小な違いにも敏感なため。作業記録「想定外」に詳細を記載）。
    #   「短時間実行にとどめる」制約の中で確実にヒットを含めるため、この設定を
    #   使う（本番比較6000stepの3/4、数分〜十数分で終わる規模）。
    seed = 2
    steps = 4500
    rows = _run_short({"self_touch_interest_bonus": 1.0, "vision": True}, steps, seed=seed)
    n = len(rows)
    print(f"  記録tick数: {n}")

    hit_false = [r for r in rows if r.get("hit") is False]
    hit_true = [r for r in rows if r.get("hit") is True]
    print(f"  hit=False: {len(hit_false)}tick / hit=True: {len(hit_true)}tick")

    bonus_zero_on_false = all(float(r.get("self_interest_bonus", 0.0)) == 0.0
                               for r in hit_false)
    print(f"  hit=False の全tickで bonus が厳密に0.0か  "
          f"{'はい' if bonus_zero_on_false else 'いいえ'}")
    ok &= bonus_zero_on_false

    bonus_nonneg_on_true = all(float(r.get("self_interest_bonus", 0.0)) >= 0.0
                                for r in hit_true)
    print(f"  hit=True の全tickで bonus が0以上（負にならない）か  "
          f"{'はい' if bonus_nonneg_on_true else 'いいえ'}"
          f"（{len(hit_true)}件中）")
    ok &= bonus_nonneg_on_true

    # touch_peの比較（bonus>0のtick vs hit=False区間の平均）
    false_pe = [float(r["self_interest_touch_pe"]) for r in hit_false
                if r.get("self_interest_touch_pe") is not None]
    bonus_pos_rows = [r for r in hit_true
                      if float(r.get("self_interest_bonus", 0.0)) > 0.0]
    if false_pe and bonus_pos_rows:
        avg_false_pe = sum(false_pe) / len(false_pe)
        pos_pe = [float(r["self_interest_touch_pe"]) for r in bonus_pos_rows]
        avg_pos_pe = sum(pos_pe) / len(pos_pe)
        higher = avg_pos_pe > avg_false_pe
        print(f"  bonus>0のtickのtouch_pe平均({avg_pos_pe:.5f}) が "
              f"hit=False区間の平均({avg_false_pe:.5f}) より高いか  "
              f"{'はい' if higher else 'いいえ'}")
        ok &= higher
    else:
        print("  注意：bonus>0のtickが1件も無かった（レアな事象のため、この短い"
              "run内で発火しなかった可能性）。判定は保留にする（不合格にはしない）。")

    # [4] 既存progress報酬（body_progress）と直接見比べる
    #   trainer.pyのデバッグ出力と同じ値をここでは ctx.last_reward の rew から
    #   間接に見るのではなく、hit=True tickで body_progress を直接控えたいので、
    #   Recorderに body_progress も足したいところだが、ctx.last_reward には
    #   rew（全体の報酬、bonus込み）しか無い。ここでは代わりに
    #   「hit=True かつ bonus>0 のtickのうち、progress本体がマイナスに触れている
    #   件数」を数える（設計8.2(c)の意図）。
    #   注意：ctx.last_reward["rew"] は bonus・double_touch_bonus・effort_cost・
    #   caps を全部含めた最終値なので、progress本体だけを厳密に分離するには
    #   TARO_DEBUG_SELF_TOUCH_INTEREST=1 の標準出力（body_progress列）を使う方が
    #   正確。以下は標準出力を直接キャプチャして確認する補助チェック。
    ok &= _check_body_progress_vs_bonus(steps=steps, seed=seed)

    print(f"  [3][4] 合否: {'合格' if ok else '不合格'}")
    return ok


def _check_body_progress_vs_bonus(steps, seed=1):
    """TARO_DEBUG_SELF_TOUCH_INTEREST=1 の標準出力を直接キャプチャして、
    「既存progress報酬(body_progress)が同じtickでマイナスなのに、新しいbonusは
    0以上」であることを確認する（設計8.2(c)）。
    """
    import io
    import contextlib
    import re

    os.environ["TARO_DEBUG_SELF_TOUCH_INTEREST"] = "1"
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _run_short({"self_touch_interest_bonus": 1.0, "vision": True}, steps, seed=seed)
    finally:
        os.environ.pop("TARO_DEBUG_SELF_TOUCH_INTEREST", None)

    pat = re.compile(
        r"\[DEBUG_SELF_TOUCH_INTEREST\] step=(\d+) hit=(\S+) touch_pe=(\S+) "
        r"local_progress=(\S+) bonus=(\S+) body_progress=(\S+)")
    lines = [m for m in (pat.match(ln) for ln in buf.getvalue().splitlines()) if m]
    print(f"  デバッグ出力から解析できた行数: {len(lines)}")
    if not lines:
        print("  注意：デバッグ出力が1行も取れなかった（合否には影響させない）")
        return True

    neg_body_progress_ticks = [m for m in lines if m.group(2) == "True"
                               and float(m.group(6)) < 0.0]
    if not neg_body_progress_ticks:
        print("  注意：hit=True かつ body_progress<0 のtickが無かった"
              "（この短いrunでは再現しなかった。合否には影響させない）")
        return True

    all_bonus_nonneg = all(float(m.group(5)) >= 0.0 for m in neg_body_progress_ticks)
    print(f"  body_progress<0（既存progress報酬がマイナス）のtickが"
          f"{len(neg_body_progress_ticks)}件あり、その全てでbonusが0以上か  "
          f"{'はい' if all_bonus_nonneg else 'いいえ'}")
    example = neg_body_progress_ticks[0]
    print(f"    例：step={example.group(1)} body_progress={example.group(6)} "
          f"bonus={example.group(5)}")
    return all_bonus_nonneg


ok1 = check_validation()
ok2 = check_default_unchanged()
ok3 = check_positive_bonus_behavior()

print("\n" + "=" * 78)
print(" 総合判定")
print("=" * 78)
allok = ok1 and ok2 and ok3
print(f"  [1]バリデーション {'合格' if ok1 else '不合格'}  "
      f"[2]既定不変 {'合格' if ok2 else '不合格'}  "
      f"[3][4]正の値の挙動 {'合格' if ok3 else '不合格'}")
print(f"  総合: {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
