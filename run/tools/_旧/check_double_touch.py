# -*- coding: utf-8 -*-
"""double_touch プラグイン（run/plugins/common/double_touch.py）の検証。

仕様：作業記録（非公開）

run/tools/check_reach_success.py と同じ流儀（検証の落とし穴チェックリスト）：
  ①答えの分かっている入力（合成データ）で判定ロジックを確かめる
  ②太郎の脳・環境は作る（build）が、学習（tr.run()）は一度も呼ばない
  ③「通ってはいけない条件」で正しく止まるかを確かめる
  ④体を作り直す（成長）ときに落ちないかを確かめる
を行う。学習そのもの（run/main.py train / tr.run()）は一切呼ばない
（この検証スクリプト自体は。3節の短時間実行は別途、実験ファイル経由で行う）。

使い方:
    .venv/Scripts/python.exe run/tools/check_double_touch.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np                                    # noqa: E402
import torch                                           # noqa: E402

from run.config import Config                          # noqa: E402
from run.trainer import Trainer, close_env              # noqa: E402
from run.plugins.common.double_touch import DoubleTouch, DEFAULT_THRESHOLD  # noqa: E402

print("=" * 78)
print(" double_touch プラグインの検証")
print("=" * 78)

all_ok = True


# ------------------------------------------------------------------ ①判定ロジック（合成データ）
def check_threshold_logic():
    """toucher・touched の presence の組み合わせから「ダブルタッチ」を
    正しく判定できるか。答えの分かった値で確認する。
    """
    print("\n[1] ダブルタッチの判定ロジック（合成データ）")
    ok = True

    class FakeTouchModule:
        group_names = ["head", "chest", "left_palm", "right_palm", "other"]

        def __init__(self):
            self._presence = {}

        def set_presence(self, **kw):
            self._presence = kw

        def part_features(self, touch_flat):
            g = len(self.group_names)
            feat = torch.zeros(g, 5)
            for nm, v in self._presence.items():
                gi = self.group_names.index(nm)
                feat[gi, 0] = v
            return feat

    class FakeFusion:
        def __init__(self, touch):
            self.touch = touch

    class FakeCfg:
        reach_arm_side = "right"

    class FakeTaro:
        def __init__(self, groups, touch):
            self.reach_touch_groups = groups
            self.target_fusion = FakeFusion(touch)
            self.cfg = FakeCfg()

    tm = FakeTouchModule()
    fake_taro = FakeTaro(["head", "chest", "left_palm"], tm)
    plugin = DoubleTouch({"threshold": 0.5})

    class FakeCtx:
        brain = object()   # None でなければよい
        dt = 0.05
        taro = fake_taro
        last = None

    ctx = FakeCtx()
    plugin.setup(ctx)
    ok &= (plugin.toucher == "right_palm")
    print(f"  toucher が reach_arm_side から right_palm になるか  "
          f"{'はい' if plugin.toucher == 'right_palm' else 'いいえ'}（実測 {plugin.toucher}）")

    # 1歩目：toucherだけ触れていて、touched側はどこも触れていない → 成立しない
    tm.set_presence(right_palm=0.9, head=0.0, chest=0.0, left_palm=0.0)
    ctx.last = {"obs_out": {"touch": np.zeros(12, dtype=np.float32)}}
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    toucher_alone_not_counted = all(m[f"dtouch_count_{nm}"] == 0 for nm in fake_taro.reach_touch_groups)
    print(f"  toucherだけ触れている（touched側は0）→ ダブルタッチ0件のままか  "
          f"{'はい' if toucher_alone_not_counted else 'いいえ'}")
    ok &= toucher_alone_not_counted

    # 2歩目：touchedだけ触れていて、toucherは触れていない → 成立しない
    tm.set_presence(right_palm=0.0, head=0.9, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    touched_alone_not_counted = (m["dtouch_count_head"] == 0)
    print(f"  touchedだけ触れている（toucherは0）→ ダブルタッチ0件のままか  "
          f"{'はい' if touched_alone_not_counted else 'いいえ'}")
    ok &= touched_alone_not_counted

    # 3歩目：toucher=0.51・touched(head)=0.49 → 片方だけしきい値未満なので成立しない
    tm.set_presence(right_palm=0.51, head=0.49, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    just_below_not_counted = (m["dtouch_count_head"] == 0)
    print(f"  toucher=0.51・touched(head)=0.49（片方だけしきい値未満）→ 成立しないか  "
          f"{'はい' if just_below_not_counted else 'いいえ'}")
    ok &= just_below_not_counted
    raw_recorded = abs(m["dtouch_presence_max_head"] - 0.49) < 1e-6
    print(f"  touched(head)=0.49 の生の値がCSV列に残るか（0.49と一致）  "
          f"{'はい' if raw_recorded else 'いいえ'}（実測 {m['dtouch_presence_max_head']}）")
    ok &= raw_recorded
    toucher_raw_recorded = abs(m["dtouch_toucher_right_palm_presence_max"] - 0.51) < 1e-6
    print(f"  toucher(right_palm)=0.51 の生の値がCSV列に残るか（0.51と一致）  "
          f"{'はい' if toucher_raw_recorded else 'いいえ'}"
          f"（実測 {m['dtouch_toucher_right_palm_presence_max']}）")
    ok &= toucher_raw_recorded

    # 4歩目：toucher=0.51・touched(head)=0.51 → 両方しきい値超えなので成立する
    tm.set_presence(right_palm=0.51, head=0.51, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    above_thresh_counted = (m["dtouch_count_head"] == 1)
    print(f"  toucher=0.51・touched(head)=0.51（両方しきい値超）→ 成立するか  "
          f"{'はい' if above_thresh_counted else 'いいえ'}")
    ok &= above_thresh_counted
    first_step_recorded = ("dtouch_first_step" in m) and (m["dtouch_first_step"] == 4)
    print(f"  初めて成立した回（4歩目）で dtouch_first_step=4 と記録されるか  "
          f"{'はい' if first_step_recorded else 'いいえ'}（実測 {m.get('dtouch_first_step')}）")
    ok &= first_step_recorded

    # 5歩目：もう成立しなくても、達成済みの記録（4）は変わらない
    tm.set_presence(right_palm=0.0, head=0.0, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    stays = ("dtouch_first_step" in m) and (m["dtouch_first_step"] == 4)
    print(f"  達成後は値が4のまま変わらないか  {'はい' if stays else 'いいえ'}"
          f"（実測 {m.get('dtouch_first_step')}）")
    ok &= stays

    print(f"  [1] 合否: {'合格' if ok else '不合格'}")
    return ok


# --------------------------------------------------------- ②「通ってはいけない条件」で止まるか
def check_guards():
    """reach_self を使っていない実験・measureモードで、正しくエラーで止まるか。"""
    print("\n[2] 通ってはいけない条件で正しく止まるか")
    ok = True

    class FakeCtx:
        brain = None     # measure相当（脳が無い）
        taro = None

    p = DoubleTouch({})
    try:
        p.setup(FakeCtx())
        raised = False
    except ValueError:
        raised = True
    print(f"  脳が無い(measure相当)で ValueError になるか  {'はい' if raised else 'いいえ'}")
    ok &= raised

    class FakeCtx2:
        brain = object()
        taro = None       # reach_self 未設定（taro自体が無い）

    try:
        p2 = DoubleTouch({})
        p2.setup(FakeCtx2())
        raised2 = False
    except ValueError:
        raised2 = True
    print(f"  taro が無い（reach_self未設定）で ValueError になるか  "
          f"{'はい' if raised2 else 'いいえ'}")
    ok &= raised2

    print(f"  [2] 合否: {'合格' if ok else '不合格'}")
    return ok


# ---------------------------------- ③実機（MuJoCo）：学習せず初期姿勢で接触がほぼ0か
def check_initial_pose_near_zero():
    """学習を一切回さず（tr.run() を呼ばない）、体だけ作って零行動で数十tick進め、
    ダブルタッチがほぼ0であることを確認する（reach_success.py と同じ流儀）。
    """
    print("\n[3] 学習を回さず、初期姿勢付近でダブルタッチがほぼ0か（実機）")
    cfg = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False},
        "run": {"steps": 40, "seed": 0, "checkpoint": 40, "K": 10},
    })
    plugin = DoubleTouch({"threshold": DEFAULT_THRESHOLD})
    tr = Trainer(cfg, plugins=[plugin], verbose=False, log_row=lambda row: None)
    ok = True
    try:
        tr.build()
        env = tr.env
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        if float(env.action_space.low[0]) >= 0.0:
            zero[:] = 0.0     # 筋肉モード=[0,1]。0=脱力
        n_ticks = 40
        for i in range(n_ticks):
            for _ in range(cfg.K):
                obs, _r, term, trunc, _info = env.step(zero)
                if term or trunc:
                    obs, _ = env.reset()
            tr.ctx.step = i + 1
            tr.ctx.last = {"obs_in": None, "obs_out": obs}
            plugin.on_step(tr.ctx)
        rep = plugin.report(tr.ctx)
        print(f"  {n_ticks}tick、零行動（力を入れない）で進めた結果:")
        for k, v in rep.items():
            print(f"    {k}: {v}")
        total = rep["ダブルタッチ回数_通し_部位別"]
        # 【注意・2026-08-02】reach_success.py の検証（[3]）と同じ既知の限界：
        #   chest は reset直後から body_support（座面）に接触してpresence≈1.0で
        #   飽和している。toucher(手のひら)がまだ動いていない初期姿勢では
        #   toucher presenceが0のはずなので、chestが飽和していてもダブルタッチは
        #   0件のままのはず（両方が閾値超のときだけ成立するため）。
        all_zero = all(v == 0 for v in total.values())
        print(f"  零行動・初期姿勢でダブルタッチが0件のままか  "
              f"{'はい' if all_zero else 'いいえ（要確認）'}")
        ok &= all_zero
    except Exception as e:   # noqa: BLE001
        print(f"  例外で失敗: {type(e).__name__}: {e}")
        ok = False
        raise
    finally:
        close_env(tr.env)
    print(f"  [3] 合否: {'合格' if ok else '不合格（要目視で原因確認）'}")
    return ok


# ---------------------------------------------- ④体を作り直しても落ちないか（成長）
def check_body_change_survives():
    """体を作り直す（成長）ときに double_touch が例外を出さずに追従するか。"""
    print("\n[4] 体を作り直しても（成長）落ちないか")
    cfg = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False, "age_months": 0.0},
        "run": {"steps": 40, "seed": 0, "checkpoint": 40, "K": 10},
    })
    plugin = DoubleTouch({})
    tr = Trainer(cfg, plugins=[plugin], verbose=False, log_row=lambda row: None)
    ok = True
    try:
        tr.build()
        before_toucher = plugin.toucher
        before_touched = list(plugin.touched)
        tr._regrow(4.0)      # 0ヶ月 → 4ヶ月へ体を作り直す（学習はしない）
        after_toucher = plugin.toucher
        after_touched = list(plugin.touched)
        # on_body_change は trainer._regrow が全プラグインへ自動で呼ぶ
        #   （run/trainer.py 293-294行）。ここで例外が出ずに来られていること自体が確認。
        same = (before_toucher == after_toucher) and (before_touched == after_touched)
        print(f"  作り直し前後で対象部位は変わらないか  {'はい' if same else 'いいえ'}"
              f"（前 toucher={before_toucher}/touched={before_touched}"
              f" 後 toucher={after_toucher}/touched={after_touched}）")
        ok &= same
        env = tr.env
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, _ = env.reset()
        for _ in range(cfg.K):
            obs, _r, _te, _tr, _info = env.step(zero)
        tr.ctx.step = 1
        tr.ctx.last = {"obs_in": None, "obs_out": obs}
        plugin.on_step(tr.ctx)
        m = plugin.metrics(tr.ctx)
        print(f"  作り直した体で on_step / metrics が例外なく動くか  はい（{m}）")
    except Exception as e:   # noqa: BLE001
        print(f"  例外で失敗: {type(e).__name__}: {e}")
        ok = False
        raise
    finally:
        close_env(tr.env)
    print(f"  [4] 合否: {'合格' if ok else '不合格'}")
    return ok


ok1 = check_threshold_logic()
ok2 = check_guards()
ok3 = check_initial_pose_near_zero()
ok4 = check_body_change_survives()

print("\n" + "=" * 78)
print(" 総合判定")
print("=" * 78)
allok = ok1 and ok2 and ok3 and ok4
print(f"  [1]判定ロジック {'合格' if ok1 else '不合格'}  "
      f"[2]ガード {'合格' if ok2 else '不合格'}  "
      f"[3]初期姿勢ほぼ0 {'合格' if ok3 else '不合格'}  "
      f"[4]成長で落ちない {'合格' if ok4 else '不合格'}")
print(f"  総合: {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
