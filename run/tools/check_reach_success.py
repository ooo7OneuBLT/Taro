# -*- coding: utf-8 -*-
"""reach_success プラグイン（run/plugins/common/reach_success.py）の検証。

仕様：作業記録（非公開）

【注意：ここは実験を回さない】検証の落とし穴チェックリストの流儀に沿って、
  ①答えの分かっている入力（合成データ）で判定ロジックを確かめる
  ②太郎の脳・環境は作る（build）が、学習（tr.run()）は一度も呼ばない
  ③「通ってはいけない条件」で正しく止まるかを確かめる
  ④体を作り直す（成長）ときに落ちないかを確かめる
を行う。学習そのもの（run/main.py train / tr.run()）は一切呼ばない。

使い方:
    .venv/Scripts/python.exe run/tools/check_reach_success.py
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
from run.context import Ctx                             # noqa: E402
from run.plugins.common.reach_success import ReachSuccess, DEFAULT_THRESHOLD  # noqa: E402

print("=" * 78)
print(" reach_success プラグインの検証")
print("=" * 78)

all_ok = True


# ------------------------------------------------------------------ ①判定ロジック（合成データ）
def check_threshold_logic():
    """presence の値から「触れた」を正しく判定できるか。答えの分かった値で確認する。"""
    print("\n[1] しきい値の判定ロジック（合成データ）")
    ok = True

    class FakeTouchModule:
        """part_features を差し替えて、presence の値を自分で決められるようにする。"""
        group_names = ["head", "chest", "left_palm", "other"]

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

    class FakeTaro:
        def __init__(self, groups, touch):
            self.reach_touch_groups = groups
            self.target_fusion = FakeFusion(touch)

    tm = FakeTouchModule()
    fake_taro = FakeTaro(["head", "chest", "left_palm"], tm)
    plugin = ReachSuccess({"threshold": 0.5})

    class FakeCtx:
        brain = object()   # None でなければよい
        dt = 0.05
        taro = fake_taro
        last = None

    ctx = FakeCtx()
    plugin.setup(ctx)

    # 1歩目：どこにも触れていない（presence=0）→ 0件のまま
    tm.set_presence(head=0.0, chest=0.0, left_palm=0.0)
    ctx.last = {"obs_out": {"touch": np.zeros(12, dtype=np.float32)}}
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    step1_zero = all(m[f"reach_touch_{nm}"] == 0 for nm in fake_taro.reach_touch_groups)
    print(f"  presence=0 のとき touch回数が0のままか  {'はい' if step1_zero else 'いいえ'}")
    ok &= step1_zero

    # 2歩目：head だけしきい値のすぐ下（0.49）→ 触れた扱いにならない
    tm.set_presence(head=0.49, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    below_thresh_not_counted = (m["reach_touch_head"] == 0)
    print(f"  presence=0.49（しきい値0.5未満）で触れた扱いにならないか  "
          f"{'はい' if below_thresh_not_counted else 'いいえ'}")
    ok &= below_thresh_not_counted
    # raw presence の値そのものはCSVに残る（しきい値を後で振り直せる証拠）
    raw_recorded = abs(m["reach_presence_max_head"] - 0.49) < 1e-6
    print(f"  presence=0.49 の生の値がCSV列に残るか（0.49と一致）  "
          f"{'はい' if raw_recorded else 'いいえ'}（実測 {m['reach_presence_max_head']}）")
    ok &= raw_recorded

    # 3歩目：head がしきい値のすぐ上（0.51）→ 触れた扱いになる
    tm.set_presence(head=0.51, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    above_thresh_counted = (m["reach_touch_head"] == 1)
    print(f"  presence=0.51（しきい値0.5超）で触れた扱いになるか  "
          f"{'はい' if above_thresh_counted else 'いいえ'}")
    ok &= above_thresh_counted

    # 4・5歩目：chest, left_palm にも触れる → 3部位すべて触れた扱い(all3)になる
    tm.set_presence(head=0.9, chest=0.9, left_palm=0.0)
    plugin.on_step(ctx)
    tm.set_presence(head=0.9, chest=0.9, left_palm=0.9)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    all3_recorded = ("reach_all3_step" in m) and (m["reach_all3_step"] == 5)
    print(f"  3部位すべてに触れた回（5歩目）で reach_all3_step=5 と記録されるか  "
          f"{'はい' if all3_recorded else 'いいえ'}（実測 {m.get('reach_all3_step')}）")
    ok &= all3_recorded

    # 6歩目：もう一度触れなくても、達成済みの記録（5）は変わらない
    tm.set_presence(head=0.0, chest=0.0, left_palm=0.0)
    plugin.on_step(ctx)
    m = plugin.metrics(ctx)
    stays = ("reach_all3_step" in m) and (m["reach_all3_step"] == 5)
    print(f"  達成後は値が5のまま変わらないか  {'はい' if stays else 'いいえ'}"
          f"（実測 {m.get('reach_all3_step')}）")
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

    p = ReachSuccess({})
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
        p2 = ReachSuccess({})
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
    接触回数がほぼ0であることを確認する。

    注意：ここは「学習を回さない」検証。tr.build() は環境と太郎(脳)を組み立てる
      だけで、重みは一切更新しない。env.step も零行動で進めるだけ。
    """
    print("\n[3] 学習を回さず、初期姿勢付近で接触がほぼ0か（実機）")
    cfg = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False},
        "run": {"steps": 40, "seed": 0, "checkpoint": 40, "K": 10},
    })
    plugin = ReachSuccess({"threshold": DEFAULT_THRESHOLD})
    tr = Trainer(cfg, plugins=[plugin], verbose=False, log_row=lambda row: None)
    ok = True
    try:
        tr.build()     # 環境＋太郎（脳）を作るだけ。学習(tr.run())は呼ばない
        env = tr.env
        u = env.unwrapped
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        if float(env.action_space.low[0]) >= 0.0:
            zero[:] = 0.0     # 筋肉モード=[0,1]。0=脱力
        obs = tr.state["obs"]
        n_ticks = 40
        for i in range(n_ticks):
            for _ in range(cfg.K):
                obs, _r, term, trunc, _info = env.step(zero)
                if term or trunc:
                    obs, _ = env.reset()
            # trainer.py の on_step と同じ形の last を手作りして渡す（測る側は読むだけ）
            tr.ctx.step = i + 1
            tr.ctx.last = {"obs_in": None, "obs_out": obs}
            plugin.on_step(tr.ctx)
        rep = plugin.report(tr.ctx)
        print(f"  {n_ticks}tick、零行動（力を入れない）で進めた結果:")
        for k, v in rep.items():
            print(f"    {k}: {v}")
        total = rep["触れた回数_通し"]
        # 【想定外・2026-08-02】chest は reset 直後（0tick時点）から presence≈1.0 で
        #   飽和している（zero行動ループを回す前から、である＝重力で崩れた結果ではない）。
        #   原因は body_support（このシーンはリクライニング60度で、体幹を座面へ固定する
        #   ためのサポートが常に胸へ接触している）。SomatosensoryCortex.part_features は
        #   接触力の**発生源**（自分の手か、外部の座面か）を区別しないため、この
        #   presence値は「自己接触」ではなく「座面との接触」を拾っている。
        #   encode_reach_goal も同じ part_features を使っているので、この限界は
        #   このプラグイン固有ではなく、reach_self の目標表現そのものに共通する。
        #   → head・left_palm（実際に手を伸ばして触れる必要がある2部位）が0のままかで
        #     判定する。chestは「座面接触との混同」という既知の限界として別扱いにする。
        reach_dependent = {k: v for k, v in total.items() if k != "chest"}
        near_zero = all(v == 0 for v in reach_dependent.values())
        chest_saturated = total.get("chest", 0) > 0
        print(f"  head・left_palm（要リーチの部位）が接触0件のままか  "
              f"{'はい' if near_zero else 'いいえ（要確認）'}")
        if chest_saturated:
            print("  注意：chest は reset直後から接触あり（座面のbody_supportとの接触。"
                  "自己接触ではない。part_features は接触の発生源を区別しない設計上の限界。"
                  "encode_reach_goalのq_touchにも同じ限界がある＝想定外として報告）")
        ok &= near_zero
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
    """体を作り直す（成長）ときに reach_success が例外を出さずに追従するか。

    注意：ここも学習(tr.run())は呼ばない。_regrow() を直接1回呼んで
      「体だけ作り直す」動作を単体で確認する（実際の学習ループと同じ経路）。
    """
    print("\n[4] 体を作り直しても（成長）落ちないか")
    cfg = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False, "age_months": 0.0},
        "run": {"steps": 40, "seed": 0, "checkpoint": 40, "K": 10},
    })
    plugin = ReachSuccess({})
    tr = Trainer(cfg, plugins=[plugin], verbose=False, log_row=lambda row: None)
    ok = True
    try:
        tr.build()
        before_groups = list(plugin.groups)
        tr._regrow(4.0)      # 0ヶ月 → 4ヶ月へ体を作り直す（学習はしない）
        after_groups = list(plugin.groups)
        # on_body_change は trainer._regrow が全プラグインへ自動で呼ぶ
        # （run/trainer.py 293-294行）。ここで例外が出ずに来られていること自体が確認。
        print(f"  作り直し前後で対象部位は変わらないか  "
              f"{'はい' if before_groups == after_groups else 'いいえ'}"
              f"（前 {before_groups} / 後 {after_groups}）")
        ok &= (before_groups == after_groups)
        # 作り直した体で1tick分、実際に測れるか（例外が出ないか）
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
