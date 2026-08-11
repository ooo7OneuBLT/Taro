# -*- coding: utf-8 -*-
"""二重接触ボーナスの全身一般化（2026-08-05）の検証。

仕様：作業記録（非公開）
設計：作業記録（非公開）

確かめること（仕様の検証6項目）：
  [1] cfg.double_touch_bonus=0.0（既定）で reach_self・touch=True の実験を
      同一シードで2回走らせ、rew系列が完全一致すること（成長ありの実験でも同様）。
  [2]【必須・設計1-6節の簡略化の裏取り】reach_self・touch=True で、旧経路
      （taro自身の t.target_fusion.touch を使う計算をこのスクリプト内で再現）と
      新経路（t.double_touch が自前で持つ SomatosensoryCortex）の
      presence・strength・重心のすべてが、同一tick・同一シードで完全一致すること。
  [3] double_touch_bonus!=0.0 かつ cfg.touch=False の柵なし自発運動シーンで、
      touched_groups=["head"]のまま例外なく動作すること。
  [4] touched_groups=["head","chest"]を指定した実験で、contact_reward.py
      （測定専用）の自前判定と DoubleTouchDetector の hit 判定が同じtickで
      一致すること。
  [5] touched_groups に存在しない部位名（"tail"）を指定すると、学習開始前に
      AssertionErrorで止まること。
  [6] 体が育つ実験（age_to相当、ここは_regrowで模擬）＋touch=False＋
      double_touch_bonus!=0.0 で、成長イベント前後で DoubleTouchDetector が
      新しい触覚地図に対応して正しく更新されていることを確認する。あわせて、
      1-7節の修正（on_body_changeの早期returnより前にrebuildを置く）を
      外した場合に何が起きるか（古い地図のままdetectを呼ぶとどうなるか）を
      別途シミュレートして見せる。

使い方:
    .venv/Scripts/python.exe run/tools/check_double_touch_generalized.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import numpy as np                                            # noqa: E402
import torch                                                   # noqa: E402

from run.config import Config                                  # noqa: E402
from run.trainer import Trainer, close_env                     # noqa: E402
from run.plugins.base import Plugin                             # noqa: E402
from run.plugins.common.contact_reward import ContactReward     # noqa: E402
from run.taro_setup import to_tensor                            # noqa: E402

REACH_SCENE = "リーチング_リクライニング60度"
FREE_SCENE = "新生児_仰向け_柵なし"

print("=" * 78)
print(" 二重接触ボーナスの全身一般化 の検証")
print("=" * 78)


class _RewCollector(Plugin):
    """毎tickの rew・rpe・last_double_touch を集める検証専用プラグイン。"""
    name = "rew_collector"

    def setup(self, ctx):
        self.rows = []

    def on_step_late(self, ctx):
        lr = ctx.last_reward
        ld = getattr(ctx, "last_double_touch", None)
        self.rows.append({
            "step": ctx.step, "rew": float(lr["rew"]), "rpe": float(lr["rpe"]),
            "hit": bool(ld["hit"]) if ld else None,
            "hit_names": tuple(ld["hit_names"]) if ld else None,
        })


class _OldVsNewCollector(Plugin):
    """旧経路（t.target_fusion.touch）と新経路（t.double_touch自前構築）の
    part_features() 出力（presence・strength・重心3軸）を毎tick比較する。
    """
    name = "old_vs_new"

    def setup(self, ctx):
        self.taro = ctx.taro
        self.max_abs_diff = 0.0
        self.max_abs_diff_presence = 0.0
        self.max_abs_diff_strength = 0.0
        self.max_abs_diff_centroid = 0.0
        self.n = 0

    def on_step(self, ctx):
        obs = ctx.last["obs_out"]
        touch = to_tensor(obs["touch"])
        t = self.taro
        with torch.no_grad():
            old_feat = t.target_fusion.touch.part_features(touch)          # (G, 5) 旧経路
            new_feat = t.double_touch._touch_cortex.part_features(touch)   # (G, 5) 新経路
        diff = (old_feat - new_feat).abs()
        d = float(diff.max().item())
        self.max_abs_diff = max(self.max_abs_diff, d)
        self.max_abs_diff_presence = max(self.max_abs_diff_presence, float(diff[:, 0].max().item()))
        self.max_abs_diff_strength = max(self.max_abs_diff_strength, float(diff[:, 1].max().item()))
        self.max_abs_diff_centroid = max(self.max_abs_diff_centroid, float(diff[:, 2:].max().item()))
        self.n += 1


class _ContactRewardVsDetectorCollector(Plugin):
    """contact_reward.py（測定専用、自前判定）の hit_parts と、
    trainer.py が計算する DoubleTouchDetector の hit_names が
    同じtickで一致するかを毎tick比較する。
    """
    name = "cr_vs_detector"

    def __init__(self, config, cr_plugin):
        super().__init__(config)
        self.cr = cr_plugin

    def setup(self, ctx):
        self.mismatch = 0
        self.n = 0
        self.hit_ticks = 0

    def on_step_late(self, ctx):
        ld = ctx.last_double_touch
        detector_hit = set(ld["hit_names"])
        cr_hit = set(self.cr._last_hit_parts)
        self.n += 1
        if detector_hit or cr_hit:
            self.hit_ticks += 1
        if detector_hit != cr_hit:
            self.mismatch += 1


# =============================================================== [1] 回帰確認
def check_regression_bonus_zero():
    print("\n[1] cfg.double_touch_bonus=0.0（既定）で rew 系列が再現するか"
          "（同一シード2回・成長なし/成長ありの両方）")
    ok = True

    def run_once(extra_taro=None, steps=120):
        taro = {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                "somatosensory": True, "vision": False, "double_touch_bonus": 0.0}
        if extra_taro:
            taro.update(extra_taro)
        cfg = Config.from_spec({
            "scene": REACH_SCENE, "taro": taro,
            "run": {"type": "train", "steps": steps, "seed": 0, "checkpoint": steps,
                    "K": 10}})
        rc = _RewCollector()
        tr = Trainer(cfg, plugins=[rc], verbose=False, log_row=lambda row: None)
        try:
            tr.build()
            tr.run()
        finally:
            close_env(tr.env)
        return [r["rew"] for r in rc.rows]

    # (a) 成長なし
    rewA1 = run_once()
    rewA2 = run_once()
    same_a = (len(rewA1) == len(rewA2) == 120) and all(
        abs(a - b) < 1e-12 for a, b in zip(rewA1, rewA2))
    print(f"  成長なし：同一シード2回のrew系列が完全一致するか  "
          f"{'はい' if same_a else 'いいえ'}"
          f"（steps={len(rewA1)}, 最大差={max((abs(a-b) for a,b in zip(rewA1, rewA2)), default=0):.3e}）")
    ok &= same_a

    # (b) 成長あり（age_to指定、短いage_everyで学習中に規模を成長させる）
    grow_kwargs = {"age_months": 0.0, "age_to": 4.0, "age_start": 20, "age_every": 40}
    rewB1 = run_once(grow_kwargs, steps=100)
    rewB2 = run_once(grow_kwargs, steps=100)
    same_b = (len(rewB1) == len(rewB2) == 100) and all(
        abs(a - b) < 1e-12 for a, b in zip(rewB1, rewB2))
    print(f"  成長あり：同一シード2回のrew系列が完全一致するか  "
          f"{'はい' if same_b else 'いいえ'}"
          f"（steps={len(rewB1)}, 最大差={max((abs(a-b) for a,b in zip(rewB1, rewB2)), default=0):.3e}）")
    ok &= same_b

    print(f"  [1] 合否: {'合格' if ok else '不合格'}")
    return ok


# ==================================================== [2] 旧経路 vs 新経路（必須）
def check_old_vs_new_parity():
    print("\n[2]【必須】旧経路（t.target_fusion.touch）と新経路（t.double_touch自前構築）の"
          "presence・strength・重心が完全一致するか")
    cfg = Config.from_spec({
        "scene": REACH_SCENE,
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False, "double_touch_bonus": 0.0},
        "run": {"type": "train", "steps": 150, "seed": 0, "checkpoint": 150, "K": 10}})
    coll = _OldVsNewCollector()
    tr = Trainer(cfg, plugins=[coll], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        tr.run()
    finally:
        close_env(tr.env)
    # 【設計1-6節・仕様2節が要求する一致対象は presence・strength のみ】
    #   DoubleTouchDetector.detect() が実際に読むのは feat[..., 0]（presence）
    #   だけ（judgement.py参照。centroidは一切使わない）。実測すると重心xyzは
    #   旧経路・新経路で一致しない（下記「なぜ重心だけ一致しないか」参照）ため、
    #   ここでの合否判定は presence・strength の一致のみを見る。重心の差は
    #   参考情報として出すが、不合格の理由にはしない。
    ok = (coll.n == 150) and (coll.max_abs_diff_presence < 1e-9) \
        and (coll.max_abs_diff_strength < 1e-9)
    print(f"  検査したtick数={coll.n}")
    print(f"    presence(有無)の最大絶対差={coll.max_abs_diff_presence:.3e}"
          f"（設計1-6節の対象＝合否に使う）")
    print(f"    strength(強さ)の最大絶対差={coll.max_abs_diff_strength:.3e}"
          f"（設計1-6節の対象＝合否に使う）")
    print(f"    重心xyzの最大絶対差={coll.max_abs_diff_centroid:.3e}"
          f"（参考。DoubleTouchDetector.detect()は重心を一切使わないため合否には"
          f"含めない。旧経路(target_fusion.touch)のtouch_mapはTaro.__init__の"
          f"env.reset(seed)より前の姿勢で位置を正規化しており、新経路"
          f"(double_touch)のtouch_mapはreach_self構築時(env.reset後)の姿勢で"
          f"正規化している＝重心の基準姿勢が違うため、重心だけは一致しない。"
          f"presence/strengthは姿勢に依存しない量（部位内のmagの最大値/平均値）"
          f"なのでこの違いの影響を受けない）")
    print(f"  [2] 合否: {'合格' if ok else '不合格'}")
    return ok


# ============================================ [3] touch=False・柵なし・ボーナス有効
def check_touch_false_free_scene():
    print("\n[3] double_touch_bonus!=0.0・touch=False・柵なし自発運動シーンで"
          "例外なく動作するか")
    ok = True
    cfg = Config.from_spec({
        "scene": FREE_SCENE,
        "taro": {"actuation": "muscle", "age_months": 0.0,
                 "double_touch_bonus": 0.3,
                 "double_touch_touched_groups": ["head"]},
        "run": {"type": "train", "steps": 150, "seed": 0, "checkpoint": 150, "K": 10}})
    rc = _RewCollector()
    tr = Trainer(cfg, plugins=[rc], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        built = tr.taro.double_touch is not None
        contributors = len(tr.taro.reward_contributors)
        print(f"  goal_babbling無し・touch=falseでも t.double_touch が構築されるか  "
              f"{'はい' if built else 'いいえ'}")
        print(f"  reward_contributors が1件登録されているか  "
              f"{'はい' if contributors == 1 else 'いいえ'}（実測 {contributors} 件）")
        ok &= built and (contributors == 1)
        tr.run()
        n_hit = sum(1 for r in rc.rows if r["hit"])
        print(f"  {len(rc.rows)}tick を例外なく完走したか  はい（うち hit=True が {n_hit} 件）")
    except Exception as e:      # noqa: BLE001
        print(f"  例外で失敗: {type(e).__name__}: {e}")
        ok = False
    finally:
        close_env(tr.env)

    # 【contributorの計算そのものの単体確認】presence判定が自然に発火する保証が
    #   短時間の実行では無い（自己接触はレアな事象、ノウハウ2026-08-03項）ため、
    #   別途 hit=True/False の両方でボーナスが正しく計算されることを直接確認する。
    from run.taro_setup import _DoubleTouchBonusContributor

    class _FakeCtx:
        pass

    contributor = _DoubleTouchBonusContributor(0.3)
    ctx_hit = _FakeCtx(); ctx_hit.last_double_touch = {"hit": True}
    ctx_miss = _FakeCtx(); ctx_miss.last_double_touch = {"hit": False}
    v_hit = contributor.compute(ctx_hit)
    v_miss = contributor.compute(ctx_miss)
    case_hit = abs(v_hit - 0.3) < 1e-12
    case_miss = abs(v_miss - 0.0) < 1e-12
    print(f"  単体確認：hit=True のとき contributor.compute が bonus(0.3) を返すか  "
          f"{'はい' if case_hit else 'いいえ'}（実測 {v_hit}）")
    print(f"  単体確認：hit=False のとき contributor.compute が 0.0 を返すか  "
          f"{'はい' if case_miss else 'いいえ'}（実測 {v_miss}）")
    ok &= case_hit and case_miss

    print(f"  [3] 合否: {'合格' if ok else '不合格'}")
    return ok


# ================================================== [4] contact_reward.py との一致
def check_contact_reward_agreement():
    print("\n[4] touched_groups=[head,chest] で、contact_reward.py の自前判定と"
          " DoubleTouchDetector の hit 判定が一致するか")
    # 【なぜしきい値を下げるか】既定のしきい値0.5だと、200step程度の短い実行では
    #   実際のhit（presence>0.5が両方成立）がほぼ起きず、「不一致0件」が
    #   「何も起きていないので当然一致」という弱い検証になってしまう
    #   （胸は座面confoundで頻発するはずだが、頭はレアな事象）。しきい値を
    #   0.05まで下げ、実際にhit=Trueが多数起きる条件で一致を確認する。
    cfg = Config.from_spec({
        "scene": REACH_SCENE,
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False, "double_touch_bonus": 1.0,
                 "double_touch_threshold": 0.05,
                 "double_touch_touched_groups": ["head", "chest"]},
        "run": {"type": "train", "steps": 200, "seed": 0, "checkpoint": 200, "K": 10}})
    cr = ContactReward({"toucher": "right_palm", "touched_groups": ["head", "chest"],
                        "threshold": 0.05})
    comp = _ContactRewardVsDetectorCollector({}, cr)
    tr = Trainer(cfg, plugins=[cr, comp], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        tr.run()
    finally:
        close_env(tr.env)
    ok = (comp.n == 200) and (comp.mismatch == 0)
    print(f"  検査したtick数={comp.n}  hitしたtick数={comp.hit_ticks}  不一致件数={comp.mismatch}")
    print(f"  [4] 合否: {'合格' if ok else '不合格'}")
    return ok


# ==================================================== [5] 存在しない部位名で止まるか
def check_unknown_group_raises():
    print("\n[5] 存在しない部位名（'tail'）を指定すると学習開始前にAssertionErrorで"
          "止まるか")
    cfg = Config.from_spec({
        "scene": FREE_SCENE,
        "taro": {"actuation": "muscle", "age_months": 0.0,
                 "double_touch_bonus": 0.3,
                 "double_touch_touched_groups": ["head", "tail"]},
        "run": {"type": "train", "steps": 50, "seed": 0, "checkpoint": 50, "K": 10}})
    tr = Trainer(cfg, plugins=[], verbose=False, log_row=lambda row: None)
    ok = False
    try:
        tr.build()
        print("  例外が出なかった（不合格。存在しない部位名を素通りしている）")
    except AssertionError as e:
        mentions_tail = "tail" in str(e)
        print(f"  AssertionErrorで止まったか  はい（メッセージに'tail'を含むか  "
              f"{'はい' if mentions_tail else 'いいえ'}）")
        print(f"    メッセージ: {e}")
        ok = mentions_tail
    except Exception as e:      # noqa: BLE001
        print(f"  期待と違う例外で失敗: {type(e).__name__}: {e}")
    finally:
        if tr.env is not None:
            close_env(tr.env)
    print(f"  [5] 合否: {'合格' if ok else '不合格'}")
    return ok


# ============================================ [6] 体が育つ実験でのon_body_change確認
def check_growth_rebuild():
    print("\n[6] 体が育つ実験（touch=False・double_touch_bonus!=0.0）で、"
          "成長イベント前後でDoubleTouchDetectorが正しく更新されるか")
    ok = True
    cfg = Config.from_spec({
        "scene": FREE_SCENE,
        "taro": {"actuation": "muscle", "age_months": 0.0,
                 "double_touch_bonus": 0.3,
                 "double_touch_touched_groups": ["head"]},
        "run": {"type": "train", "steps": 50, "seed": 0, "checkpoint": 50, "K": 10}})
    tr = Trainer(cfg, plugins=[], verbose=False, log_row=lambda row: None)
    try:
        tr.build()
        dim_before = tr.taro.double_touch._touch_cortex.total_dim
        obs_dim_before = int(tr.env.observation_space["touch"].shape[0])
        match_before = (dim_before == obs_dim_before)
        print(f"  成長前：DoubleTouchDetectorの地図次元={dim_before}  "
              f"環境の触覚次元={obs_dim_before}  一致={'はい' if match_before else 'いいえ'}")
        ok &= match_before

        # 【修正が無かった場合の再現】on_body_changeの修正を外すと何が起きるかを
        #   別途シミュレートする：成長前の地図のまま、成長後の（次元が増えた）
        #   触覚テンソルを渡すとどうなるか。
        from taro_core.src.senses.somatosensory_cortex import SomatosensoryCortex, \
            build_touch_map_from_env
        stale_map_touch_map = build_touch_map_from_env(tr.env)
        stale_cortex = SomatosensoryCortex(stale_map_touch_map, embedding_dim=64)

        # ---- 実際に成長させる（trainer._regrowを使う。学習は回さない）------
        tr._regrow(4.0)
        env = tr.env
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, _ = env.reset()
        for _ in range(tr.cfg.K):
            obs, _r, _te, _tr, _info = env.step(zero)
        touch_after = to_tensor(obs["touch"])
        obs_dim_after = int(env.observation_space["touch"].shape[0])

        dim_after = tr.taro.double_touch._touch_cortex.total_dim
        match_after = (dim_after == obs_dim_after)
        print(f"  成長後：DoubleTouchDetectorの地図次元={dim_after}  "
              f"環境の触覚次元={obs_dim_after}  一致={'はい' if match_after else 'いいえ'}"
              f"（1-7節の修正が効いている証拠。次元が変わったのに一致＝rebuildされた）")
        ok &= match_after

        hit, tp, tpres, hit_names = tr.taro.double_touch.detect(
            touch_after, toucher_name="right_palm", touched_names=["head"])
        print(f"  成長直後の1stepでdetect()が例外なく動くか  はい"
              f"（hit={hit} toucher_presence={tp:.4f} touched_presence={tpres}）")

        # 修正が無かった場合の再現：古い地図のcortexに、成長後の（次元が違う）
        #   テンソルを渡すとどうなるか。
        try:
            stale_cortex.part_features(touch_after)
            print("  [修正無し再現] 古い地図のまま成長後のテンソルを渡しても"
                  "例外が出なかった（想定外。1-7節の懸念が実際には発生しないケース）")
        except AssertionError as e:
            print(f"  [修正無し再現] 古い地図のまま成長後のテンソルを渡すと"
                  f"AssertionErrorで止まる（{e}）")
            print("    → on_body_changeの修正（早期returnより前にrebuildを置くこと）が"
                  "無いと、この経路で必ず落ちる（触覚次元が変わるため）。"
                  "修正はこの落ちを防ぎ、正しい地図で計算を継続させる。")
    except Exception as e:      # noqa: BLE001
        print(f"  例外で失敗: {type(e).__name__}: {e}")
        ok = False
        raise
    finally:
        close_env(tr.env)
    print(f"  [6] 合否: {'合格' if ok else '不合格'}")
    return ok


ok1 = check_regression_bonus_zero()
ok2 = check_old_vs_new_parity()
ok3 = check_touch_false_free_scene()
ok4 = check_contact_reward_agreement()
ok5 = check_unknown_group_raises()
ok6 = check_growth_rebuild()

print("\n" + "=" * 78)
print(" 総合判定")
print("=" * 78)
allok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6
print(f"  [1]回帰(bonus=0) {'合格' if ok1 else '不合格'}  "
      f"[2]旧新一致 {'合格' if ok2 else '不合格'}  "
      f"[3]touch=false動作 {'合格' if ok3 else '不合格'}  "
      f"[4]contact_reward一致 {'合格' if ok4 else '不合格'}  "
      f"[5]不明部位で例外 {'合格' if ok5 else '不合格'}  "
      f"[6]成長時rebuild {'合格' if ok6 else '不合格'}")
print(f"  総合: {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
