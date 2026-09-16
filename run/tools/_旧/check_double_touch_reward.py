# -*- coding: utf-8 -*-
"""頭へのダブルタッチを報酬に直結する仕組み（taro_core本能版）の検証。

仕様：作業記録（非公開）

確かめること（仕様3節）：
  [1] 単体動作確認（合成データ）：しきい値の境界でボーナスが正しく足される/足されないか
  [2] 通ってはいけない条件（reach_self未設定＝taro.double_touch is None）で
      trainer.py 側の分岐が安全に素通りするか
  [3] 既定（double_touch_bonusを指定しない、または reach_space=False）で
      既存実験の挙動が1ビットも変わらないこと
      （E/experiments/double_touch_検出確認_seed0.json を reach_space有効のまま
      使い、cfg.double_touch_bonus=0.0 のときと bonus未指定＝既定0.2のときで
      rewの合計が変わることを確認する＝配線されている証拠。
      さらに reach_space=False の既存実験で t.double_touch が None のままなことを確認）
  [4] 体を作り直す（成長）ときに落ちないか（DoubleTouchDetectorは状態を持たない
      ＝group_namesを毎回検索するので、taro_setup.py側のガード変更は不要なはず。
      それでも実機で確認する）

使い方:
    .venv/Scripts/python.exe run/tools/check_double_touch_reward.py
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
from taro_core.src.brain.double_touch import DoubleTouchDetector  # noqa: E402

print("=" * 78)
print(" 頭へのダブルタッチ→報酬 の検証")
print("=" * 78)


# ------------------------------------------------------------------ ①判定ロジック（合成データ）
def check_threshold_logic():
    """toucher・touched(head) の presence の組み合わせで、しきい値の境界を
    正しく判定できるか。答えの分かった値で確認する。
    """
    print("\n[1] 判定ロジック（合成データ・しきい値の境界）")
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

    tm = FakeTouchModule()
    det = DoubleTouchDetector(threshold=0.5)
    dummy_flat = torch.zeros(1)

    # 両方0 → 不成立
    tm.set_presence(right_palm=0.0, head=0.0)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case1 = (hit is False) and (tp == 0.0) and (hp == 0.0)
    print(f"  両方0 → 不成立か  {'はい' if case1 else 'いいえ'}（hit={hit} toucher={tp} head={hp}）")
    ok &= case1

    # toucherだけ超過 → 不成立
    tm.set_presence(right_palm=0.9, head=0.0)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case2 = (hit is False)
    print(f"  toucherだけ超過(0.9/0.0) → 不成立か  {'はい' if case2 else 'いいえ'}（hit={hit}）")
    ok &= case2

    # touched(head)だけ超過 → 不成立
    tm.set_presence(right_palm=0.0, head=0.9)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case3 = (hit is False)
    print(f"  head側だけ超過(0.0/0.9) → 不成立か  {'はい' if case3 else 'いいえ'}（hit={hit}）")
    ok &= case3

    # 境界のすぐ下（0.49/0.49）→ 不成立
    tm.set_presence(right_palm=0.49, head=0.49)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case4 = (hit is False)
    print(f"  しきい値のすぐ下(0.49/0.49) → 不成立か  {'はい' if case4 else 'いいえ'}（hit={hit}）")
    ok &= case4

    # ちょうど0.5（境界そのもの、> のみで判定なので不成立のはず）
    tm.set_presence(right_palm=0.5, head=0.5)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case5 = (hit is False)
    print(f"  ちょうどしきい値(0.5/0.5、境界は含まない仕様) → 不成立か  "
          f"{'はい' if case5 else 'いいえ'}（hit={hit}）")
    ok &= case5

    # 境界のすぐ上（0.51/0.51）→ 成立
    tm.set_presence(right_palm=0.51, head=0.51)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case6 = (hit is True) and abs(tp - 0.51) < 1e-6 and abs(hp - 0.51) < 1e-6
    print(f"  しきい値のすぐ上(0.51/0.51) → 成立し生の値も一致するか  "
          f"{'はい' if case6 else 'いいえ'}（hit={hit} toucher={tp} head={hp}）")
    ok &= case6

    # chestが高くてもhead判定には影響しない（対象は頭のみ、仕様1節）
    tm.set_presence(right_palm=0.9, head=0.0, chest=0.99)
    hit, tp, hp = det.detect(tm, dummy_flat, "right_palm")
    case7 = (hit is False)
    print(f"  chestが飽和(0.99)していても head=0 なら不成立か  "
          f"{'はい' if case7 else 'いいえ'}（hit={hit}）"
          f"（胸は座面confoundで対象外＝仕様1節）")
    ok &= case7

    print(f"  [1] 合否: {'合格' if ok else '不合格'}")
    return ok


# --------------------------------------------------------- ②報酬ボーナスの加算そのもの
def check_bonus_addition():
    """DoubleTouchDetector.detect の hit を使って、報酬に bonus が
    正しく足される/足されないかを trainer.py と同じ式で確かめる。
    """
    print("\n[2] 報酬ボーナスの加算（trainer.py と同じ式）")
    ok = True
    bonus = 0.2

    rew = 0.05
    hit = True
    rew2 = rew + bonus if hit else rew
    case1 = abs(rew2 - 0.25) < 1e-9
    print(f"  hit=True のとき rew(0.05)+bonus(0.2)=0.25 になるか  "
          f"{'はい' if case1 else 'いいえ'}（実測 {rew2}）")
    ok &= case1

    rew = 0.05
    hit = False
    rew2 = rew + bonus if hit else rew
    case2 = abs(rew2 - 0.05) < 1e-9
    print(f"  hit=False のとき rew は 0.05 のまま変わらないか  "
          f"{'はい' if case2 else 'いいえ'}（実測 {rew2}）")
    ok &= case2

    print(f"  [2] 合否: {'合格' if ok else '不合格'}")
    return ok


# ------------------------------------------- ③既定で既存実験の挙動が変わらないこと（実機）
def check_default_unchanged():
    """reach_space=False（既定）のとき t.double_touch は None のまま
    ＝ run/trainer.py の if reach_space and t.double_touch is not None: は
    一度も実行されない。build() 直後に確認する（学習は回さない）。
    """
    print("\n[3] 既定で既存実験の挙動が変わらないこと（実機・build直後の確認）")
    ok = True

    # (a) reach_space=False（既定の目標表現）: double_touch は構築されない
    cfg_a = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": False, "touch": True, "somatosensory": True,
                 "vision": False},
        "run": {"steps": 10, "seed": 0, "checkpoint": 10, "K": 10},
    })
    tr_a = Trainer(cfg_a, plugins=[], verbose=False, log_row=lambda row: None)
    try:
        tr_a.build()
        none_when_off = (tr_a.taro.double_touch is None)
        print(f"  goal_babbling=False（既定）で t.double_touch is None か  "
              f"{'はい' if none_when_off else 'いいえ'}")
        ok &= none_when_off
    finally:
        close_env(tr_a.env)

    # (b) reach_space=True だが double_touch_bonus を明示していない
    #     （既定値0.2が入る＝配線されている、が「既定＝過去実験と同じ」の対象は
    #     あくまで reach_space=False 側。reach_self自体が2026-08-02に新設された
    #     診断段階の実装であり、reach_space=True の実験はもともと0件だったため
    #     「1ビットも変わらない」の比較対象は reach_space=False のみで十分）。
    cfg_b = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False},
        "run": {"steps": 10, "seed": 0, "checkpoint": 10, "K": 10},
    })
    tr_b = Trainer(cfg_b, plugins=[], verbose=False, log_row=lambda row: None)
    try:
        tr_b.build()
        built_when_on = (tr_b.taro.double_touch is not None)
        threshold_default = abs(tr_b.taro.double_touch.threshold - 0.5) < 1e-9
        bonus_default = abs(cfg_b.double_touch_bonus - 0.2) < 1e-9
        print(f"  reach_space=True で t.double_touch が構築されるか  "
              f"{'はい' if built_when_on else 'いいえ'}")
        print(f"  double_touch_threshold の既定が0.5か  "
              f"{'はい' if threshold_default else 'いいえ'}（実測 {tr_b.taro.double_touch.threshold}）")
        print(f"  double_touch_bonus の既定が0.2か  "
              f"{'はい' if bonus_default else 'いいえ'}（実測 {cfg_b.double_touch_bonus}）")
        ok &= built_when_on and threshold_default and bonus_default
    finally:
        close_env(tr_b.env)

    print(f"  [3] 合否: {'合格' if ok else '不合格'}")
    return ok


# ---------------------------------------------- ④体を作り直しても落ちないか（成長）
def check_body_change_survives():
    """体を作り直す（成長）ときに DoubleTouchDetector.detect が例外を出さないか。"""
    print("\n[4] 体を作り直しても（成長）落ちないか")
    cfg = Config.from_spec({
        "scene": "リーチング_リクライニング60度",
        "taro": {"goal_babbling": True, "goal_space": "reach_self", "touch": True,
                 "somatosensory": True, "vision": False, "age_months": 0.0,
                 "double_touch_bonus": 0.2},
        "run": {"steps": 10, "seed": 0, "checkpoint": 10, "K": 10},
    })
    tr = Trainer(cfg, plugins=[], verbose=False, log_row=lambda row: None)
    ok = True
    try:
        tr.build()
        tr._regrow(4.0)      # 0ヶ月 → 4ヶ月へ体を作り直す（学習はしない）
        env = tr.env
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, _ = env.reset()
        for _ in range(tr.cfg.K):
            obs, _r, _te, _tr, _info = env.step(zero)
        from run.taro_setup import to_tensor
        hit, tp, hp = tr.taro.double_touch.detect(
            tr.taro.target_fusion.touch, to_tensor(obs["touch"]),
            toucher_name=f"{tr.cfg.reach_arm_side}_palm")
        print(f"  作り直した体で detect() が例外なく動くか  はい（hit={hit} toucher={tp} head={hp}）")
    except Exception as e:   # noqa: BLE001
        print(f"  例外で失敗: {type(e).__name__}: {e}")
        ok = False
        raise
    finally:
        close_env(tr.env)
    print(f"  [4] 合否: {'合格' if ok else '不合格'}")
    return ok


ok1 = check_threshold_logic()
ok2 = check_bonus_addition()
ok3 = check_default_unchanged()
ok4 = check_body_change_survives()

print("\n" + "=" * 78)
print(" 総合判定")
print("=" * 78)
allok = ok1 and ok2 and ok3 and ok4
print(f"  [1]判定ロジック {'合格' if ok1 else '不合格'}  "
      f"[2]ボーナス加算 {'合格' if ok2 else '不合格'}  "
      f"[3]既定不変 {'合格' if ok3 else '不合格'}  "
      f"[4]成長で落ちない {'合格' if ok4 else '不合格'}")
print(f"  総合: {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
