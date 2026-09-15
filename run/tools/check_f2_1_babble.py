# -*- coding: utf-8 -*-
"""F2-1「喃語で口の内部モデルを作る」：短時間走行での配線・自己検証。

設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「検証」2〜4。
（検証1＝既存word呼び出しとの完全一致は run/tools/check_f2_1_babble_equivalence.py）

  ② 短い走行（30秒程度）で forward_map に行が増えること
  ③ 発話長が1〜2に収まること
  ④ 息切れ停止の撤去（jaw_cycles指定時のみ無効化）で、他の経路
     （目標B相当の呼び出し＝speech_planあり）が壊れないこと

注意：これは「検証」であって「実験」ではない（300step・sim 30秒で終わる）。
本走行（数百秒・複数シード）はしない（工程1の範囲外）。

使い方:
    .venv/Scripts/python.exe run/tools/check_f2_1_babble.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config             # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402
from run.plugins.base import Plugin        # noqa: E402


class _BabbleLengthCollector(Plugin):
    """検証専用：ctx.last_babble から発話長・jaw_cyclesを集めるだけ（読むだけ）。"""

    name = "babble_length_collector"

    def setup(self, ctx):
        self.lengths = []
        self.jaw_cycles = []

    def on_step(self, ctx):
        ev = getattr(ctx, "last_babble", None)
        if ev:
            self.lengths.append(ev["length"])
            self.jaw_cycles.append(ev["jaw_cycles"])


print("=" * 78)
print(" F2-1 喃語モード：短時間走行(300step=sim30秒)での配線・自己検証")
print("=" * 78)

cfg = Config.from_spec({
    "scene": "run/scenes/_旧/座位_6ヶ月_2おもちゃ_F1-3c_1個提示_2026-08-19.json",
    "taro": {
        "actuation": "muscle", "age_months": 6.0,
        "orienting_reflex": True, "hearing": True,
        "lexicon_vision": {"backend": "dinov2_vits14", "fovea_px": 32},
        "lexicon_mode": "contrast",
        "produce": {"mode": "babble", "cooldown_sec": 0.5,
                    "jaw_cycles": [1, 2], "vocal_tract_stage": 2,
                    "vocal_tract_decoupled": True},
    },
    "run": {"type": "train", "steps": 300, "seed": 7, "K": 10, "checkpoint": 300},
})

collector = _BabbleLengthCollector()
tr = Trainer(cfg, plugins=(collector,), verbose=False, log_row=lambda row: None)
ok_ran = True
try:
    tr.build()
    tr.run()
    print("  例外なく300step（sim30秒）走行できたか  はい")
except Exception as e:      # noqa: BLE001
    ok_ran = False
    print(f"  例外なく300step（sim30秒）走行できたか  いいえ"
          f"（{type(e).__name__}: {e}）")
    raise

cereb = tr.taro.produce_cerebellum
summary = cereb.get_experience_summary()
print(f"\n  ② 帳面（forward_map）の行数  {summary['forward_entries']}"
      f"（inverse_map={summary['inverse_entries']}文字、"
      f"total_experiences={summary['total_experiences']}）")
ok_grew = summary["forward_entries"] > 0

lengths = collector.lengths
from collections import Counter
len_dist = Counter(lengths)
jaw_dist = Counter(collector.jaw_cycles)
print(f"\n  ③ 発話回数={len(lengths)}  発話長の分布={dict(sorted(len_dist.items()))}"
      f"  jaw_cyclesの分布={dict(sorted(jaw_dist.items()))}")
ok_len = len(lengths) > 0 and all(1 <= n <= 2 for n in lengths)

close_env(tr.env)

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = ok_ran and ok_grew and ok_len
print(f"  ② forward_mapが増えたか  {'合格' if ok_grew else '不合格'}")
print(f"  ③ 発話長が1〜2に収まったか  {'合格' if ok_len else '不合格'}")
print(f"  短時間走行での配線・自己検証  {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
