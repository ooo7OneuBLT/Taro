# -*- coding: utf-8 -*-
"""喃語の測定器（babble_probe）の自己検証。

依頼書の検証1〜3に対応：
  検証1：既定OFFの原則 … babble_probe を足しても、既存の学習（発話含む）が
         乱数列を含め1ビットも変わらないこと
  検証2：短い喃語走行（300step）で①〜④のファイルが作られ、中身が妥当なこと
  検証3：同じseedで2回走らせて、記録が完全一致すること（測定器が乱数を
         消費していない証拠）

使い方:
    .venv/Scripts/python.exe run/tools/check_babble_probe.py
"""
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config              # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402
from run.plugins.common.babble_probe import BabbleProbe  # noqa: E402
from run.plugins.base import Plugin         # noqa: E402

SCRATCH = os.path.join(_R, "run", "tools", "_scratch_babble_probe")
os.makedirs(SCRATCH, exist_ok=True)

SPEC = {
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
}


class _Collector(Plugin):
    """検証専用：発声語の列とcerebellumの最終状態を比較用に集める（読むだけ）。"""
    name = "collector"

    def setup(self, ctx):
        self.words = []

    def on_step(self, ctx):
        ev = getattr(ctx, "last_babble", None)
        if ev:
            self.words.append(ev["generated_word"])


def run_once(with_probe, growth_out=None, events_out=None, practice_out=None,
             growth_every=100, seed=7):
    import copy
    spec = copy.deepcopy(SPEC)
    spec["run"]["seed"] = seed
    cfg = Config.from_spec(spec)
    collector = _Collector()
    plugins = [collector]
    probe = None
    if with_probe:
        probe = BabbleProbe({"events_out": events_out, "growth_out": growth_out,
                             "practice_out": practice_out, "growth_every": growth_every})
        plugins.append(probe)
    tr = Trainer(cfg, plugins=tuple(plugins), verbose=False, log_row=lambda row: None)
    tr.build()
    tr.run()
    cereb = tr.taro.produce_cerebellum
    snapshot = {
        "words": list(collector.words),
        "forward_map": dict(cereb.forward_map),
        "inverse_map": dict(cereb.inverse_map),
        "experience_count": dict(cereb.experience_count),
    }
    close_env(tr.env)
    return snapshot, probe


print("=" * 78)
print(" 検証1：babble_probe を足しても既存の学習が1ビットも変わらないか")
print("=" * 78)
snap_without, _ = run_once(with_probe=False)
snap_with, _ = run_once(with_probe=True)
ok1 = (snap_without["words"] == snap_with["words"]
       and snap_without["forward_map"] == snap_with["forward_map"]
       and snap_without["inverse_map"] == snap_with["inverse_map"]
       and snap_without["experience_count"] == snap_with["experience_count"])
print(f"  発声語の列が完全一致か        {snap_without['words'] == snap_with['words']}"
      f"（{len(snap_without['words'])}件）")
print(f"  forward_map が完全一致か      {snap_without['forward_map'] == snap_with['forward_map']}")
print(f"  experience_count が完全一致か {snap_without['experience_count'] == snap_with['experience_count']}")
print(f"  検証1  {'合格' if ok1 else '不合格'}")

print()
print("=" * 78)
print(" 検証2：短い喃語走行(300step)で①〜④のファイルが作られ、中身が妥当か")
print("=" * 78)
events_out = os.path.join(SCRATCH, "発話イベント.csv")
growth_out = os.path.join(SCRATCH, "帳面の成長.csv")
practice_out = os.path.join(SCRATCH, "音ごとの練習回数.csv")
snap2, probe2 = run_once(with_probe=True, growth_out=growth_out, events_out=events_out,
                          practice_out=practice_out, growth_every=100)

files_ok = os.path.exists(events_out) and os.path.exists(growth_out) and os.path.exists(practice_out)
print(f"  3ファイルとも作られたか  {files_ok}"
      f"（{events_out}, {growth_out}, {practice_out}）")

with open(events_out, encoding="utf-8") as fp:
    ev_rows = list(csv.DictReader(fp))
mismatch = [r for r in ev_rows if int(r["char_count"]) != int(r["jaw_cycles"])
            and int(r["char_count"]) > int(r["jaw_cycles"])]
# 文字数はjaw_cycles以下のはず（EOS等での早期終了はあっても超過はしない）
over = [r for r in ev_rows if int(r["char_count"]) > int(r["jaw_cycles"])]
exact_match = [r for r in ev_rows if int(r["char_count"]) == int(r["jaw_cycles"])]
print(f"  ①発話イベント行数={len(ev_rows)}"
      f"  文字数==jaw_cyclesの行={len(exact_match)}/{len(ev_rows)}"
      f"  文字数がjaw_cyclesを超えた行={len(over)}（0であるべき）")
ok_events = len(ev_rows) > 0 and len(over) == 0

with open(growth_out, encoding="utf-8") as fp:
    gr_rows = list(csv.DictReader(fp))
fwd_seq = [int(r["forward_entries"]) for r in gr_rows]
mono = all(fwd_seq[i] <= fwd_seq[i + 1] for i in range(len(fwd_seq) - 1))
cons_rates = [float(r["consonant_rate_cumulative"]) for r in gr_rows] + \
             [float(r["consonant_rate_window"]) for r in gr_rows]
cons_ok = all(0.0 <= c <= 1.0 for c in cons_rates)
print(f"  ②帳面の成長行数={len(gr_rows)}  forward_entries推移={fwd_seq}"
      f"  単調増加か={mono}")
print(f"  ④子音率が0〜1に収まるか={cons_ok}"
      f"  最終行 累積={gr_rows[-1]['consonant_rate_cumulative'] if gr_rows else None}"
      f" 区間={gr_rows[-1]['consonant_rate_window'] if gr_rows else None}")

with open(practice_out, encoding="utf-8") as fp:
    pr_rows = list(csv.DictReader(fp))
practice_sum = sum(int(r["practice_count"]) for r in pr_rows)
exp_total = sum(snap2["experience_count"].values())
print(f"  ③音ごとの練習回数：文字種={len(pr_rows)}"
      f"  合計={practice_sum}  cerebellum.experience_count総和={exp_total}"
      f"  一致={practice_sum == exp_total}")

ok2 = (files_ok and ok_events and mono and cons_ok and practice_sum == exp_total
       and len(gr_rows) > 0 and len(pr_rows) > 0)
print(f"  検証2  {'合格' if ok2 else '不合格'}")

print()
print("=" * 78)
print(" 検証3：同じseedで2回走らせて記録が完全一致するか（乱数を消費していない証拠）")
print("=" * 78)
events_out_a = os.path.join(SCRATCH, "発話イベントA.csv")
growth_out_a = os.path.join(SCRATCH, "帳面の成長A.csv")
practice_out_a = os.path.join(SCRATCH, "音ごとの練習回数A.csv")
events_out_b = os.path.join(SCRATCH, "発話イベントB.csv")
growth_out_b = os.path.join(SCRATCH, "帳面の成長B.csv")
practice_out_b = os.path.join(SCRATCH, "音ごとの練習回数B.csv")
snap_a, _ = run_once(with_probe=True, growth_out=growth_out_a, events_out=events_out_a,
                      practice_out=practice_out_a, growth_every=100, seed=11)
snap_b, _ = run_once(with_probe=True, growth_out=growth_out_b, events_out=events_out_b,
                      practice_out=practice_out_b, growth_every=100, seed=11)
with open(events_out_a, encoding="utf-8") as fp:
    a_ev = fp.read()
with open(events_out_b, encoding="utf-8") as fp:
    b_ev = fp.read()
with open(growth_out_a, encoding="utf-8") as fp:
    a_gr = fp.read()
with open(growth_out_b, encoding="utf-8") as fp:
    b_gr = fp.read()
with open(practice_out_a, encoding="utf-8") as fp:
    a_pr = fp.read()
with open(practice_out_b, encoding="utf-8") as fp:
    b_pr = fp.read()
ok3 = (a_ev == b_ev and a_gr == b_gr and a_pr == b_pr
       and snap_a["words"] == snap_b["words"])
print(f"  ①イベントCSVが完全一致  {a_ev == b_ev}")
print(f"  ②成長CSVが完全一致      {a_gr == b_gr}")
print(f"  ③練習回数CSVが完全一致  {a_pr == b_pr}")
print(f"  検証3  {'合格' if ok3 else '不合格'}")

print()
print("=" * 78)
print(" 判定")
print("=" * 78)
allok = ok1 and ok2 and ok3
print(f"  検証1（既定OFFの原則）        {'合格' if ok1 else '不合格'}")
print(f"  検証2（①〜④の妥当性）        {'合格' if ok2 else '不合格'}")
print(f"  検証3（決定性・乱数不消費）    {'合格' if ok3 else '不合格'}")
print(f"  総合  {'合格' if allok else '不合格'}")

import shutil
shutil.rmtree(SCRATCH, ignore_errors=True)

if not allok:
    sys.exit(1)
