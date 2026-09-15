# -*- coding: utf-8 -*-
"""F2-2「見た物の名前を言う」：短時間走行での配線・自己検証。

設計：F/docs/設計_F2-2_見た物の名前を言う.md 第4部「検証」2〜5。
（検証1＝既定OFFの原則は別途 run/tools/_scratch_f2_2 で確認済み。作業記録参照）

  2. 太郎が実際に「わんわん」と発話すること（文字列を報告）
  3. 発話長が計画どおり4モーラであること（stamina に切られていないこと）
  4. 「ぶーぶー」が「ぶあぶあ」になること（＝hear()フォールバックが切れている証拠）
  5. 帳面が読み込まれていること（forward_map 339行・inverse_map 68種）

15ヶ月シーン（座位_15ヶ月_2おもちゃ_F2-2_1個提示_fovea_2026-08-23）と
F2-1の帳面つきモデル（F/models/F2-1_喃語_6to12ヶ月_300s_seed10_2026-08-23.pt）を使う。

使い方:
    .venv/Scripts/python.exe run/tools/check_f2_2_short_run.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config              # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402
from run.plugins.common.word_production import WordProduction  # noqa: E402

print("=" * 78)
print(" F2-2 wordモード：短時間走行での配線・自己検証")
print("=" * 78)

cfg = Config.from_spec({
    "scene": "run/scenes/_旧/座位_15ヶ月_2おもちゃ_F2-2_1個提示_fovea_2026-08-23.json",
    "taro": {
        "actuation": "muscle", "age_months": 15.0,
        "orienting_reflex": True, "hearing": True,
        "lexicon_vision": {"backend": "dinov2_vits14", "fovea_px": 32},
        "lexicon_mode": "contrast",
        "lexicon_eta_pull": 0.05, "lexicon_eta_push": 0.025,
        "produce": {
            "threshold": 0.80, "cooldown_sec": 1.0, "max_length": 8,
            "develop_from_age": True, "vocal_tract_decoupled": True,
        },
        "model": "F/models/F2-1_喃語_6to12ヶ月_300s_seed10_2026-08-23.pt",
    },
    "run": {"type": "train", "steps": 3000, "seed": 30, "K": 10, "checkpoint": 3000},
})

wp = WordProduction({"events_out": None})
tr = Trainer(cfg, plugins=(wp,), verbose=False, log_row=lambda row: None)
tr.build()

# 検証5：帳面が読み込まれていること
cereb = tr.taro.produce_cerebellum
print(f"\n  検証5：帳面（forward_map={len(cereb.forward_map)}行、"
      f"inverse_map={len(cereb.inverse_map)}種）")
ok5 = len(cereb.forward_map) > 0 and len(cereb.inverse_map) > 0

# 【実装作業⑦】stamina（肺活量）が計画を切らないか。taro_core/Fパイプラインに
# 肺(Lungs)が移植されているかを実測する。
has_lungs = hasattr(tr.taro, "lungs")
print(f"\n  実装作業⑦：taro.lungs 属性の有無  {has_lungs}"
      f"（無ければ stamina 機構がF側に無い＝generate()にstamina引数を渡していない"
      f"ため計画が切られることはない。作業記録参照）")

tr.run()
close_env(tr.env)

print(f"\n  例外なく3000step（sim300秒）走行できたか  はい")

wp.setup_done = True  # noop, keep for readability
events = wp.rows
print(f"\n  発話回数={len(events)}")
for r in events[:20]:
    print(f"    sim_sec={r['sim_sec']:.1f} toy={r['toy']} target={r['target_word']}"
          f" -> generated={r['generated_word']}"
          f" plan_length={r['plan_length']} known_moras={r['known_moras']}"
          f" exact_match={r['exact_match']} reward={r['reward']}")

wan = [r for r in events if r["target_word"] == "わんわん"]
bu = [r for r in events if r["target_word"] == "ぶーぶー"]

print(f"\n  検証2：「わんわん」発話回数={len(wan)}")
if wan:
    said = [r["generated_word"] for r in wan]
    print(f"    実際に言った文字列（先頭10件）: {said[:10]}")
ok2 = any(r["generated_word"] == "わんわん" for r in wan)
print(f"    「わんわん」と完全一致した発話があるか  {'はい（合格）' if ok2 else 'いいえ（不合格）'}")

print(f"\n  検証3：発話長が計画(4モーラ)どおりか")
lens = [len(r["generated_word"]) for r in events]
from collections import Counter
print(f"    発話長の分布={dict(sorted(Counter(lens).items()))}")
ok3 = len(lens) > 0 and all(n == 4 for n in lens)

print(f"\n  検証4：「ぶーぶー」発話回数={len(bu)}")
if bu:
    said_bu = [r["generated_word"] for r in bu]
    print(f"    実際に言った文字列（先頭10件）: {said_bu[:10]}")
    ok4 = any(w == "ぶあぶあ" for w in said_bu)
    ok4_no_exact = not any(w == "ぶーぶー" for w in said_bu)
else:
    ok4 = False
    ok4_no_exact = True
    print("    発話が1回も無く判定不能")
print(f"    「ぶあぶあ」が出たか  {'はい（合格）' if ok4 else 'いいえ（不合格/未観測）'}")

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
print(f"  検証5（帳面が読み込まれているか）  {'合格' if ok5 else '不合格'}")
print(f"  検証2（「わんわん」と言えるか）    {'合格' if ok2 else '不合格'}")
print(f"  検証3（発話長=4モーラ）            {'合格' if ok3 else '不合格'}")
print(f"  検証4（「ぶあぶあ」になるか）      {'合格' if ok4 else '不合格/未観測'}")
