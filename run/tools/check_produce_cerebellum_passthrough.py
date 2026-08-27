# -*- coding: utf-8 -*-
"""作業A（2026-08-24）：帳面（発話小脳 produce_cerebellum）がフェーズをまたぐと
静かに消える問題の修正を検証する。

背景：`run/taro_setup.py` の `_setup_produce()` は、実験ファイルに
`taro.produce` キーが無いと既定OFF分岐へ入り、`_load()` が保存モデルから
退避しておいた帳面（`_pending_produce_cerebellum`）を、以前は無条件に
`None` へ上書きして捨てていた（実測：帳面339行・68種を持つモデルを
produceキー無し実験で走らせて保存すると、blobから"produce_cerebellum"
キーが消えていた）。今回、OFF分岐でも退避データを保持し、save()で
素通しして保存するよう修正した。ここではその3点セットを検証する。

  制約1（既定OFFの原則）：帳面を持たないモデルでは、保存blobのキーが
        1つも増えないこと。
  制約2（バイト互換）：帳面を持つモデルをproduce無しで走らせて保存→
        再度読み込んで、forward_map/inverse_map/experience_countが
        件数・中身とも完全一致すること。
  制約3（静かに落ちない）：save()内のassertが機能上健在であること
        （直接の破壊確認は危険なので、assert文自体の存在とロジックを
        ソース上で確認する形にする）。

使い方:
    .venv/Scripts/python.exe run/tools/check_produce_cerebellum_passthrough.py

注意：短時間走行（300step=sim30秒）を2回行う。本走行ではない。
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

import torch  # noqa: E402

from run.config import Config              # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402

# 一時ファイルの置き場。環境変数 TARO_SCRATCH があればそれを使い、
# 無ければOSの一時ディレクトリ配下に作る（個人のパスを埋め込まない）。
_SCRATCH = os.environ.get("TARO_SCRATCH") or os.path.join(
    tempfile.gettempdir(), "taro_scratch")
os.makedirs(_SCRATCH, exist_ok=True)
PATH_WITH_CEREB = os.path.join(_SCRATCH, "check_pc_with_cereb.pt")
PATH_PASSTHROUGH = os.path.join(_SCRATCH, "check_pc_passthrough.pt")
PATH_BASELINE_IN = os.path.join(_SCRATCH, "check_pc_baseline_in.pt")
PATH_BASELINE_OUT = os.path.join(_SCRATCH, "check_pc_baseline_out.pt")

SCENE = "座位_6ヶ月_2おもちゃ_F1-3c_1個提示_2026-08-19"

print("=" * 78)
print(" 作業A：produce_cerebellum のフェーズ跨ぎ保存 検証")
print("=" * 78)

# ---------------------------------------------------------------------
# 手順1：produce ON（babble）で短時間走行し、帳面に経験を積んで保存する
# ---------------------------------------------------------------------
cfg1 = Config.from_spec({
    "scene": SCENE,
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
tr1 = Trainer(cfg1, verbose=False, log_row=lambda row: None)
tr1.build()
tr1.run()
cereb1 = tr1.taro.produce_cerebellum
n_fwd = len(cereb1.forward_map)
n_inv = len(cereb1.inverse_map)
n_exp = len(cereb1.experience_count)
print(f"\n[手順1] babble走行後の帳面  forward_map={n_fwd}件"
      f" inverse_map={n_inv}件 experience_count={n_exp}件")
assert n_fwd > 0, "帳面が育っていない（この検証の前提が崩れている）"
tr1.taro.save(PATH_WITH_CEREB)
close_env(tr1.env)

blob_with_cereb = torch.load(PATH_WITH_CEREB, weights_only=False)
assert "produce_cerebellum" in blob_with_cereb, "produce ON保存直後にキーが無い(前提崩壊)"
ref_fwd = blob_with_cereb["produce_cerebellum"]["forward_map"]
ref_inv = blob_with_cereb["produce_cerebellum"]["inverse_map"]
ref_exp = blob_with_cereb["produce_cerebellum"]["experience_count"]

# ---------------------------------------------------------------------
# 手順2：その帳面を持つモデルを「produceキー無し」の実験（OFF）で読み込み、
#        1歩も進めずに保存する（＝語の学習フェーズを模した最短ケース）。
# ---------------------------------------------------------------------
cfg2 = Config.from_spec({
    "scene": SCENE,
    "taro": {
        "actuation": "muscle", "age_months": 6.0,
        "orienting_reflex": True, "hearing": True,
        "lexicon_vision": {"backend": "dinov2_vits14", "fovea_px": 32},
        "lexicon_mode": "contrast",
        "model": PATH_WITH_CEREB,
        # produce キーを書かない = 既定OFF
    },
    "run": {"type": "train", "steps": 1, "seed": 7, "K": 10, "checkpoint": 1},
})
tr2 = Trainer(cfg2, verbose=False, log_row=lambda row: None)
tr2.build()
print(f"\n[手順2] OFF設定で読み込み直後  produce_cerebellum={tr2.taro.produce_cerebellum}"
      f"（Noneであるべき＝OFF既定は変えない）")
assert tr2.taro.produce_cerebellum is None, "OFFなのにインスタンスが作られている(既定OFF崩壊)"
tr2.taro.save(PATH_PASSTHROUGH)
close_env(tr2.env)

blob_passthrough = torch.load(PATH_PASSTHROUGH, weights_only=False)
ok_key_present = "produce_cerebellum" in blob_passthrough
print(f"\n[制約2の下準備] OFF保存後に'produce_cerebellum'キーがあるか  "
      f"{'あり' if ok_key_present else 'なし'}")

ok_byte_equiv = False
if ok_key_present:
    got = blob_passthrough["produce_cerebellum"]
    ok_fwd = got["forward_map"] == ref_fwd
    ok_inv = got["inverse_map"] == ref_inv
    ok_exp = got["experience_count"] == ref_exp
    ok_byte_equiv = ok_fwd and ok_inv and ok_exp
    print(f"  forward_map一致: {ok_fwd}（件数 保存前={len(ref_fwd)} 保存後={len(got['forward_map'])}）")
    print(f"  inverse_map一致: {ok_inv}（件数 保存前={len(ref_inv)} 保存後={len(got['inverse_map'])}）")
    print(f"  experience_count一致: {ok_exp}（件数 保存前={len(ref_exp)} 保存後={len(got['experience_count'])}）")

# ---------------------------------------------------------------------
# 手順3：帳面を一度も持ったことが無いモデル（cfg.model無し・produceキー無し）
#        で保存し、blobのキーが従来から1つも増えていないことを確認する。
# ---------------------------------------------------------------------
cfg3a = Config.from_spec({
    "scene": SCENE,
    "taro": {
        "actuation": "muscle", "age_months": 6.0,
        "orienting_reflex": True, "hearing": True,
        "lexicon_vision": {"backend": "dinov2_vits14", "fovea_px": 32},
        "lexicon_mode": "contrast",
    },
    "run": {"type": "train", "steps": 1, "seed": 7, "K": 10, "checkpoint": 1},
})
tr3a = Trainer(cfg3a, verbose=False, log_row=lambda row: None)
tr3a.build()
assert tr3a.taro.produce_cerebellum is None
assert getattr(tr3a.taro, "_pending_produce_cerebellum", None) is None, (
    "cfg.model無しなのにpendingが存在する(前提崩壊)")
tr3a.taro.save(PATH_BASELINE_IN)
close_env(tr3a.env)

blob_baseline_in = torch.load(PATH_BASELINE_IN, weights_only=False)
keys_in = set(blob_baseline_in.keys())
ok_no_new_key_fresh = "produce_cerebellum" not in keys_in
print(f"\n[制約1・その1] 帳面ゼロの新規モデル保存に'produce_cerebellum'キーが"
      f"無いか  {'なし(合格)' if ok_no_new_key_fresh else 'あり(不合格)'}")

# 続けて、そのcfg.model無しモデルを一旦保存したものを「続きから」OFFで
# もう一段読み込み直して保存 → キー集合が増えないことも見る（構成の往復）
cfg3b = Config.from_spec({
    "scene": SCENE,
    "taro": {
        "actuation": "muscle", "age_months": 6.0,
        "orienting_reflex": True, "hearing": True,
        "lexicon_vision": {"backend": "dinov2_vits14", "fovea_px": 32},
        "lexicon_mode": "contrast",
        "model": PATH_BASELINE_IN,
    },
    "run": {"type": "train", "steps": 1, "seed": 7, "K": 10, "checkpoint": 1},
})
tr3b = Trainer(cfg3b, verbose=False, log_row=lambda row: None)
tr3b.build()
assert getattr(tr3b.taro, "_pending_produce_cerebellum", None) is None, (
    "帳面ゼロモデルの読み込みなのにpendingが非None(前提崩壊)")
tr3b.taro.save(PATH_BASELINE_OUT)
close_env(tr3b.env)

blob_baseline_out = torch.load(PATH_BASELINE_OUT, weights_only=False)
keys_out = set(blob_baseline_out.keys())
ok_no_new_key_roundtrip = "produce_cerebellum" not in keys_out
extra_keys = keys_out - keys_in
print(f"[制約1・その2] 帳面ゼロモデルをOFFで1往復しても新キーが増えないか  "
      f"{'なし(合格)' if not extra_keys else f'増えた: {extra_keys}(不合格)'}")

# ---------------------------------------------------------------------
# 制約3：assertがソース上に健在か（既に save() を直接読んで確認済みだが、
#        機械チェックとして文字列存在確認を残す）
# ---------------------------------------------------------------------
with open(os.path.join(_R, "run", "taro_setup.py"), encoding="utf-8") as f:
    src = f.read()
ok_assert_present = (
    "発話小脳(produce_cerebellum)を持っている" in src
    and "assert (self.produce_cerebellum is None and _pending_pc is None) or" in src
)
print(f"\n[制約3] save()の『静かに落ちない』assertがソースに健在か  "
      f"{'あり(合格)' if ok_assert_present else 'なし(不合格)'}")

print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = (ok_no_new_key_fresh and ok_no_new_key_roundtrip and ok_key_present
         and ok_byte_equiv and ok_assert_present)
print(f"  制約1（帳面ゼロで新キー無し）  {'合格' if (ok_no_new_key_fresh and ok_no_new_key_roundtrip) else '不合格'}")
print(f"  制約2（帳面ありOFF素通しで完全一致）  {'合格' if (ok_key_present and ok_byte_equiv) else '不合格'}")
print(f"  制約3（静かに落ちないassert健在）  {'合格' if ok_assert_present else '不合格'}")
print(f"  総合  {'合格' if allok else '不合格'}")
if not allok:
    sys.exit(1)
