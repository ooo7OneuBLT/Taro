# -*- coding: utf-8 -*-
"""F2-1「喃語で口の内部モデルを作る」：既定OFFの原則の実測確認。

設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「検証」1・4。

  検証1：mode 未指定・jaw_cycles 未指定（＝既存のF2 word産出呼び出し）で
         generate() の出力（生成文字列・log_probs）が乱数列を含め完全一致すること。
         新（現行コード）と旧（git HEAD時点の taro_core/src/brain/taro_brain.py）
         を別プロセスで動かして突き合わせる（同一プロセスの二重importはモジュール
         キャッシュで新しい方しか読めないため）。

         【実装作業⑥・想定外の経緯】設計は息切れ停止（breath_pressure/
         stop_prob）を「設定で戻せる形にはせず削除」としていたが、字義どおり
         削除すると既存のF2 word産出呼び出し（speech_planを渡さないため
         is_babble=Trueの同じ分岐を通っていた）の乱数列が変わってしまうことが
         この検証で判明した。そのため「jaw_cyclesが指定されている（＝F2-1の
         喃語モード）ときだけ息切れ停止を無効化する」形に変更した
         （taro_core/src/brain/taro_brain.py generate() 内のコメント参照）。

  検証4：息切れ停止の撤去（jaw_cycles指定時のみ）で、他の経路
         （目標B相当の呼び出し＝BrocasArea.plan()が作ったspeech_planを渡す
         呼び出し）が壊れないこと。speech_planありの経路はもともと
         is_babble=Falseだったため無関係だが、例外なく完走することを確認する。

使い方:
    .venv/Scripts/python.exe run/tools/check_f2_1_babble_equivalence.py
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\claude\AI\Taro"
os.chdir(ROOT)
BRAIN_DIR = os.path.join(ROOT, "taro_core", "src", "brain")
SCRATCH = os.path.join(ROOT, "run", "tools", "_scratch_f2_1")
os.makedirs(SCRATCH, exist_ok=True)
OLD_FILE = os.path.join(SCRATCH, "taro_brain_old.py")

# git HEAD時点（このF2-1実装より前）のtaro_brain.pyを取り出す
r = subprocess.run(["git", "show", "HEAD:taro_core/src/brain/taro_brain.py"],
                    cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
if r.returncode != 0:
    print("git show に失敗（gitリポジトリでない可能性）。検証1をスキップします。")
    print(r.stderr)
    sys.exit(1)
with open(OLD_FILE, "w", encoding="utf-8") as fp:
    fp.write(r.stdout)

RUN_SCRIPT = r"""
import sys, random
import torch
sys.path.insert(0, r"__BRAIN_DIR__")
__EXTRA_PATH__
from vocal_tract import VocalTract
__IMPORT_LINE__

results = []
for trial in range(5):
    random.seed(1234 + trial)
    torch.manual_seed(1234 + trial)
    brain = TaroBrain(vocab_size=3)
    vt = VocalTract()
    vt.stage = 2
    vt.force_decouple()
    char2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    for ch in vt.get_all_chars():
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)
    brain.resize_embedding(len(char2idx))
    brain.set_vocab_mapping(char2idx)
    # 既存のF2 word産出呼び出しと同じ引数（run/trainer.py _apply_word_production）：
    # speech_plan・cerebellum・jaw_cyclesはどれも渡さない。
    generated, log_probs, hidden = brain.generate(
        hidden=None, max_length=8, eos_idx=2,
        vocal_tract=vt, ne_level=0.5)
    results.append((generated, [round(float(x), 6) for x in log_probs]))
print(results)
"""

script_new = (RUN_SCRIPT.replace("__BRAIN_DIR__", BRAIN_DIR)
              .replace("__EXTRA_PATH__", "")
              .replace("__IMPORT_LINE__", "from taro_brain import TaroBrain"))
p_new = subprocess.run([sys.executable, "-c", script_new], capture_output=True, text=True, encoding="utf-8")

script_old = (RUN_SCRIPT.replace("__BRAIN_DIR__", BRAIN_DIR)
              .replace("__EXTRA_PATH__", f"sys.path.insert(0, r'{SCRATCH}')")
              .replace("__IMPORT_LINE__", "from taro_brain_old import TaroBrain"))
p_old = subprocess.run([sys.executable, "-c", script_old], capture_output=True, text=True, encoding="utf-8")

print("=" * 78)
print(" 検証1：既存word産出呼び出し（mode/jaw_cycles未指定）の出力が新旧で一致するか")
print("=" * 78)
if p_new.returncode != 0:
    print("新コードでエラー:\n", p_new.stderr[-3000:])
if p_old.returncode != 0:
    print("旧コードでエラー:\n", p_old.stderr[-3000:])
ok1 = p_new.returncode == 0 and p_old.returncode == 0 and \
      p_new.stdout.strip() == p_old.stdout.strip()
print("新:", p_new.stdout.strip()[:300])
print("旧:", p_old.stdout.strip()[:300])
print(f"5トライアルとも完全一致  {'はい（合格）' if ok1 else 'いいえ（不合格）'}")

# ---- 検証4：speech_planありの経路（目標B相当）が壊れないか ------------------
print()
print("=" * 78)
print(" 検証4：speech_planあり（目標B相当）の経路が壊れないか")
print("=" * 78)
sys.path.insert(0, BRAIN_DIR)
from taro_brain import TaroBrain            # noqa: E402
from vocal_tract import VocalTract          # noqa: E402
from cerebellum import Cerebellum           # noqa: E402
from left_frontal_lobe.brocas_area import BrocasArea  # noqa: E402

ok4 = True
try:
    brain = TaroBrain(vocab_size=3)
    vt = VocalTract()
    vt.stage = 2
    vt.force_decouple()
    char2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    for ch in vt.get_all_chars():
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)
    brain.resize_embedding(len(char2idx))
    brain.set_vocab_mapping(char2idx)

    cereb = Cerebellum()
    cereb.learn_from_experience(0, 6, 1, 2, "わ")
    cereb.learn_from_experience(4, 3, 1, 0, "ん")
    brocas = BrocasArea()
    brocas.plan("わんわん", cereb, vt)
    plan_len = brocas.get_plan_length()
    generated, log_probs, hidden = brain.generate(
        hidden=None, max_length=8, eos_idx=2, vocal_tract=vt, ne_level=0.5,
        cerebellum=cereb, speech_plan=brocas)
    print(f"  計画長={plan_len}  生成文字数={len(generated)}  例外なし")
except Exception as e:      # noqa: BLE001
    ok4 = False
    print(f"  例外発生：{type(e).__name__}: {e}")
print(f"  speech_planあり経路が壊れていないか  {'はい（合格）' if ok4 else 'いいえ（不合格）'}")

print()
print("=" * 78)
print(" 判定")
print("=" * 78)
allok = ok1 and ok4
print(f"  検証1（既定OFFの原則・既存word呼び出し）  {'合格' if ok1 else '不合格'}")
print(f"  検証4（speech_plan経路が壊れない）        {'合格' if ok4 else '不合格'}")

import shutil
shutil.rmtree(SCRATCH, ignore_errors=True)

if not allok:
    sys.exit(1)
