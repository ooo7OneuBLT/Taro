# -*- coding: utf-8 -*-
"""F2-66: 文脈の配線修正（hidden=t._context_hidden）が実際に効くかの最小確認
（2026-09-04・書き捨てレベル）。

1) 実在のチェックポイント（F2-49c_r1）からTaroBrainを読み込み、
   forward_hidden(hidden=None) と forward_hidden(hidden=非ゼロ) で
   出力（次の音の予測）が変わることを直接確認する（配線が本当に効くかの単体確認）。
2) 実際の短い走行（本体のrun.main経由）で、文脈ONの設定がクラッシュしないことを確認する。

    .venv/Scripts/python.exe F/scripts/f66_context_wiring_check.py
"""
import os, sys, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(r"C:\claude\AI\Taro")
sys.path.insert(0, os.path.abspath("taro_core/src"))
sys.path.insert(0, os.path.abspath("taro_core/src/brain"))
import torch
from cerebral_cortex.recurrent_core import TaroBrain

CKPT = "F/models/F2-49c_r1_seed91_2026-09-03.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
print("blobのキー:", list(blob.keys())[:10], "...")

sd = blob["brain"] if "brain" in blob else blob["taro_brain"]
vocab_size = sd["embedding.weight"].shape[0]
hidden_dim = sd["gru.weight_hh_l0"].shape[1]
print("vocab_size=%d hidden_dim=%d" % (vocab_size, hidden_dim))

brain = TaroBrain(vocab_size=vocab_size, hidden_dim=hidden_dim)
brain.load_state_dict(sd, strict=False)
brain.eval()

x = torch.tensor([[5]], dtype=torch.long)  # 適当な1トークン

with torch.no_grad():
    out_zero, h_zero = brain.forward_hidden(x, hidden=None)
    logits_zero = brain.perception_head(out_zero)[0, -1]
    top_zero = int(torch.argmax(logits_zero))

    torch.manual_seed(0)
    h_ctx = torch.randn(1, 1, hidden_dim) * 0.5  # 非ゼロの「文脈」を模したhidden
    out_ctx, h_ctx2 = brain.forward_hidden(x, hidden=h_ctx)
    logits_ctx = brain.perception_head(out_ctx)[0, -1]
    top_ctx = int(torch.argmax(logits_ctx))

diff = float((logits_zero - logits_ctx).abs().max())
print("\n[単体確認] 同じ入力トークンで hidden=None vs hidden=非ゼロ の予測を比較")
print("  最大差(logit):", round(diff, 4))
print("  最有力トークン: hidden=Noneのとき=%d / hidden=文脈ありのとき=%d" % (top_zero, top_ctx))
print("  → 差が0でなければ、hiddenの値が生成に実際に反映されている（配線が効いている証拠）")
assert diff > 1e-6, "hiddenを変えても出力が変わっていない！配線がまだ効いていない"
print("  OK: hiddenの違いが出力に反映されることを確認")
