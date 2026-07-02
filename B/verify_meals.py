"""
時間割授乳（B2-12）の動作確認。短期シミュで
  - 1日の食事回数が人間域(6〜10)に収まるか
  - 時間割授乳のうち「満腹寄り(hunger<0.5)」で行われた割合＝語と空腹の脱相関
を実測する。使い方: python verify_meals.py [日数]（既定10）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import torch
torch.set_num_threads(2)
from environment.parent_sim_b import run_simulation_b

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
r = run_simulation_b(max_sim_seconds=DAYS * 86400, verbose=False, run_name="verify_meals")

feeds_day = r["feed_count"] / DAYS
meals = r["meal_count"]
low = r["meal_low_hunger"]
frac = (low / meals * 100) if meals else 0.0
print(f"\n=== 時間割授乳 動作確認（{DAYS}日）===")
print(f"総食事: {r['feed_count']}回  → {feeds_day:.1f}回/日（人間目安6〜10）")
print(f"  うち時間割授乳: {meals}回  オンデマンド: {r['feed_count']-meals}回")
print(f"  時間割のうち満腹寄り(hunger<0.5): {low}回  = {frac:.0f}%  ← 語と空腹の脱相関")
print(f"泣き/日: {r['cry_count']/DAYS:.1f}  要求語/日: {r['request_count']/DAYS:.1f}")
