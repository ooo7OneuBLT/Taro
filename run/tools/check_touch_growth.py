# -*- coding: utf-8 -*-
"""触覚が「体を育てても同じ部位を読み続けるか」を確かめる。

【なぜ要るか、2026-07-31】太郎は触覚ONで体を育てることを禁止していた。
理由は「体を育てるとセンサー点が 0ヶ月4,824 → 4ヶ月9,804次元 に変わる」
（落とし穴チェックリスト 項75）。

さらに悪いことに、旧 SomatosensoryCortex は**エラーを出さずに壊れていた**。
部位ごとに `nn.Linear(点数×3, dim)` を置き、`index_select` で
作った時点のインデックス表を持ち続けるので、配列が長くなっても
**範囲内に収まって例外にならず、別の部位を読む**（項86）。

⇒ 2026-07-31 に部位ごとの出力を「有無・強さ・重心xyz」の5つ
  （**センサ点数によらず固定**）に作り変えた。このスクリプトはその検証。

確かめること
  ① 月齢を変えても「センサーを持つ部位の集合」が同じか
  ② パラメータ数が月齢で変わらないか（＝重みが引き継げるか）
  ③ **同じ場所を触ったら、月齢が違っても同じ部位が反応するか**（本丸）
  ④ 地図を差し替えずに次元の違う観測を入れたら**例外で止まるか**
     （静かに壊れないこと。項86 の再発防止）

使い方:
    .venv/Scripts/python.exe run/tools/check_touch_growth.py
    .venv/Scripts/python.exe run/tools/check_touch_growth.py 0 2 4
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
for p in ("run/scene_tools", "D/scripts", "taro_core/src/body", "taro_core/src/brain",
          "taro_core/src/senses", "taro_core/src/wrapper", "MIMo", ""):
    sys.path.insert(0, os.path.join(_R, p))
os.chdir(_R)

import numpy as np              # noqa: E402
import torch                    # noqa: E402
import e_scene                  # noqa: E402
from somatosensory_cortex import (       # noqa: E402
    SomatosensoryCortex, build_touch_map_from_env)

ages = [float(a) for a in sys.argv[1:]] or [0.0, 2.0, 4.0]

print("=" * 78)
print(" 触覚は体を育てても同じ部位を読み続けるか")
print("=" * 78)


def _make(age):
    sc = e_scene.load("新生児_仰向け_柵なし")
    sc["body"]["age_months"] = age
    sc["fingerprint"] = None
    env, _ = e_scene.build(sc, seed=0, verbose=False)
    return env


maps = {}
for age in ages:
    env = _make(age)
    if getattr(env.unwrapped, "touch", None) is None:
        print(f"注意 月齢{age}: 触覚が有効になっていない")
        env.close()
        continue
    maps[age] = build_touch_map_from_env(env)
    tm = maps[age]
    print(f"\n--- 月齢 {age:.1f} ヶ月")
    print(f"  センサー点 {tm.n_points:,} 点（flat {tm.total_dim:,}次元）")
    print(f"  部位グループ {len(tm.group_names)} 個")
    env.close()

if len(maps) < 2:
    print("\n月齢が1つしかないので比較できない")
    sys.exit(0)

keys = sorted(maps)
base = maps[keys[0]]

# ---------------------------------------------------------------- ①部位の集合
print("\n" + "=" * 78)
print(" ① 部位グループの集合が全月齢で同じか")
print("=" * 78)
same_groups = all(maps[a].group_names == base.group_names for a in keys)
print(f"  {'はい' if same_groups else 'いいえ'}")
for a in keys:
    print(f"    月齢{a:.1f}: {len(maps[a].group_names)}部位  "
          f"{maps[a].n_points:,}点")
if not same_groups:
    for a in keys:
        d = set(maps[a].group_names) ^ set(base.group_names)
        if d:
            print(f"    注意 月齢{a:.1f} の差分: {sorted(d)}")

# ---------------------------------------------------------------- ②層の形
print("\n" + "=" * 78)
print(" ② 脳の側（層の形・パラメータ数）が月齢で変わらないか")
print("=" * 78)
brains = {a: SomatosensoryCortex(maps[a], embedding_dim=64) for a in keys}
n_params = {a: sum(p.numel() for p in brains[a].parameters()) for a in keys}
for a in keys:
    print(f"    月齢{a:.1f}: {n_params[a]:,} パラメータ"
          f"（統合層 入力{brains[a].integrate.in_features}"
          f" → 出力{brains[a].integrate.out_features}）")
same_params = len(set(n_params.values())) == 1
print(f"  全月齢で同じか  {'はい' if same_params else 'いいえ'}")

# ---------------------------------------------------------------- ③同じ場所
print("\n" + "=" * 78)
print(" ③ 同じ場所を触ったら、月齢が違っても同じ部位が反応するか（本丸）")
print("=" * 78)
print("  やり方：ある部位の点だけに力を入れた偽の観測を作り、")
print("         その部位の『有無』が最大になるかを見る")


def _probe(brain, tm, group):
    """group の点にだけ力を入れた観測を作り、部位ごとの『有無』を返す。"""
    gi = tm.group_names.index(group)
    f = np.zeros((tm.n_points, 3), dtype=np.float32)
    f[tm.part_of_point == gi] = 1.0
    with torch.no_grad():
        feat = brain.part_features(torch.as_tensor(f.reshape(-1)))
    return feat[..., 0].numpy()          # 0列目＝有無


targets = ["right_palm", "left_foot", "head", "right_ff"]
targets = [t for t in targets if t in base.group_names]
ok3 = True
for tgt in targets:
    line = f"    {tgt:16s}"
    for a in keys:
        # 月齢 a の体の地図で脳を作り直す（＝成長したときにやること）
        b = SomatosensoryCortex(maps[keys[0]], embedding_dim=64)
        b.rebuild(maps[a])
        peak = _probe(b, maps[a], tgt)
        top = b.group_names[int(np.argmax(peak))]
        hit = (top == tgt)
        ok3 &= hit
        line += f"  月齢{a:.0f}→{top}{'' if hit else ' 不一致'}"
    print(line)
print(f"  すべて一致したか  {'はい' if ok3 else 'いいえ'}")

# ------------------------------------------------------- ④静かに壊れないこと
print("\n" + "=" * 78)
print(" ④ 地図を差し替えずに違う体の観測を入れたら、止まるか")
print("=" * 78)
print("  （旧版はここで**例外を出さずに別の部位を読んでいた**＝落とし穴 項86）")
a0, a1 = keys[0], keys[-1]
b = SomatosensoryCortex(maps[a0], embedding_dim=64)
x_new = torch.zeros(maps[a1].total_dim)
try:
    with torch.no_grad():
        b(x_new)
    print(f"  注意 止まらなかった。月齢{a0}の脳に月齢{a1}の観測"
          f"（{maps[a1].total_dim:,}次元）が通ってしまった")
    ok4 = False
except AssertionError as e:
    print(f"  止まった：{str(e).splitlines()[0]}")
    ok4 = True

# rebuild すれば通ること
b.rebuild(maps[a1])
with torch.no_grad():
    y = b(x_new)
print(f"  rebuild 後は通る：出力 {tuple(y.shape)}")

# ------------------------------------------------- ⑤実環境で正しい部位が出るか
print("\n" + "=" * 78)
print(" ⑤ 実際に寝かせたとき、床に触れている部位が反応するか")
print("=" * 78)
print("  仰向けなので、背中側（後頭部・体幹・尻）が反応するはず")
ok5 = True
for a in keys:
    env = _make(a)
    obs, _ = env.reset(seed=0)
    for _ in range(20):                       # 少し落ち着かせる
        obs, *_ = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
    tm = build_touch_map_from_env(env)
    b = SomatosensoryCortex(tm, embedding_dim=64)
    with torch.no_grad():
        feat = b.part_features(torch.as_tensor(obs["touch"], dtype=torch.float32))
    pres = feat[..., 0].numpy()
    order = np.argsort(-pres)
    touched = [(tm.group_names[i], float(pres[i])) for i in order if pres[i] > 1e-6]
    print(f"\n    月齢{a:.1f}：触れている部位 {len(touched)}／{len(tm.group_names)}")
    for name, v in touched[:8]:
        print(f"      {name:20s} 有無{v:.3f}")
    if not touched:
        print("      注意 どこにも触れていない。床との接触が取れていない可能性")
        ok5 = False
    env.close()

# --------------------------------------------------- ⑥学習の勾配が流れているか
print("\n" + "=" * 78)
print(" ⑥ 触覚の各パラメータに学習の勾配が届くか")
print("=" * 78)
print("  （層があっても勾配がゼロなら『繋がっているつもり』で学習していない）")
b = SomatosensoryCortex(maps[keys[0]], embedding_dim=64)
x = torch.rand(maps[keys[0]].total_dim) * 0.1
b(x).sum().backward()
ok6 = True
for name, p in b.named_parameters():
    g = p.grad
    val = None if g is None else float(g.abs().sum())
    live = g is not None and val > 0
    ok6 &= live
    print(f"    {name:16s} 勾配{'あり' if live else 'なし'}"
          + (f"（合計{val:.4g}）" if val is not None else ""))

# ---------------------------------------------------------------- まとめ
print("\n" + "=" * 78)
print(" 判定")
print("=" * 78)
allok = same_groups and same_params and ok3 and ok4 and ok5 and ok6
for name, v in [("① 部位の集合が同じ", same_groups),
                ("② 層の形が同じ", same_params),
                ("③ 同じ場所→同じ部位", ok3),
                ("④ 地図なしでは止まる", ok4),
                ("⑤ 実環境で接触が出る", ok5),
                ("⑥ 勾配が届く", ok6)]:
    print(f"  {name:24s} {'合格' if v else '不合格'}")
print()
print("  ⇒ " + ("触覚ONで体を育てられる" if allok else
                "注意 まだ問題がある。上の不合格の項を見る"))

# 参考：部位ごとの点数の変化
print("\n" + "-" * 78)
print(" 部位ごとの点数（増えてよい。要約が5次元固定なので層の形は変わらない）")
print("-" * 78)
counts = {a: dict(zip(maps[a].group_names, maps[a].counts)) for a in keys}
top = sorted(base.group_names, key=lambda g: -counts[keys[-1]].get(g, 0))[:10]
print(f"{'部位':<20}" + "".join(f"{a:>12.1f}ヶ月" for a in keys))
for g in top:
    print(f"{g:<20}" + "".join(f"{int(counts[a].get(g, 0)):>17,}" for a in keys))
