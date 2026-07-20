"""E1環境の動作確認：おもちゃは「届く所にあり・自分が動かすと動く」か。

【何を確かめるか】設計(全体設計§10)が成り立つ最低条件。
  ①配置   : 静止時は触れていないが、肩からは REACH_MAX(0.160m)以内＝暴れれば届く。
  ②接触   : 運動性喃語の間に**実際に接触**するか（距離でなくMuJoCoの接触で判定）。
             どの部位が触れるかも見る（手/脚/胴）。
  ③随伴性 : **太郎が動くとおもちゃが動き、動かなければ動かない**か。
             ＝Rochatのおしゃぶり実験と同じ構造（自分の行為→結果）。

【検証設計の変遷（測り方を2回直した）】
 第1版: 「おもちゃが動いたtick割合」→ 置き直し(respawn)の瞬間移動まで数えていた＝欠陥。
 第2版: respawnを除き「手が近い/遠い」で層別 → NEAR/FAR比 2.8x で判定NGに見えた。
        しかし**随伴性は"手"に限らない**（乳児研究の古典mobile paradigmは"足で蹴って"
        随伴を学ぶ）。手だけを見るのは問い自体が狭かった。
 本版  : **行動あり(babbling) vs 行動なし(action=0)** でおもちゃの動きを比較する。
        これが「自分の運動が結果を生むか」の正しい対照＝随伴性の定義そのもの。

使い方: python e_toy_check.py [n_steps] [age]
"""
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
from e_toy_env import ToySupineEnv, REACH_MAX  # noqa: E402

HAND_PARTS = ("right_hand", "left_hand", "right_fingers", "left_fingers")


def rollout(env, n_steps, active):
    """active=True: ランダム行動（運動性喃語）／False: 無行動。おもちゃの動きを集計。"""
    obs, _ = env.reset(seed=0)
    n_act = env.action_space.shape[0]
    prev = env.data.body("test_object1").xpos.copy()
    moves, contacts, n_contact_tick = [], Counter(), 0
    n_resp0 = env.n_respawn
    for i in range(n_steps):
        a = (np.random.uniform(-1.0, 1.0, size=n_act) if active
             else np.zeros(n_act))
        for _ in range(20):
            obs, _, te, tr, _ = env.step(a)
            p = env.data.body("test_object1").xpos
            if not env.respawned_this_step:
                moves.append(float(np.linalg.norm(p - prev)))
            prev = p.copy()
            cs = env.toy_contacts()
            body_parts = [c for c in cs if c != "world"]
            if body_parts:
                n_contact_tick += 1
                contacts.update(body_parts)
            if te or tr:
                break
        if te or tr:
            obs, _ = env.reset()
            prev = env.data.body("test_object1").xpos.copy()
    return (np.asarray(moves), contacts, n_contact_tick,
            env.n_respawn - n_resp0)


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    age = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    print("=== E1 toy environment check ===")
    env = ToySupineEnv(toy=True, age=age, vision_params=None, touch_params=None)
    obs, _ = env.reset(seed=0)
    d_sh0, d_hd0 = env.toy_distance(), env.hand_toy_distance()
    print(f"\n[1] placement at reset (age={age:.0f}mo)")
    print(f"    shoulder -> toy : {d_sh0:.3f} m  (reachable if < {REACH_MAX:.3f})")
    print(f"    hand     -> toy : {d_hd0:.3f} m  (not touching)")

    mv_act, ct_act, ntick_act, resp_act = rollout(env, n_steps, active=True)
    mv_still, ct_still, ntick_still, resp_still = rollout(env, n_steps, active=False)
    env.close()

    print(f"\n[2] contact (MuJoCo contacts, not distance)")
    print(f"    babbling : contact on {ntick_act/max(len(mv_act),1)*100:5.1f}% of ticks"
          f"   (respawns={resp_act})")
    print(f"    still    : contact on {ntick_still/max(len(mv_still),1)*100:5.1f}% of ticks")
    if ct_act:
        print("    which body parts touch the toy while babbling:")
        tot = sum(ct_act.values())
        for name, c in ct_act.most_common(6):
            tag = " <- HAND" if name in HAND_PARTS else ""
            print(f"      {name:20s} {c/tot*100:5.1f}%{tag}")
    else:
        print("    (no contact at all while babbling)")

    print(f"\n[3] contingency: does MY movement move the toy?")
    print(f"    babbling : total {mv_act.sum():.3f} m, "
          f"mean {mv_act.mean()*1000:.3f} mm/tick")
    print(f"    still    : total {mv_still.sum():.3f} m, "
          f"mean {mv_still.mean()*1000:.3f} mm/tick")
    ratio = mv_act.mean() / max(mv_still.mean(), 1e-9)
    print(f"    ratio (babbling / still) : {ratio:.1f}x")

    print("\n=== verdict ===")
    hand_share = (sum(ct_act[p] for p in HAND_PARTS if p in ct_act)
                  / max(sum(ct_act.values()), 1))
    checks = [
        ("placement (reachable & not pre-touching)",
         d_sh0 < REACH_MAX and d_hd0 > 0.02),
        ("real contact happens while babbling", ntick_act > 0),
        ("contingency: my movement moves the toy", ratio > 3.0),
        ("hand is among the parts that touch it", hand_share > 0.05),
    ]
    for name, ok in checks:
        print(f"  {name:42s} : {'OK' if ok else 'NG'}")
    print(f"\n  hand share of contacts : {hand_share*100:.1f}%")
    if ntick_act == 0:
        print("  -> never touched. Move the toy closer / make it bigger.")
    elif hand_share < 0.05:
        print("  -> touched, but almost never by the HAND. E1 aims at reaching,")
        print("     so consider moving the toy into the hand's sweep (see e_reach_space).")


if __name__ == "__main__":
    main()
