"""シーン方式そのものの自己テスト — 「食い違いが本当に止まるか」を確かめる。

【なぜ要るか、2026-07-29】食い違いを検知する仕組みを入れても、
**その仕組み自体が効いていなければ意味がない**。
＝落とし穴チェックリスト 項62「設定した値が本当に体に届いているか」と
項「『動かない』はまず計測器を疑う」を、この仕組みにも適用する。

【何を確かめるか】
  1. 同じシーンを2回作ったら、指紋が一致するか（再現できるか）
  2. わざと条件を変えたら、照合が**止めてくれるか**
     ・四肢の屈曲を変える  … 2026-07-28 に実際に起きた食い違い
     ・首の支える角度を変える … 同上（30度で保存し60度で測っていた）
     ・リクライニング角を変える
  3. 姿勢（qpos）が本当に復元されているか

使い方:
    .venv/Scripts/python.exe E/scripts/e_scene_selftest.py [シーン名]
"""
import os
import sys
import copy
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402
import e_scene          # noqa: E402

DEFAULT = "視線誘導反射_仰向け_頭を支える"
_results = []


def check(label, ok, detail=""):
    _results.append((label, ok))
    print(f"  {'OK  ' if ok else '★NG '} {label}" + (f"   {detail}" if detail else ""))
    return ok


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    scene = e_scene.load(name)
    print("=" * 78)
    print(f" シーン方式の自己テスト — 「{name}」")
    print("=" * 78)
    print(f"  {scene.get('note','')}")

    # ---- 1. 同じシーンを作り直したら一致するか ----------------------------
    print("\n1. 同じシーンをもう一度作ったら、同じ環境になるか")
    print("-" * 78)
    env, hands = e_scene.build(scene, orient=False, seed=0)
    diffs = e_scene.verify(scene, env, strict=False, verbose=False)
    check("保存した指紋と一致する", not diffs,
          "" if not diffs else f"{len(diffs)}件ずれた")
    for t in diffs:
        print(f"        {t}")

    # ---- 2. 姿勢（qpos）が復元されているか --------------------------------
    print("\n2. 保存した姿勢が、そのまま復元されているか")
    print("-" * 78)
    q_saved = np.asarray(scene["state"]["qpos"], dtype=float)
    q_now = np.asarray(env.unwrapped.data.qpos, dtype=float)
    err = float(np.max(np.abs(q_saved - q_now)))
    check("qpos が一致する", err < 1e-9, f"最大の差 {err:.2e}")

    fp = e_scene.fingerprint(env)
    rec = scene["fingerprint"]
    print(f"     首 記録={rec['head_angles_deg']}")
    print(f"        今  ={fp['head_angles_deg']}")
    print(f"     体幹の傾き 記録={rec['trunk_tilt_deg']}度 / 今={fp['trunk_tilt_deg']}度")
    env.close()

    # ---- 3. わざと違う条件にしたら、止めてくれるか ------------------------
    print("\n3. わざと条件を変えたら、照合が止めてくれるか")
    print("-" * 78)
    print("   （ここで止まらないなら、仕組みが効いていない）")

    cases = [
        ("首を支える角度を +30度 ずらす（30度で保存し60度で測っていた事故）",
         _shift_hold_tilt),
        ("リクライニングを +20度 変える",
         lambda s: s["world"].update(recline_deg=s["world"]["recline_deg"] + 20.0)),
        ("月齢を +2.0ヶ月 変える",
         lambda s: s["body"].update(age_months=s["body"]["age_months"] + 2.0)),
        ("柵の有無を反転",
         lambda s: s["world"].update(fence=not s["world"]["fence"])),
    ]
    # ★四肢の屈曲は3ヶ月までしか効かない（infant_body.FLEXION_UNTIL_MO = 3.0）。
    #   4ヶ月のシーンで反転しても**本当に何も変わらない**ので、テストにならない。
    #   ⇒ 2026-07-28 に見つけた「Viewerと測定で flexion が違う」という食い違いは、
    #     4ヶ月の実験には実害がなく、影響するのは0〜3ヶ月の実験（運動性喃語の学習など）。
    if scene["body"]["age_months"] < 3.0:
        cases.insert(0, ("四肢の屈曲を反転（0〜3ヶ月でのみ意味を持つ）",
                         lambda s: s["body"].update(flexion=not s["body"]["flexion"])))
    else:
        print(f"   （四肢の屈曲は3ヶ月までしか効かないので、"
              f"{scene['body']['age_months']:g}ヶ月のこのシーンでは試さない）")

    for label, mutate in cases:
        bad = copy.deepcopy(scene)
        mutate(bad)
        try:
            env2, _ = e_scene.build(bad, orient=False, seed=0)
        except Exception as e:
            # 姿勢の長さが変わる（柵の有無など）ならここで落ちる＝それも「止まった」
            check(label, True, f"環境を作る時点で止まった: {str(e)[:60]}")
            continue
        try:
            d2 = e_scene.verify(scene, env2, strict=False, verbose=False)
            check(label, bool(d2), f"{len(d2)}件のずれを検知"
                  if d2 else "★見逃した（照合をすり抜けた）")
            for t in d2[:3]:
                print(f"        {t}")
        finally:
            env2.close()

    # ---- まとめ ------------------------------------------------------------
    ng = [l for l, ok in _results if not ok]
    print("\n" + "=" * 78)
    if ng:
        print(f" ★{len(ng)}件が不合格 — 仕組みが効いていない")
        for l in ng:
            print(f"    {l}")
    else:
        print(f" 全{len(_results)}件が合格 — 食い違いは検知できる")
    print("=" * 78)
    return 1 if ng else 0


def _shift_hold_tilt(s):
    h = s["setup"].get("head_hold") or {}
    tgt = dict(h.get("target_deg") or {})
    tgt["head_tilt"] = float(tgt.get("head_tilt", 0.0)) + 30.0
    h["target_deg"] = tgt
    s["setup"]["head_hold"] = h


if __name__ == "__main__":
    sys.exit(main())
