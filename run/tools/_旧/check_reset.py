"""`env.reset()` を繰り返すと、毎回同じ状態になるかを調べる。

【なぜ要るか、2026-07-30】同じ設定・同じ乱数の種で学習を2回回すと結果が一致しない
（落とし穴チェックリスト 項79）。切り分けで「自己モデルの測定（`e_probes.evaluate`）を
入れると一致しなくなる」ことが分かった。evaluate が学習ループと違うことをしているのは
**環境を進めることと、エピソードが終わったら `env.reset()` を呼ぶこと**の2つ。

`env.reset()` は太郎の体を初期姿勢に戻すが、シーンで作った環境は
  ・実験者の手で頭を支える
  ・椅子でリクライニングを保つ
  ・四肢の筋緊張をかける
  ・おもちゃを胸の上に置く
といった設定を **reset のあとに1回だけ**適用している（`scene_io.build`）。
＝2回目以降の reset で何が起きるかは、確かめられていなかった。

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_reset            （1回目）
    .venv/Scripts/python.exe -m run.tools.check_reset            （2回目）
  2回の出力を比べる。同じなら reset は決定的。違えばここが原因。

注意：出力は「人が読んで比べる」ためのもの。数字が長いので、
  最後に載せる要約（ハッシュ）だけを比べればよい。
"""
import hashlib
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings                                     # noqa: E402
warnings.filterwarnings("ignore")
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

SCENE = "新生児_仰向け_柵あり"
N_RESET = 6           # reset を何回試すか
N_TICK = 50           # reset のあと何 tick 進めてから測るか


def _h(arr):
    """配列を短い文字列にする（比べやすくするため）。"""
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.float64).tobytes()
                        ).hexdigest()[:12]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    from run.plugins.common import scene as scene_mod
    env, sc, hands = scene_mod.build(SCENE, taro={"actuation": "muscle",
                                                  "age_months": 0.0},
                                     seed=0, verbose=False, hybrid=True)
    u = env.unwrapped
    m, d = u.model, u.data
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)

    print("=" * 78)
    print(f" env.reset() は毎回同じ状態になるか（シーン: {SCENE}）")
    print("=" * 78)
    print(f"  {'回':>3} {'reset直後のqpos':>16} {'+50tick後のqpos':>16} "
          f"{'手の高さ(cm)':>12} {'接触の数':>9}")

    rows = []
    for k in range(N_RESET):
        if k == 0:
            obs, _ = env.reset(seed=0)      # 1回目だけ種を渡す（学習と同じ）
        else:
            obs, _ = env.reset()            # 2回目以降は種なし（evaluate と同じ）
        h0 = _h(d.qpos)
        for _ in range(N_TICK):
            obs, _r, te, tr, _i = env.step(zero)
        h1 = _h(d.qpos)
        hand_z = float(d.body("right_hand").xpos[2]) * 100.0
        rows.append((h0, h1, hand_z, int(d.ncon)))
        print(f"  {k:>3} {h0:>16} {h1:>16} {hand_z:>12.3f} {int(d.ncon):>9}")

    print("-" * 78)
    # 1回目と2回目以降で「reset 直後の姿勢」が同じか
    same0 = len({r[0] for r in rows}) == 1
    print(f"  reset直後の姿勢が全回同じ : {'はい' if same0 else 'いいえ（回ごとに違う）'}")
    if not same0:
        print("    注意種なし reset は姿勢を揺らす（jitter）ので、これは異常ではない。"
              "大事なのは**2回実行したときに同じ列が出るか**")
    # 注意：ここで Python の組み込み `hash()` を使ってはいけない。
    #   文字列のハッシュは**起動ごとに変わる**（PYTHONHASHSEED）ので、
    #   中身が同じでも要約が違い、「非決定的だ」と誤診する。
    #   2026-07-30 に実際にこれで誤診した（落とし穴 項81の同型・同じ日に2回目）。
    #   → 比べるものは必ず**中身から**作る（sha1 など、実行に依らないもの）。
    print(f"\n  この実行の要約（2回実行して**この行だけ**比べる）")
    print(f"    {_h([int(r[0], 16) for r in rows] + [r[2] for r in rows])}")
    print(f"    詳細: " + " ".join(f"{r[0]}/{r[1]}" for r in rows))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
