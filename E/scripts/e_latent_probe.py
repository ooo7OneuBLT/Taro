"""【段階1のモデルは段階2に使えるか】潜在変数 z に視覚の情報が残っているかを測る。

【なぜ測るか（ユーザーの指摘）】
段階1は「視覚を**入力**には入れたが、**予測対象**には入れない」状態で自己モデルを学習した。
ニューラルネットは**使わない情報を捨てる**傾向があるので、内部表現 z が視覚を
無視するように育っている可能性がある（固有感覚だけを予測するなら、そのほうが精度が上がる）。
そうなっていると、段階2で視覚を予測しようにも**材料が z に残っていない**。

【配線チェックとの違い】
配線チェックで確認したのは「融合ベクトル sv に視覚が効くか」まで。
その先の
    融合ベクトル sv → GRU → pc_latent → z
で消えていないかは**未確認**だった。ここを測る。

【測り方】
同じ物理状態で、**視覚だけを変えた** obs を作り、z がどれだけ変わるかを見る。
 (a) 手を消す（geomのα=0）＝段階2で区別したい当の違い
 (b) 画面を暗くする（全画素0）＝視覚が完全に変わる極端な条件
比較のため、固有感覚を揺らしたときの z の変化も測る（＝z が何にどれだけ反応するかの物差し）。

【判定】
 手を消したときの z の変化が、固有感覚を揺らしたときと比べて無視できるほど小さいなら、
 z は視覚を捨てている → 段階1のモデルは段階2に使えない（C5から視覚込みでやり直す）。

使い方: python e_latent_probe.py [n_samples]
  環境変数 C5_CKPT で測るモデルを指定
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "D", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
import paths  # noqa: E402
paths.setup_brain_path()
sys.path.insert(0, paths.MIMO_DIR)

import mimoEnv  # noqa: F401,E402
import mujoco  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402
from e_hand_in_view import hand_in_view  # noqa: E402
from e_encoder_probe import hand_geoms  # noqa: E402


def get_z(brain, fusion, emb_proj, obs, prev_a, hidden, rng_seed=None):
    """その観測から内部表現 z を作る（学習ループと同じ手順）。

    ⚠️【重要・2026-07-20】`pc_latent.infer` は**確率的**：
      z = mean + randn * std   ＋ その場の誤差回帰の中でも randn を使う
    ので、**同じ入力でも呼ぶたびに違う z を返す**。実測すると、入力を変えていないのに
    z の 131% も変わった（＝「視覚を変えたときの差125%」と同じ大きさ）。
    第1版はこれに気づかず「視覚は z に残っている」と誤判定した。
    → **rng_seed を渡して乱数を固定**し、2つの入力で同じ乱数列を使う。
      こうすると差は「入力の違い」だけになる。
    （この確認はチェックリスト項目6「測定器そのものの健康診断」に書いてあった＝読み落とし）
    """
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    sv = fusion.encode(obs)
    cf = fusion.encode(obs).detach()
    emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
    out, _ = brain.motor_gru(emb, hidden)
    z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
    return z.detach()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    half_fov = te.VISION_FOVY / 2.0
    hg = hand_geoms(m)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    rows = []
    tick = 0
    print(f"\n手が視野に入った場面を {n} 個集めて、z の変化を測ります…")

    while len(rows) < n and tick < 20000:
        tick += 1
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        inview = bool(hand_in_view(m, d))
        if inview:
            rs = 12345 + len(rows)   # この場面で使う乱数列（4条件で共通にする）
            z0 = get_z(brain, fusion, emb_proj, obs, prev_a, hidden, rs)
            scale = float(z0.abs().mean())

            # (a) 手だけ消した観測
            saved = m.geom_rgba[hg].copy()
            m.geom_rgba[hg, 3] = 0.0
            raw._vision_cache = None
            v_nohand = raw.get_vision_obs()
            m.geom_rgba[hg] = saved
            o_a = dict(obs); o_a["eye_left"] = v_nohand["eye_left"]; o_a["eye_right"] = v_nohand["eye_right"]
            z_a = get_z(brain, fusion, emb_proj, o_a, prev_a, hidden, rs)

            # (b) 画面を真っ黒にした観測（視覚が完全に変わる極端な条件）
            o_b = dict(obs)
            o_b["eye_left"] = np.zeros_like(np.asarray(obs["eye_left"]))
            o_b["eye_right"] = np.zeros_like(np.asarray(obs["eye_right"]))
            z_b = get_z(brain, fusion, emb_proj, o_b, prev_a, hidden, rs)

            # (c) 比較用：固有感覚を1.0ずらしたとき（z が何にどれだけ反応するかの物差し）
            o_c = dict(obs)
            o_c["observation"] = np.asarray(obs["observation"], dtype=float) + 1.0
            z_c = get_z(brain, fusion, emb_proj, o_c, prev_a, hidden, rs)

            rows.append((float((z_a - z0).abs().mean()),
                         float((z_b - z0).abs().mean()),
                         float((z_c - z0).abs().mean()), scale))
            raw._vision_cache = None
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    if not rows:
        print("手が視野に入る場面を集められませんでした")
        env.close(); return
    R = np.asarray(rows)
    ck = os.path.basename(os.environ.get("C5_CKPT", "?"))
    print(f"\n=== z に視覚は残っているか（{ck}／{len(rows)}場面／{tick}tick走査）===")
    print(f"  z 自体の大きさ             : {R[:,3].mean():.5f}")
    print(f"  (a) 手を消したときの z の変化 : {R[:,0].mean():.5f}"
          f"   （z の {100*R[:,0].mean()/max(R[:,3].mean(),1e-9):.2f}%）")
    print(f"  (b) 画面を真っ黒にしたとき   : {R[:,1].mean():.5f}"
          f"   （z の {100*R[:,1].mean()/max(R[:,3].mean(),1e-9):.2f}%）")
    print(f"  (c) 固有感覚を1.0ずらしたとき : {R[:,2].mean():.5f}"
          f"   （z の {100*R[:,2].mean()/max(R[:,3].mean(),1e-9):.2f}%）＝物差し")

    print("\n=== 判定 ===")
    ra = R[:, 0].mean() / max(R[:, 2].mean(), 1e-12)
    rb = R[:, 1].mean() / max(R[:, 2].mean(), 1e-12)
    print(f"  手の有無 / 固有感覚 の比 = {ra:.4f}")
    print(f"  視覚全体 / 固有感覚 の比 = {rb:.4f}")
    if rb < 0.01:
        print("  → z は視覚をほぼ捨てている。段階1のモデルは段階2に使えない")
        print("     （C5から『視覚も予測対象』でやり直すべき）")
    elif ra < 0.01:
        print("  → 視覚全体は残っているが、手の有無は z に届いていない")
        print("     （段階2で予測しようにも材料が薄い。要検討）")
    else:
        print("  → z に視覚も手の有無も残っている。段階1のモデルを段階2に使える")
    env.close()


if __name__ == "__main__":
    main()
