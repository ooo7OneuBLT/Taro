"""【E1の指標】太郎の手が「両目の視野内」にある割合を測る＝hand regard の測定尺度。

【なぜこの指標か】
E1の問いは「**新しい本能を足さずに、既存のprogress報酬だけで『自分の手を見る』行動が
創発するか**」。人間側の一次文献の判定は**創発**（White 1966：環境操作で出現日が46日↔66日と
動く＝反射ではない。[参考文献リスト §目標E-15](../../doc/参考文献リスト.md)）。
したがって太郎に「手を見る本能」は入れず、**行動が増えるかどうかを外から測る**しかない。
その測定尺度がこれ。

【測り方】
各tickで、左右それぞれの手の中心が**左目・右目の両方のカメラ画角に入っているか**を幾何で判定する。
 ・視線＝カメラのz軸の逆（MuJoCoのカメラは-z方向を見る）
 ・画角の半角＝fovy/2（正方形レンダなので水平も同じ）
 ・**両目とも**に入っている場合を「視野内」とする（片目だけなら人間でも両眼視は成立しない）
⚠️**遮蔽（体や柵に隠れて実際には見えない）は考慮していない**＝これは幾何的な上限値。
⚠️手のbody中心だけで判定＝指先が入っていても中心が外なら「外」になる（過小評価側の誤差）。

【この指標をどう使うか】
 (1) **学習前のベースライン**（このスクリプトの用途）＝偶然どれだけ入るか
 (2) 学習後に同じ指標を測り、**有意に増えていれば「手を見ようとしている」**
 (3) 環境の豊かさ（おもちゃ・柵）を変えた二条件で比較＝Whiteの追試

使い方:
  python e_hand_in_view.py [n_ticks]
  環境変数 E_TOY_OBJ=0 でおもちゃなし / E_FENCE=0 で柵なし（＝「貧しい環境」）
"""
import os
import sys
import warnings

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
import torch  # noqa: E402
import d_c5_motor_quality as mq  # noqa: E402
import e_toy_env as te  # noqa: E402

EYES = ("eye_left", "eye_right")
HANDS = ("left_hand", "right_hand")


def eye_angles(model, data, target_pos):
    """各眼から見て、target_pos が視線から何度ずれているか（度）。"""
    out = {}
    for cam in EYES:
        cid = int(model.camera(cam).id)
        R = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)
        gaze = -R[:, 2]                      # MuJoCoのカメラは-z方向を見る
        v = np.asarray(target_pos, dtype=float) - np.array(data.cam_xpos[cid], dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-9:
            out[cam] = 0.0
            continue
        c = float(np.dot(gaze, v) / n)
        out[cam] = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    half_fov = te.VISION_FOVY / 2.0
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    policy = mq.make_policy(brain, fusion, emb_proj, cereb, n_act, babble=True)
    raw = env.unwrapped
    m, d = raw.model, raw.data

    rich = []
    if getattr(raw, "_toy", False):
        rich.append("おもちゃ")
    if getattr(raw, "_fence", False):
        rich.append("柵")
    print(f"\n=== 環境: {'／'.join(rich) if rich else '★何もない（貧しい環境）'}"
          f"  視野の半角 {half_fov:.0f}° ===")

    torch.manual_seed(seed); np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    rec = {h: {"both": [], "any": [], "ang": []} for h in HANDS}
    dist = {h: [] for h in HANDS}

    for t in range(n):
        a, hidden = policy(obs, prev_a, hidden)
        ctrl = mq.rescale_action(a, env.action_space); prev_a = a
        for k in range(mq.K):
            obs, r, term, trunc, info = env.step(ctrl)
            if term or trunc:
                break
        for h in HANDS:
            p = np.array(d.body(h).xpos, dtype=float)
            ang = eye_angles(m, d, p)
            inside = [ang[c] <= half_fov for c in EYES]
            rec[h]["both"].append(all(inside))
            rec[h]["any"].append(any(inside))
            rec[h]["ang"].append(min(ang.values()))
            cid = int(m.camera("eye_left").id)
            dist[h].append(float(np.linalg.norm(p - np.array(d.cam_xpos[cid]))))
        if term or trunc:
            obs, _ = env.reset()
            hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)

    print(f"\n--- {n} tick の実測（学習前のベースライン） ---")
    for h in HANDS:
        b = np.asarray(rec[h]["both"]); an = np.asarray(rec[h]["any"])
        ag = np.asarray(rec[h]["ang"]); ds = np.asarray(dist[h])
        print(f"  {h:11s} 両目の視野内 {b.mean()*100:5.1f}%   片目でも {an.mean()*100:5.1f}%"
              f"   視線からのずれ mean {ag.mean():5.1f}° / min {ag.min():5.1f}°"
              f"   目からの距離 mean {ds.mean()*100:4.1f}cm")
    both_any = np.asarray(rec["left_hand"]["both"]) | np.asarray(rec["right_hand"]["both"])
    print(f"  ★どちらかの手が両目の視野内: {both_any.mean()*100:.1f}% のtick")
    # 目からの距離が「見える距離」かも併記（新生児は中央値19cmにしかピントが合わない）
    allds = np.concatenate([np.asarray(dist[h]) for h in HANDS])
    print(f"  参考: 手と目の距離 全体 mean {allds.mean()*100:.1f}cm / max {allds.max()*100:.1f}cm"
          f"   （新生児がピントを合わせられるのは中央値19cm＝Haynes et al. 1965）")
    env.close()


if __name__ == "__main__":
    main()
