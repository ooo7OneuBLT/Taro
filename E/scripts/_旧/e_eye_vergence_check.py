"""左右の眼球が「共同して動く」のか「寄り目になる」のかを切り分ける。

【なぜ要るか、2026-07-28】ユーザーの目視：
> 完璧じゃないけど、**寄り目になるようになった気もする**
> けど**右目の反射の効きが一部悪い**（動いてないわけではなさそう）

太郎の実装は左右のアクチュエータに**同じ指令**を書く（Hering の等神経支配の法則）。
＝設計上は寄り目にならないはず。にもかかわらず寄り目に見えるなら、
指令より後（VOR・筋の特性・可動域）で左右差が生まれている。

【切り分けの考え方】
    左目の角度 と 右目の角度 を並べて、
      差が一定             → 共同運動（オフセットがあるだけ）
      差が時間で変わる     → 何かが左右を別々に動かしている
    さらに「本来どうあるべきか」＝**それぞれの目から見たおもちゃの方向**も計算する。
    近いものを見るには左右で違う角度が要る（＝これが輻輳）。
      必要な角度の差 と 実際の角度の差 を比べれば、寄り目が足りているか分かる。

使い方:
    .venv/Scripts/python.exe E/scripts/e_eye_vergence_check.py
    E_TOY_POS=0.33,0.005,0.25 で位置を指定（既定は viewer_saved.json の保存値）
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import json
import warnings
import contextlib
import io

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "taro_core", "src", "brain"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

AGE = float(os.environ.get("E_AGE", "4.0"))
SECONDS = float(os.environ.get("E_SECONDS", "12.0"))
SAVE_PATH = os.path.join(_HERE, os.pardir, "docs", "viewer_saved.json")


def _saved_toy_pos():
    if os.environ.get("E_TOY_POS"):
        return np.array([float(x) for x in os.environ["E_TOY_POS"].split(",")])
    try:
        with open(SAVE_PATH, encoding="utf-8") as fp:
            s = json.load(fp)
        return np.array(s["toy_pos"], dtype=float), s
    except Exception:
        return None, {}


def eye_target_angles(m, d, toy_pos):
    """左右それぞれの目から見た、おもちゃの方向（水平・垂直）[度]。

    これが「本来向くべき角度」。左右で違うのが正常で、その差が輻輳。
    """
    out = {}
    for nm in ("eye_left", "eye_right"):
        cid = int(m.camera(nm).id)
        eye = np.array(d.cam_xpos[cid], dtype=float)
        R = np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)
        right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
        v = np.array(toy_pos, dtype=float) - eye
        n = np.linalg.norm(v)
        if n < 1e-9:
            out[nm] = (0.0, 0.0)
            continue
        v = v / n
        out[nm] = (float(np.degrees(np.arctan2(np.dot(v, right), np.dot(v, fwd)))),
                   float(np.degrees(np.arctan2(np.dot(v, up), np.dot(v, fwd)))))
    return out


def main():
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    os.environ.setdefault("E_FENCE", "0")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    from e_head_hold import CaregiverHands

    pos, saved = _saved_toy_pos()
    if pos is None:
        print("注意保存が読めない。E_TOY_POS で位置を指定してください")
        return
    if "toy_half_size" in saved and "E_TOY_RADIUS" not in os.environ:
        os.environ["E_TOY_RADIUS"] = str(saved["toy_half_size"])

    kw = body_kwargs_from_env(AGE, verbose=False)
    kw["flexion"] = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = ToySupineEnv(actuation_model=MuscleModel,
                           vision_params=infant_vision_params(acuity_age=AGE),
                           age=AGE, toy=True, vor=True, orient=True, **kw)
        env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    hands = CaregiverHands(m, d)
    hands.hold()
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)

    # 眼球の関節
    eyeq = {}
    for j in range(m.njnt):
        nm = m.joint(j).name
        for side in ("left", "right"):
            for ax in ("horizontal", "vertical"):
                if nm.endswith(f"{side}_eye_{ax}"):
                    eyeq[(side, ax)] = int(m.jnt_qposadr[j])

    # おもちゃを保存された位置へ置いて固定する
    toy_bid = int(m.body("test_object1").id)
    toy_jid = int(m.body_jntadr[toy_bid])
    toy_qadr = int(m.jnt_qposadr[toy_jid])
    toy_dadr = int(m.jnt_dofadr[toy_jid])
    toy_gadr = int(m.body("test_object1").geomadr[0])
    reflex = u._orienting
    reflex.reset()
    # 登場の演出を飛ばす
    for _ in range(int(0.3 / dt)):
        d.qpos[toy_qadr:toy_qadr + 3] = pos
        d.qvel[toy_dadr:toy_dadr + 6] = 0.0
        env.step(a)

    ipd = float(np.linalg.norm(
        np.array(d.cam_xpos[int(m.camera("eye_left").id)])
        - np.array(d.cam_xpos[int(m.camera("eye_right").id)])))
    eyes_mid = 0.5 * (np.array(d.cam_xpos[int(m.camera("eye_left").id)])
                      + np.array(d.cam_xpos[int(m.camera("eye_right").id)]))
    dist = float(np.linalg.norm(pos - eyes_mid))

    print("=" * 78)
    print(" 左右の眼球：共同運動か、寄り目か（2026-07-28）")
    print("=" * 78)
    print(f"  おもちゃの位置 {pos}   目からの距離 {dist*100:.1f}cm")
    print(f"  瞳孔間距離 {ipd*100:.2f}cm")
    print(f"  → 両目で捉えるのに要る寄り目 "
          f"{2*np.degrees(np.arctan2(ipd/2, dist)):.1f}度")
    print(f"  体年齢 {AGE:g}ヶ月  頭は抑えている")

    base_rgba = m.geom_rgba[toy_gadr].copy()
    dim_rgba = base_rgba.copy(); dim_rgba[:3] *= 0.25
    rows = []
    n = int(SECONDS / dt)
    for k in range(n):
        m.geom_rgba[toy_gadr] = base_rgba if (np.sin(2*np.pi*2.5*k*dt) >= 0) else dim_rgba
        d.qpos[toy_qadr:toy_qadr + 3] = pos          # おもちゃは動かさない
        d.qvel[toy_dadr:toy_dadr + 6] = 0.0
        env.step(a)
        if k % max(1, int(0.5 / dt)):
            continue
        tgt = eye_target_angles(m, d, pos)
        rows.append(dict(
            t=k * dt,
            lh=float(np.degrees(d.qpos[eyeq[("left", "horizontal")]])),
            rh=float(np.degrees(d.qpos[eyeq[("right", "horizontal")]])),
            lv=float(np.degrees(d.qpos[eyeq[("left", "vertical")]])),
            rv=float(np.degrees(d.qpos[eyeq[("right", "vertical")]])),
            tlh=tgt["eye_left"][0], trh=tgt["eye_right"][0],
            tlv=tgt["eye_left"][1], trv=tgt["eye_right"][1],
            sacc=int(reflex.n_saccades)))

    print("\n" + "=" * 78)
    print(" 水平（左右を見る）")
    print("=" * 78)
    print(f"{'t[s]':>6}{'左目':>9}{'右目':>9}{'左-右':>9}   |"
          f"{'左の目標':>10}{'右の目標':>10}{'目標の差':>10}   |{'残るずれ':>10}")
    print(f"{'':>6}{'実際の角度':>27}   |{'おもちゃの方向':>30}   |{'左':>5}{'右':>5}")
    for r in rows:
        print(f"{r['t']:>6.1f}{r['lh']:>9.2f}{r['rh']:>9.2f}{r['lh']-r['rh']:>9.2f}   |"
              f"{r['tlh']:>10.2f}{r['trh']:>10.2f}{r['tlh']-r['trh']:>10.2f}   |"
              f"{r['tlh']-r['lh']:>5.1f}{r['trh']-r['rh']:>5.1f}")

    print("\n" + "=" * 78)
    print(" まとめ")
    print("=" * 78)
    d_act = [r["lh"] - r["rh"] for r in rows]
    d_tgt = [r["tlh"] - r["trh"] for r in rows]
    el = [abs(r["tlh"] - r["lh"]) for r in rows]
    er = [abs(r["trh"] - r["rh"]) for r in rows]
    print(f"  実際の左右差   最小 {min(d_act):+.2f}  最大 {max(d_act):+.2f}  "
          f"変化幅 {max(d_act)-min(d_act):.2f}度")
    print(f"  必要な左右差   最小 {min(d_tgt):+.2f}  最大 {max(d_tgt):+.2f}  "
          f"（＝輻輳。近いほど大きい）")
    print(f"  残るずれ       左目 平均 {np.mean(el):.2f}度  右目 平均 {np.mean(er):.2f}度")
    print(f"  サッケード     {rows[-1]['sacc']}発")
    print("\n  読み方")
    print("  ・実際の左右差の『変化幅』が小さい → 共同運動（寄り目はしていない）")
    print("  ・必要な左右差が大きいのに実際が小さい → 輻輳が足りない")
    print("  ・左右で『残るずれ』が違う → 片方だけ合っている＝両目で同じものを見ていない")
    env.close()


if __name__ == "__main__":
    main()
