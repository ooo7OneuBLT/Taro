"""【配線チェック】各感覚・機構が"繋いだつもり"で終わっていないかを機械的に検証する。

【なぜ作るか＝同じ失敗を3回した】
2026-07-20 の1日だけで、目視で偶然見つかった「動いていなかった」が3件あった：
  ①**視覚・触覚が脳に繋がっていなかった**（`build()` が `vision_params=None, touch_dim=0` のまま。
    環境側にパラメータを用意しただけで渡しておらず、「視覚を入力に足した」という報告は誤りだった）
  ②**視力フィルタが無効だった**（`acuity=0.0` は MIMo の truthy 判定で弾かれる。
    別のバグを避けるつもりで機能ごと殺していた）
  ③**柵の色指定が効いていなかった**（MjSpecのデフォルト材質 matgeom を継承して茶色に描かれていた）
いずれも「設定した値がそのまま効いている」と思い込んだために起きた。
→ **入力を揺らして出力が動くか**を毎回確かめる。動かなければ配線が死んでいる。

【検証項目】
 1. 融合ベクトルに各感覚が実際に効いているか（その感覚だけ変えて出力が動くか）
 2. 視力フィルタ(acuity)が効いているか（かける前後で画像が変わるか）
 3. 視覚obsが一人称か（第三者視点になっていないか＝MIMo既知のバグ）
 4. 左右の眼が別の画を見ているか（両方とも同じカメラを見ていないか）
 5. VORが眼球アクチュエータを実際に上書きしているか
 6. 視覚の間引き(VISION_MIN_DT)が意図どおり働いているか
 7. 柵・床・空の色が実際に一致しているか
 8. 予測対象(nat_head)に何が入っているか＝視覚が予測対象に入っているか

使い方: python e_wiring_check.py
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

OK, NG, WARN = "  [OK]  ", "  [NG]  ", "  [warn]"
results = []


def check(name, passed, detail):
    results.append((name, passed))
    print(f"{OK if passed else NG}{name}: {detail}")


def main():
    env, brain, fusion, emb_proj, cereb, n_act = mq.build("off", age=0)
    raw = env.unwrapped
    m, d = raw.model, raw.data
    obs, _ = env.reset(seed=0)
    print("\n=== 配線チェック ===")

    # --- 1. 各感覚が融合ベクトルに効いているか -------------------------------
    base = fusion.encode(obs).detach().clone()
    for key in ("observation", "touch", "vestibular", "interoception"):
        if key not in obs:
            check(f"感覚 {key}", False, "obsに存在しない")
            continue
        o2 = {k: (np.asarray(v).copy() if not torch.is_tensor(v) else v.clone())
              for k, v in obs.items()}
        o2[key] = np.asarray(o2[key], dtype=float) + 1.0     # その感覚だけ大きく揺らす
        try:
            diff = float((fusion.encode(o2) - base).abs().max())
        except Exception as e:
            check(f"感覚 {key}", False, f"encodeが失敗 {type(e).__name__}")
            continue
        check(f"感覚 {key} が融合に効く", diff > 1e-6, f"揺らしたときの出力変化 {diff:.6f}")
    if "eye_left" in obs:
        o2 = dict(obs)
        o2["eye_left"] = np.zeros_like(np.asarray(obs["eye_left"]))
        diff = float((fusion.encode(o2) - base).abs().max())
        check("感覚 vision が融合に効く", diff > 1e-6, f"左目を黒画像にしたときの変化 {diff:.6f}")
    else:
        check("感覚 vision が融合に効く", False, "obsに eye_left が無い＝視覚が繋がっていない")

    # ★測定の前提：視界が**単色**だと「ぼかしても変わらない／左右も同じ」になり、
    #   配線が生きていても NG と出る（偽陰性）。実際に一度これで誤判定した。
    #   → 構造（柵など）が視野に入る向きへ頭を回してから測る。
    import mujoco

    def _variance():
        """視界の**空間的な**分散。⚠️RGBのチャンネル間の差を拾ってはいけない。

        第1版は画像全体の std を見ていたが、視界が単色 PLAIN_RGBA(140,158,178) で
        埋まっていても std=15.5 になる（＝RGBの3値の散らばり）ので「構造あり」と
        誤判定した。その結果、単色の画で「ぼかしても変わらない」「左右で同じ」
        「頭を回しても変わらない」という3つの偽NGを出した。
        → **グレースケール化して空間方向の分散**を見る。
        """
        raw._vision_cache = None
        img = np.asarray(raw.get_vision_obs()["eye_left"], float)
        return float(img.mean(axis=2).std())

    _head_j = [j for j in range(m.njnt) if "head" in m.joint(j).name]
    if _variance() < 3.0 and _head_j:
        for ang in (0.5, 1.0, -0.5, -1.0, 1.4, -1.4):
            for j in _head_j:
                d.qpos[m.jnt_qposadr[j]] = ang
            mujoco.mj_forward(m, d)
            if _variance() >= 3.0:
                break
    _v = _variance()
    print(f"  （視界の空間分散 {_v:.1f}"
          f"{'＝構造が写っている状態で測定' if _v >= 3.0 else ' ★単色のまま＝以下の視覚テストは判定不能'}）")

    # --- 2. 視力フィルタが効いているか ---------------------------------------
    if raw.vision is not None:
        af = getattr(raw.vision, "_acuity_functions", {}).get("eye_left")
        if af is None:
            check("視力フィルタ(acuity)", False, "★_acuity_functions が None＝フィルタ無効")
        else:
            # ★実際のパイプライン同士で比べる：生の描画 vs 太郎が受け取るobs（acuity適用済み）
            ren = mujoco.Renderer(m, height=te.VISION_RES, width=te.VISION_RES)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = int(m.camera("eye_left").id)
            m.cam_fovy[cam.fixedcamid] = te.VISION_FOVY
            ren.update_scene(d, camera=cam)
            plain = ren.render().astype(float)
            raw._vision_cache = None
            taro = np.asarray(raw.get_vision_obs()["eye_left"], float)
            dd = float(np.abs(plain - taro).mean())
            check("視力フィルタ(acuity)", dd > 0.5,
                  f"生の描画 vs 太郎の入力 の平均画素差 {dd:.2f}（MTF平均={float(np.mean(af)):.3f}）")
            # ⚠️ここで ren.close() を呼んではいけない。MuJoCoのRendererはOpenGLコンテキストを
            #   共有しており、片方を閉じると**以降の描画がすべて同じ画を返す**（実際、これで
            #   「左右の眼が同じ」「頭を回しても変わらない」という2つの偽NGが出た。
            #   同じ手順を単独で実行すると左右差26.5で正常だったことから切り分けた）。
            #   プロセス終了時にまとめて解放されるので明示的なcloseは不要。
    else:
        check("視力フィルタ(acuity)", False, "vision モジュールが None")

    # --- 3-4. 視覚obsが一人称か／左右で違うか --------------------------------
    raw._vision_cache = None
    vis = raw.get_vision_obs()
    l, r = np.asarray(vis["eye_left"], float), np.asarray(vis["eye_right"], float)
    check("左右の眼が別の画を見ている", float(np.abs(l - r).mean()) > 0.5,
          f"左右の平均画素差 {float(np.abs(l - r).mean()):.2f}")
    # 一人称かの判定：頭を大きく回して視覚が変わるか（第三者視点なら変化が小さい）
    jid = None
    for j in range(m.njnt):
        if "head" in m.joint(j).name and "tilt" not in m.joint(j).name:
            jid = j; break
    if jid is not None:
        q0 = float(d.qpos[m.jnt_qposadr[jid]])
        d.qpos[m.jnt_qposadr[jid]] = q0 + 0.8
        mujoco.mj_forward(m, d)
        raw._vision_cache = None                 # キャッシュを捨てて撮り直させる
        v2 = np.asarray(raw.get_vision_obs()["eye_left"], float)
        d.qpos[m.jnt_qposadr[jid]] = q0
        mujoco.mj_forward(m, d); raw._vision_cache = None
        dd = float(np.abs(l - v2).mean())
        check("視覚が一人称（頭を回すと変わる）", dd > 1.0,
              f"頭を0.8rad回したときの平均画素差 {dd:.2f}")

    # --- 5. VORが眼球を上書きしているか --------------------------------------
    if getattr(raw, "_vor", None) is not None:
        act = np.zeros(n_act)
        d.qvel[:] = 0.0
        d.cvel[int(m.body("head").id)][:3] = [0.0, 1.0, 0.0]   # 頭を回している状況を作る
        out = raw._vor.override(act, m, d, raw.dt)
        eye_ids = [u["aid"] for u in raw._vor.units]
        moved = float(np.abs(np.asarray(out)[eye_ids]).max())
        other = float(np.abs(np.delete(np.asarray(out), eye_ids)).max())
        check("VORが眼球だけを上書き", moved > 1e-6 and other < 1e-12,
              f"眼球ch最大 {moved:.4f} / 眼球以外 {other:.2e}")
        d.cvel[int(m.body("head").id)][:3] = 0.0
    else:
        check("VOR", False, "VORが有効でない（E_VOR=0?）")

    # --- 6. 視覚の間引き ------------------------------------------------------
    raw._vision_cache = None
    a1 = np.asarray(raw.get_vision_obs()["eye_left"], float)
    t_before = float(raw._vision_t)
    a2 = np.asarray(raw.get_vision_obs()["eye_left"], float)   # 同じsim時刻＝キャッシュのはず
    check("視覚の間引きが働く", float(raw._vision_t) == t_before and np.array_equal(a1, a2),
          f"同一時刻の2回目がキャッシュ（更新周期 {te.VISION_MIN_DT}s）")

    # --- 7. 色の一致 ----------------------------------------------------------
    if getattr(raw, "_plain", False):
        fl = None; fe = None
        for i in range(m.ngeom):
            nm = m.geom(i).name or ""
            if nm == "floor":
                fl = m.geom_rgba[i].copy()
            elif nm.startswith("fence_post") and fe is None:
                fe = m.geom_rgba[i].copy()
        if fl is not None and fe is not None:
            check("床と柵が同色", bool(np.allclose(fl, fe)),
                  f"床 {np.round(fl,3)} / 柵 {np.round(fe,3)}")
        elif fl is not None:
            check("床の色", True, f"床 {np.round(fl,3)}（柵なし条件）")

    # --- 8. 予測対象に何が入っているか ---------------------------------------
    from fusion import to_tensor
    prop_dim = to_tensor(obs["observation"]).shape[0]
    sdim = fusion.encode(obs).shape[0]
    print(f"{WARN}予測対象: 固有感覚 {prop_dim}次元のみ（融合は{sdim}次元だが"
          f"**視覚・触覚は予測対象に入っていない**）＝E1で足す予定の未実装項目")

    n_ng = sum(1 for _, p in results if not p)
    print(f"\n=== 結果: {len(results)-n_ng}/{len(results)} 通過"
          f"{'  ★NGあり＝配線が死んでいる' if n_ng else '  すべて配線が生きている'} ===")
    env.close()


if __name__ == "__main__":
    main()
