"""視線誘導反射が「おもちゃを視野の中心へ寄せられるか」を測る（本番の判定）。

【なぜこれが本当の判定か】それまでは「1発で真の位置と一致するか」を見ていたが、
人間の新生児はそうならない。**第一サッケードは著しく低振幅で目標に届かず、
同じ振幅の追加サッケードが階段状に続く**（Aslin & Salapatek 1975）。
＝正しい判定は「繰り返して近づくか」。

【測り方】
  ・太郎は仰向けで脱力（action=0）。反射だけが目と首を動かす
  ・おもちゃを視野の端に置き、親が揺らしている想定で振動させる
    （文献の定位実験も出現・点滅・移動のいずれかの時間変化を伴う）
  ・毎ステップ、描き分け（segmentation）でおもちゃの真の重心を測る
  ・反射ON / OFF を同じ条件で比べる

【読み方】
  ずれが単調に減る               → 反射が働いている
  減らない・増える               → 働いていない（向きが逆なら符号の誤り）
  階段状に減る                   → 新生児らしい（1発では届かない）
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
import os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
          os.path.join(_ROOT, "taro_core"),
          os.path.join(_ROOT, "taro_core", "src", "body"),
          os.path.join(_ROOT, "run", "scene_tools"), _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)
try: sys.stdout.reconfigure(errors="replace")
except Exception: pass

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Meiryo", "Yu Gothic", "MS Gothic", "IPAexGothic"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

import e_visibility as VIS

# おもちゃの初期位置[m]。E_ORIENT_OFFSETS で振れる。
# 注意：新生児が定位できる範囲は方向で違う：水平・斜めは30度まで、
#   **垂直は10度まで**（Aslin & Salapatek 1975）。距離10.9cmなら
#   0.02m ≒ 10度、0.04m ≒ 20度。垂直を ±0.04 で測るのは能力の範囲外。
START_OFFSETS = [float(x) for x in
                 os.environ.get("E_ORIENT_OFFSETS", "-0.04,-0.02,0.02,0.04").split(",")]
BLINK_HZ = 2.5     # おもちゃを点滅させる周期[Hz]
SECONDS = float(os.environ.get("E_SECONDS", "6.0"))
OUT_DIR = os.path.join(_ROOT, "E", "logs", "orient_converge")
# どの軸で測るか。E_ORIENT_AXIS=v で上下方向。
#   水平の符号は実測で直したが、垂直は「水平と同じ向き」と仮定しただけだった。
AXIS = os.environ.get("E_ORIENT_AXIS", "h")
IS_V = (AXIS == "v")

# 【2026-07-28】体年齢を 0ヶ月 → 4ヶ月に変更した。
#
# 【なぜ4ヶ月か】この反射を実装した目的は**リーチング**（運動発達ロードマップ段階3、
# 人間年齢3〜5ヶ月・体年齢4m）の前提を満たすこと。`E/docs/全体設計_目標E.md` §2.5 が
# 「対象への視覚的注意・固視」を**リーチの唯一の強い一方向依存**（von Hofsten）と位置づけ、
# 視線誘導反射をそのボトルネックA としている。＝反射を検証する月齢は、それを使う月齢に
# 揃えるべき（ユーザーの判断、2026-07-28）。
#
# 注意：【引き受ける逸脱】4ヶ月の人間は**皮質（前頭眼野など）が定位に関与し始める**
# （Johnson 1990）が、太郎は上丘だけで「対象を選んで見続ける」注意機構を持たない。
# 設計図はこれを既知の穴とし「本能で足すか創発に委ねるかは要判断」と保留にしている。
# 4ヶ月で回すことで、設計の予測（注意が無いとリーチが空振りする）を実際に確かめられる。
#
# 注意：人間の視覚定位実験（Hunter & Richards 2003）の最年長群は**14週齢＝3.2ヶ月**なので、
# 4ヶ月はその近傍。比較は可能だが厳密に同月齢ではない。
AGE = float(os.environ.get("E_AGE", "4.0"))

# 【2026-07-29】既定のシーン。E_SCENE=名前 で切り替えられる。
#   環境の条件（月齢・柵・リクライニング・頭の支え・おもちゃ）はシーンが持つので、
#   ここから下の AGE / HEAD_HOLD は**シーンの値で上書きされる**（表示用に残している）。
DEFAULT_SCENE = os.environ.get("E_SCENE", "視線誘導反射_仰向け_頭を支える")

# 実験中は**実験者が頭を抑える**（既定ON）。
# 人間の乳児実験でも実験者が支えている（Hunter & Richards 2003：5週齢は頭の両側に枕、
# 8〜14週齢は親が手で頭を抑える）。太郎の首が倒れて対象を捉え続けられないのは、
# 「首がすわっていない」からではなく「支えていない」からだった。
# `E/docs/全体設計_目標E.md` も E1 では「外部支持でも可」と明記している。
# 詳細と実装は `E/scripts/e_head_hold.py`。
HEAD_HOLD = os.environ.get("E_HEAD_HOLD", "1") == "1"

# 【2026-07-28】シードを振れるようにした。
#
# 【なぜ要るか】2026-07-27 に「高速化で悪化した（0.154→0.490）」と判断したが、
# 実際は神経ノイズのシードが毎回変わっていただけで、同じ条件3回で
# 0.319 / 0.215 / 0.189 とばらついていた（＝ばらつきの幅0.13）。
# 今日の改善幅（0.175→0.076＝0.10）は**このばらつきより小さい可能性がある**。
# 1シードの結果だけでは本物かどうか言えないので、複数シードで確かめる。
#
# 注意：2つのシードを揃える必要がある：
#     env.reset(seed=)      … 物理の初期ゆらぎ
#     E_ORIENT_SEED         … 反射の神経ノイズ（e_orienting_v2.py）
SEED = int(os.environ.get("E_SEED", "0"))
os.environ.setdefault("E_ORIENT_SEED", str(SEED))


def run(orient_on, off, env, u, m, d, dt, toy_gadr, right, toy_bid, n_steps,
        hands=None, scene=None):
    """1条件を走らせ、時系列（時刻・ずれ・眼球角度・サッケード数）を返す。"""
    import e_scene
    # 【2026-07-29】`env.reset()` ではなくシーンへ戻す。
    #   reset だけだと**シーンの姿勢が失われ**、2条件目からは既定の姿勢で
    #   走ってしまう（Viewer と測定の食い違いと同じ型）。
    e_scene.reset_to_scene(env, scene, hands=(hands if HEAD_HOLD else None),
                           seed=SEED)
    reflex = u._orienting
    if reflex is not None:
        reflex.reset()
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    ts, errs, eyes, sacc = [], [], [], []
    cmds, tgts, hdirs = [], [], []
    snap = {}   # 途中の一枚（生の画像・動きマップ）を診断用に取っておく

    # おもちゃを**シーンの位置から** off だけ横（または上下）へずらす。
    #   ＝「視野の端に置いて、反射が中心へ寄せられるか」を測るための条件。
    #   注意：移行前は「親が運んでくる途中で行き先をずらす」実装だったが、
    #     シーンでは位置が最初から決まっているので運搬を待つ必要がない。
    #     `place_toy` は固定位置（`_rest_pos`）ごと動かすので、次の step で
    #     元の場所へ引き戻されることもない。
    e_scene.place_toy(env, right * off, relative=True)
    for _ in range(int(0.2 / dt)):
        env.step(a)                       # 落ち着かせる

    # 動きを作る手段は「点滅」。おもちゃを物理的に動かすと顔と当たって壊れる。
    #   文献の定位実験も点滅光を使い（Lewis & Maurer 系）、上丘のニューロンは
    #   静止した点滅ドットにも動く刺激とほぼ同等に応答する（J Neurophysiol 2004）。
    base_rgba = m.geom_rgba[toy_gadr].copy()
    dim_rgba = base_rgba.copy()
    dim_rgba[:3] *= 0.25
    every = max(1, int(round(0.02 / dt)))
    for i in range(n_steps):
        phase = (np.sin(2 * np.pi * BLINK_HZ * i * dt) >= 0)
        m.geom_rgba[toy_gadr] = base_rgba if phase else dim_rgba
        env.step(a)
        if i % every:
            continue
        if orient_on and not snap and i >= n_steps // 2:
            _im = u.get_vision_obs()
            snap["rgb"] = (np.asarray(_im.get("eye_left"))
                           if isinstance(_im, dict) and _im.get("eye_left") is not None
                           else None)
            snap["motion"] = (np.asarray(reflex.motion_map).copy()
                              if (reflex is not None and reflex.motion_map is not None)
                              else None)
            snap["dir"] = (reflex.v_dir if IS_V else reflex.h_dir) if reflex else 0.0
        sv = VIS.visible_by_segment(m, d, toy_bid, "eye_left", size=64)
        ts.append(i * dt)
        # 画像の y は下向きなので、上下方向は符号を反転して「上が正」に揃える
        if sv["seen"]:
            errs.append(-sv["cy"] if IS_V else sv["cx"])
        else:
            errs.append(float("nan"))
        eyes.append(reflex._angle_deg(reflex.eye_qadr["v" if IS_V else "h"])
                    if (reflex is not None and orient_on) else 0.0)
        sacc.append(reflex.n_saccades if (reflex is not None and orient_on) else 0)
        if reflex is not None and orient_on:
            cmds.append(float(getattr(reflex, "last_eye_cmd", 0.0)))
            tgts.append(float(reflex._tgt["eye_v" if IS_V else "eye_h"]))
            hdirs.append(float(reflex.v_dir if IS_V else reflex.h_dir))
        else:
            cmds.append(0.0); tgts.append(0.0); hdirs.append(0.0)
    m.geom_rgba[toy_gadr] = base_rgba
    return (np.array(ts), np.array(errs), np.array(eyes), np.array(sacc),
            np.array(cmds), np.array(tgts), np.array(hdirs), snap)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ.setdefault("E_ORIENT_V", "2")
    os.environ.setdefault("E_TOY_MODE", "hold")
    os.environ.setdefault("E_TOY_SHAPE", "sphere")
    os.environ.setdefault("E_TOY_RADIUS", "0.0056")
    from e_toy_env import ToySupineEnv, infant_vision_params
    from mimoActuation.muscle import MuscleModel
    from e_body_config import body_kwargs_from_env
    import e_orienting_v2 as OR

    from e_head_hold import CaregiverHands

    results = {}
    # 【2026-07-29】シーン方式へ移行した。
    #   環境の条件（月齢・柵・リクライニング・頭の支え・おもちゃ）は
    #   **シーンファイルが全部持つ**ので、ここでは組み立てない。
    #   ⇒ Viewer で見た状態と、この測定が走る状態が構造的に同じになる。
    #   注意：移行前は `fence` を指定しておらず既定（柵あり）で走っていたが、
    #     Viewer 側は柵なしで見ていた＝**実際に食い違っていた**。
    import e_scene
    global AGE, HEAD_HOLD
    scene = e_scene.from_env_var(default=DEFAULT_SCENE)
    AGE = float(scene["body"]["age_months"])
    HEAD_HOLD = bool(scene["setup"].get("head_hold"))
    print(f"[scene] 「{scene['name']}」で測ります")
    if scene.get("note"):
        print(f"        {scene['note']}")
    for orient_on in (True, False):
        env, hands = e_scene.build(scene, orient=orient_on, vor=True,
                                   seed=SEED, verbose=False)
        u = env.unwrapped
        m, d = u.model, u.data
        # 保存した状態と本当に同じ環境かを確かめる。違ったらここで止まる
        #   ＝「見ていたのとは別の実験」を黙って走らせない
        e_scene.verify(scene, env, strict=True, verbose=orient_on)
        e_scene.reset_to_scene(env, scene, hands=hands, seed=SEED)
        dt = float(m.opt.timestep) * int(u.frame_skip)
        n_steps = int(SECONDS / dt)
        toy_bid = int(m.body("test_object1").id)
        toy_gadr = int(m.body("test_object1").geomadr[0])
        cam_id = int(m.camera("eye_left").id)
        eye = np.array(d.cam_xpos[cam_id], dtype=float)
        R = np.array(d.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        right, up, fwd = R[:, 0], R[:, 1], -R[:, 2]
        shift = up if IS_V else right
        # 注意：おもちゃの置き場所は「親が運び始める瞬間」に環境が決めるので、
        #   reset 直後はまだ決まっていない（`_rest_pos` が退避位置か未設定）。
        #   ここで距離を測ると nan や無意味な値になる（表示が「nan cm」になっていた）。
        #   → 決まっていなければ環境に決めさせてから測る。
        if getattr(u, "_rest_pos", None) is None or \
                not np.all(np.isfinite(np.asarray(u._rest_pos, dtype=float))):
            if hasattr(u, "_set_anchor"):
                u._set_anchor()
        _rp = np.asarray(getattr(u, "_rest_pos", None), dtype=float)
        dist = float(np.linalg.norm(_rp - eye)) if _rp.size == 3 else float("nan")
        base = eye + fwd * dist

        if orient_on:
            print(f"=== 視線誘導反射：おもちゃを中心へ寄せられるか"
                  f"（{'上下' if IS_V else '左右'}方向）===")
            print(f"  顔からおもちゃまで {dist*100:.1f} cm   {SECONDS} 秒間")
            print(f"  1発で詰める割合 SACCADE_FRAC={OR.SACCADE_FRAC}"
                  f"  間隔 {OR.SACCADE_LATENCY}s  閾値 {OR.SACCADE_MIN_STRENGTH}")
            print(f"  首の分担 {OR.NECK_SHARE}（暫定で首は動かさない）")
            print(f"  体年齢 {AGE:g}ヶ月（リーチングの月齢に揃えた）"
                  f"  視力も同じ月齢")
            print(f"  頭 {'実験者が抑える' if HEAD_HOLD else '支えなし'}"
                  f"（人間の乳児実験と同じ条件＝Hunter & Richards 2003）\n")

        for off in START_OFFSETS:
            key = (orient_on, off)
            results[key] = run(orient_on, off, env, u, m, d, dt,
                               toy_gadr, shift, toy_bid, n_steps, hands=hands,
                               scene=scene)
        env.close()

    # ---- 表 ----------------------------------------------------------------
    print("  ずれは符号つき（正＝おもちゃが視野の右）。眼球の動きも符号つき。")
    print("    正しければ、おもちゃが右にあるとき眼球も右へ動いてずれが減る\n")
    print(f"{'初期位置':>10}{'反射':>6}{'最初のずれ':>11}{'最後のずれ':>11}"
          f"{'後半の|ずれ|平均':>16}{'|ずれ|最小':>11}{'サッケード':>11}")
    _rows = []      # JSONに残す行（複数シードの集計用）
    for off in START_OFFSETS:
        for on in (True, False):
            ts, errs, eyes, sacc, cmds, tgts, hdirs, snap = results[(on, off)]
            good = errs[~np.isnan(errs)]
            e0 = float(np.nanmean(errs[:5])) if len(errs) >= 5 else float("nan")
            e1 = float(np.nanmean(errs[-5:])) if len(errs) >= 5 else float("nan")
            emin = float(np.abs(good).min()) if len(good) else float("nan")
            # 振動しているので「最後の値」は運で決まる。後半の平均で評価する。
            half = errs[len(errs) // 2:]
            ehalf = float(np.nanmean(np.abs(half))) if len(half) else float("nan")
            de = float(eyes[-1] - eyes[0]) if len(eyes) else 0.0
            print(f"{off*100:>+9.1f}cm{'ON' if on else 'OFF':>6}"
                  f"{e0:>11.3f}{e1:>11.3f}{ehalf:>16.3f}{emin:>11.3f}"
                  f"{int(sacc[-1]):>11d}")
            _rows.append(dict(axis=AXIS, seed=SEED, age=AGE, head_hold=HEAD_HOLD,
                              offset=float(off), orient=bool(on),
                              first=e0, last=e1, half_mean=ehalf, min_abs=emin,
                              saccades=int(sacc[-1]), seconds=SECONDS))

    # 【2026-07-28】結果をJSONで残す。複数シードを回して**まとめて集計する**ため。
    #   1シードの結果だけでは、改善がばらつきに埋もれているか判断できない。
    import json as _json
    _jp = os.path.join(OUT_DIR, f"result_{AXIS}_seed{SEED}.json")
    with open(_jp, "w", encoding="utf-8") as _fp:
        _json.dump(_rows, _fp, ensure_ascii=False, indent=1)
    print(f"\n  → 結果を保存: {os.path.relpath(_jp, _ROOT)}")

    # ---- グラフ ------------------------------------------------------------
    # 注意：squeeze=False：条件が1つのとき axes が1次元になって axes[0, j] が失敗する
    #   （下の診断図では reshape で対処していたが、こちらは漏れていた）
    fig, axes = plt.subplots(2, len(START_OFFSETS), squeeze=False,
                             figsize=(4.0 * len(START_OFFSETS), 6.4), sharex=True)
    for j, off in enumerate(START_OFFSETS):
        ax = axes[0, j]
        for on, c, lb in ((True, "tab:red", "反射ON"), (False, "tab:gray", "反射OFF")):
            ts, errs, eyes, sacc, cmds, tgts, hdirs, snap = results[(on, off)]
            ax.plot(ts, errs, color=c, lw=1.4, label=lb)
        ts, errs, eyes, sacc, cmds, tgts, hdirs, snap = results[(True, off)]
        ax.plot(ts, hdirs, color="tab:green", lw=1.1, ls=":",
                label="反射が出す向き h_dir")
        ax.set_title(f"初期位置 {off*100:+.1f} cm")
        ax.set_ylabel("中心からのずれ")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=9)
        ax2 = axes[1, j]
        ts, errs, eyes, sacc, cmds, tgts, hdirs, snap = results[(True, off)]
        ax2.plot(ts, eyes, color="tab:blue", lw=1.4, label="実際の角度")
        ax2.plot(ts, tgts, color="tab:orange", lw=1.0, ls="--", label="目標角度")
        if j == 0:
            ax2.legend(fontsize=8)
        ax2.set_xlabel("時刻[秒]")
        ax2.set_ylabel(f"眼球の{'垂直' if IS_V else '水平'}角[度]")
        ax2.grid(alpha=0.3)
    fig.suptitle(f"視線誘導反射（{'上下' if IS_V else '左右'}方向）：おもちゃを視野の中心へ寄せられるか"
                 "（上＝ずれ／下＝眼球の角度）", fontsize=12)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"converge_{AXIS}_seed{SEED}.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"\n  グラフを保存: {png}")

    # ---- 診断図：反射が働いている最中の「目に映る像」と「動き検出の出力」----
    fig2, ax2s = plt.subplots(2, len(START_OFFSETS),
                              figsize=(3.4 * len(START_OFFSETS), 7.0))
    if len(START_OFFSETS) == 1:
        ax2s = ax2s.reshape(2, 1)
    for j, off in enumerate(START_OFFSETS):
        snap = results[(True, off)][-1]
        axa, axb = ax2s[0, j], ax2s[1, j]
        rgb = snap.get("rgb")
        if rgb is not None:
            im = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 1)
            axa.imshow(im)
        axa.set_title(f"{off*100:+.1f} cm  目に映る像", fontsize=10)
        mo = snap.get("motion")
        if mo is not None and np.asarray(mo).max() > 0:
            axb.imshow(mo, cmap="inferno")
            axb.set_title(f"動き検出（最大 {float(np.asarray(mo).max()):.3f}）"
                          f"／向き {snap.get('dir', 0.0):+.2f}", fontsize=9)
        else:
            axb.set_title("動き検出＝空", fontsize=9)
        for ax in (axa, axb):
            ax.set_xticks([]); ax.set_yticks([])
    fig2.suptitle(f"反射が働いている最中の診断（{'上下' if IS_V else '左右'}方向）",
                  fontsize=12)
    fig2.tight_layout()
    png3 = os.path.join(OUT_DIR, f"diag_{AXIS}_seed{SEED}.png")
    fig2.savefig(png3, dpi=110)
    plt.close(fig2)
    print(f"  診断図を保存: {png3}")
    VIS.close_renderers()


if __name__ == "__main__":
    main()
