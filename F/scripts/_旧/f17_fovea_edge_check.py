"""F1-7 追加検証A・B（ユーザー指摘、2026-08-22）。
A: 中心窩カメラが本当に周辺カメラの中央と同じものを見ているかの目視確認。
B: 208px全体をエンコーダへ渡す構成で、FFTの周期境界による折り返し（落とし穴チェック
   リスト項126）がエンコーダ入力に混入していないかの事実確認。対処はしない。
"""
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mujoco  # noqa: E402
from run.config import Config              # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402
from vision_backends import fovea_crop      # noqa: E402
import e_toy_env as TE                      # noqa: E402
sys.path.insert(0, os.path.join(_ROOT, "E", "scripts"))
from f14_direct_readout import eye_addrs, set_eye_deg, solve_gaze, get_vision  # noqa: E402

FIG_DIR = os.path.join(_ROOT, "F", "logs", "F1-7_中心窩_2026-08-22")
os.makedirs(FIG_DIR, exist_ok=True)
# 【2026-09-14】以前はここに開発機の絶対パス（Claude Codeのscratchpad）を直書きしていたが、
#   ユーザー名がリポジトリに残るうえ、そのセッションフォルダ自体が既に消えていた。
#   _ROOT基準の F/logs/_scratch に変更（F/logs/ は .gitignore 済み＝出力は公開されない）。
_SCRATCHPAD_DIR = os.path.join(_ROOT, "F", "logs", "_scratch")
os.makedirs(_SCRATCHPAD_DIR, exist_ok=True)
BASE_SCENE_NAME = "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21"
TARO_BASE = dict(actuation="muscle", age_months=6.0, lr=0.0)


def _make_scene_variant(fovea_camera):
    import copy, json
    import run.scene_tools.scene_io as scene_io
    scene = copy.deepcopy(scene_io.load(BASE_SCENE_NAME))
    scene.pop("_path", None)
    scene["body"]["fovea_camera"] = bool(fovea_camera)
    out_path = os.path.join(_SCRATCHPAD_DIR, f"f17_edge_{'on' if fovea_camera else 'off'}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(scene, fp, ensure_ascii=False, indent=2)
    return out_path


def _build():
    path = _make_scene_variant(True)
    spec = {"name": "f17_edge", "scene": path, "taro": dict(TARO_BASE),
            "run": {"type": "train", "steps": 1, "seed": 0, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    return tr


def isolate(model, data, keep_body, hide_body):
    gadr_keep = int(model.body(keep_body).geomadr[0])
    gadr_hide = int(model.body(hide_body).geomadr[0])
    rgba_keep = model.geom_rgba[gadr_keep].copy()
    model.geom_rgba[gadr_keep] = rgba_keep
    model.geom_rgba[gadr_hide][3] = 0.0
    return gadr_keep, gadr_hide, rgba_keep


def restore(model, gadr_keep, gadr_hide, rgba_keep, rgba_hide):
    model.geom_rgba[gadr_keep] = rgba_keep
    model.geom_rgba[gadr_hide] = rgba_hide


def main():
    log = []

    def say(s=""):
        print(s)
        log.append(str(s))

    tr = _build()
    env, taro = tr.env, tr.taro
    tr.env.reset(seed=0)
    model, data = env.unwrapped.model, env.unwrapped.data
    addrs = eye_addrs(model)
    gadr1 = int(model.body("test_object1").geomadr[0])
    gadr2 = int(model.body("test_object2").geomadr[0])
    rgba1_orig = model.geom_rgba[gadr1].copy()
    rgba2_orig = model.geom_rgba[gadr2].copy()

    # ========================================================================
    # 検証A：中心窩=周辺中央、の目視確認（箱・球それぞれ中心へ視線固定）
    # ========================================================================
    say("=== 検証A：視線を物体中心に固定した状態での 周辺 vs 中心窩 ===")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for row, (toy, body_name, gadr_show, gadr_hide) in enumerate((
        ("toy1(箱)", "test_object1", gadr1, gadr2),
        ("toy2(球)", "test_object2", gadr2, gadr1),
    )):
        model.geom_rgba[gadr1] = rgba1_orig
        model.geom_rgba[gadr2] = rgba2_orig
        model.geom_rgba[gadr_hide][3] = 0.0
        mujoco.mj_forward(model, data)
        target = np.array(data.body(body_name).xpos, dtype=float).copy()
        h0, v0, err0 = solve_gaze(model, data, addrs, target)
        say(f"  {toy}: h={h0:+.2f} v={v0:+.2f} 残差=({err0[0]:+.3f},{err0[1]:+.3f})")
        set_eye_deg(data, addrs, h0, v0)
        mujoco.mj_forward(model, data)
        imgs = get_vision(env)
        periph = np.asarray(imgs["eye_left"]).astype(np.uint8)
        fovea = np.asarray(imgs["eye_left_fovea"]).astype(np.uint8)
        center32 = np.asarray(fovea_crop(periph, 32))
        center32_up = cv2.resize(center32, (fovea.shape[1], fovea.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        axes[row, 0].imshow(periph); axes[row, 0].set_title(f"{toy}\n周辺全体128px"); axes[row, 0].axis("off")
        axes[row, 1].imshow(center32_up); axes[row, 1].set_title("周辺・中央32px\n(最近傍で拡大表示)"); axes[row, 1].axis("off")
        axes[row, 2].imshow(fovea); axes[row, 2].set_title(f"中心窩{fovea.shape[0]}px"); axes[row, 2].axis("off")
    fig.suptitle("検証A：視線固定・周辺全体/周辺中央32px(拡大)/中心窩")
    fig.tight_layout()
    figA = os.path.join(FIG_DIR, "検証A_視線固定_周辺中央vs中心窩.png")
    fig.savefig(figA, dpi=150)
    plt.close(fig)
    say(f"[図A] {figA}")

    # ========================================================================
    # 検証B：FFT折り返し（周期境界）の事実確認
    # ========================================================================
    say("\n=== 検証B：中心窩画像へのFFT折り返し混入の確認 ===")
    model.geom_rgba[gadr1] = rgba1_orig
    model.geom_rgba[gadr2] = rgba2_orig
    model.geom_rgba[gadr2][3] = 0.0   # toy2(球)を隠し、toy1(箱)だけを対象にする
    mujoco.mj_forward(model, data)
    target = np.array(data.body("test_object1").xpos, dtype=float).copy()
    h0, v0, _ = solve_gaze(model, data, addrs, target)
    # 中心窩の半視野角7.5度に近い所まで寄せ、箱を画面端付近に押しやる
    h_edge = h0 + 6.5
    set_eye_deg(data, addrs, h_edge, v0)
    mujoco.mj_forward(model, data)
    imgs_with = get_vision(env)
    fovea_with = np.asarray(imgs_with["eye_left_fovea"]).astype(np.float64)

    # 同じ目位置で、箱も隠して「実体なし」の背景画像を撮る
    model.geom_rgba[gadr1][3] = 0.0
    env.unwrapped._vision_t = None
    mujoco.mj_forward(model, data)
    imgs_without = get_vision(env)
    fovea_without = np.asarray(imgs_without["eye_left_fovea"]).astype(np.float64)
    model.geom_rgba[gadr1] = rgba1_orig  # 復元

    diff = np.abs(fovea_with - fovea_without).max(axis=2)   # (H,W) 実体差分の最大チャンネル差
    h, w = diff.shape
    say(f"  視線: h={h_edge:+.2f}(中心から+6.5度) v={v0:+.2f}（箱を画面右寄りへ）")
    say(f"  画像サイズ: {h}x{w}")

    thresh = 8.0   # 8/255 ≈ 3%の輝度差を「実体でない滲み」の検出しきい値とする
    affected = diff > thresh
    # 「箱が実際に写っている側」(右半分、仮)と「反対側」(左半分)を分けて集計
    right_half = affected[:, w // 2:]
    left_half = affected[:, :w // 2]
    say(f"  差分>{thresh}(/255) の画素比率: 全体={affected.mean()*100:.2f}% "
        f"/ 箱が写る側(右半分)={right_half.mean()*100:.2f}% "
        f"/ 反対側(左半分)={left_half.mean()*100:.2f}%")
    # 反対端（さらに外側1/4列）だけに絞った比率も出す
    far_edge = affected[:, :w // 4]
    say(f"  反対側のさらに外側1/4列だけの比率={far_edge.mean()*100:.2f}%"
        f"（最大差分={diff[:, :w // 4].max():.2f}/255）")

    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 4))
    axes2[0].imshow(fovea_with.astype(np.uint8)); axes2[0].set_title("箱あり(端寄り配置)"); axes2[0].axis("off")
    axes2[1].imshow(fovea_without.astype(np.uint8)); axes2[1].set_title("箱なし(同じ目位置)"); axes2[1].axis("off")
    im = axes2[2].imshow(diff, cmap="inferno", vmin=0, vmax=50)
    axes2[2].set_title("差分(最大chの絶対値)"); axes2[2].axis("off")
    fig2.colorbar(im, ax=axes2[2], fraction=0.046)
    fig2.suptitle("検証B：中心窩画像へのFFT折り返し(周期境界)の有無")
    fig2.tight_layout()
    figB = os.path.join(FIG_DIR, "検証B_FFT折り返し確認.png")
    fig2.savefig(figB, dpi=150)
    plt.close(fig2)
    say(f"[図B] {figB}")

    close_env(env)

    log_path = os.path.join(_ROOT, "E", "logs", "F1-7_中心窩", "f17_edge_check_標準出力.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
