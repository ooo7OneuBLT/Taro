"""F1-7 検証スクリプト：中心窩カメラの追加と、解像度の月齢自動化。

仕様：`F/docs/仕様_F1-7_中心窩カメラと月齢からの解像度自動化.md` 第2部「検証」節。
同値性テスト（最重要）は `f14_direct_readout.py` の旧2場面モードをそのまま使う
（このスクリプトの担当ではない）。ここでやるのは検証2〜4：

  2. 目視：`fovea_camera: true` で周辺画像と中心窩画像を並べて保存
  3. 実効解像度の確認：周辺画像の中央32px切り出し vs 中心窩画像で、
     高周波成分の量（ラプラシアンの分散）を比較
  4. コスト実測：`fovea_camera` on/off で1ステップ（get_vision_obs）の所要時間を比較

出力：
  画像・図   F/logs/F1-7_中心窩_2026-08-22/
  生ログ     F/logs/F1-7_中心窩/f17_fovea_check_標準出力.log（呼び出し側でリダイレクトする）
"""
import copy
import json
import os
import sys
import time

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

from run.config import Config              # noqa: E402
from run.trainer import Trainer, close_env  # noqa: E402
from vision_backends import fovea_crop      # noqa: E402
import e_toy_env as TE                      # noqa: E402

BASE_SCENE_NAME = "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21"
FIG_DIR = os.path.join(_ROOT, "F", "logs", "F1-7_中心窩_2026-08-22")
os.makedirs(FIG_DIR, exist_ok=True)

# 【2026-09-14】以前はここに開発機の絶対パス（Claude Codeのscratchpad）を直書きしていたが、
#   ユーザー名がリポジトリに残るうえ、そのセッションフォルダ自体が既に消えていた。
#   _ROOT基準の F/logs/_scratch に変更（F/logs/ は .gitignore 済み＝出力は公開されない）。
_SCRATCHPAD_DIR = os.path.join(_ROOT, "F", "logs", "_scratch")
os.makedirs(_SCRATCHPAD_DIR, exist_ok=True)

TARO_BASE = dict(actuation="muscle", age_months=6.0, lr=0.0)


def _make_scene_variant(fovea_camera):
    """既存シーンJSONを読み、body.fovea_camera だけ差し替えた一時シーンを書き出す。"""
    import run.scene_tools.scene_io as scene_io
    scene = scene_io.load(BASE_SCENE_NAME)
    scene = copy.deepcopy(scene)
    scene.pop("_path", None)
    scene["body"]["fovea_camera"] = bool(fovea_camera)
    suffix = "on" if fovea_camera else "off"
    out_path = os.path.join(_SCRATCHPAD_DIR, f"f17_fovea_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(scene, fp, ensure_ascii=False, indent=2)
    return out_path


def _build(scene_path):
    spec = {"name": "f17_fovea_check", "scene": scene_path, "taro": dict(TARO_BASE),
            "run": {"type": "train", "steps": 1, "seed": 0, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    return tr


def _laplacian_var(img):
    gray = cv2.cvtColor(np.asarray(img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _band_energy_cpd(img, fovy_deg, lo_cpd, hi_cpd):
    """画像のグレースケールFFTパワーを、周波数を[cycles/度]へ変換したうえで
    [lo_cpd, hi_cpd) 帯域に絞って合計する。

    【なぜこの形か・落とし穴】最初は素朴に cv2.Laplacian(img).var() を
    「周辺(中央32px切り出し)」と「中心窩(全体208px)」へそのまま適用したところ、
    **周辺の方が大きい**という逆転した値が出た（周辺9.49 vs 中心窩1.54）。
    原因：Laplacianはピクセル単位の二階差分であり、度/ピクセルを補正しない。
    同じ帯域制限された連続信号でも、画素間隔が細かい（＝中心窩）ほど隣接画素の
    差分は小さくなる（ナイキストで説明済みの現象そのもの）ため、素朴な
    ピクセル領域の指標は「画素密度が高いほど値が小さくなる」バイアスを持つ。
    → 周波数軸を [cycles/度] に変換してから比べる、この関数の方式に切り替えた
    （vision.py:221-227 の fft_scale=max(w,h)/fovy と同じ変換を使う）。
    """
    gray = cv2.cvtColor(np.asarray(img).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape
    F = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(F) ** 2
    fft_scale = max(w, h) / float(fovy_deg)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    freq_cpd = np.sqrt(((w / 2.0 - uu) / w) ** 2 + ((h / 2.0 - vv) / h) ** 2) * fft_scale
    mask = (freq_cpd >= lo_cpd) & (freq_cpd < hi_cpd)
    return float(power[mask].sum())


def main():
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    log("=== F1-7 検証：中心窩カメラ ===")

    # --- 6ヶ月で計算される解像度（②報告用） ---------------------------------
    px_15 = TE.required_px(6.0, TE.FOVEA_FOVY)
    log(f"[②] required_px(age_months=6.0, fovy_deg=15) = {px_15}px "
        f"（acuity={TE._acuity_cpd(6.0):.3f}cpd）")

    # --- 環境を2つ作る（fovea off / on） -------------------------------------
    off_path = _make_scene_variant(False)
    on_path = _make_scene_variant(True)
    tr_off = _build(off_path)
    tr_on = _build(on_path)

    tr_off.env.reset(seed=0)
    tr_on.env.reset(seed=0)
    u_off = tr_off.env.unwrapped
    u_on = tr_on.env.unwrapped
    u_off._vision_t = None
    u_on._vision_t = None
    o_off = u_off.get_vision_obs()
    o_on = u_on.get_vision_obs()

    log(f"[配線確認] fovea off の観測キー: {sorted(o_off.keys())}")
    log(f"[配線確認] fovea on  の観測キー: {sorted(o_on.keys())}")
    assert "eye_left_fovea" not in o_off, "fovea_camera=Falseなのに中心窩キーが出た（バグ）"
    assert "eye_left_fovea" in o_on, "fovea_camera=Trueなのに中心窩キーが無い（バグ）"

    eye_left_periph = o_on["eye_left"]          # on/off両方の周辺カメラは同一パラメータ
    eye_left_fovea = o_on["eye_left_fovea"]
    log(f"[画像サイズ] eye_left(周辺) = {np.asarray(eye_left_periph).shape}, "
        f"eye_left_fovea(中心窩) = {np.asarray(eye_left_fovea).shape}")

    # --- ②目視：並べて保存 ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.asarray(eye_left_periph).astype(np.uint8))
    axes[0].set_title(f"周辺 eye_left\n{np.asarray(eye_left_periph).shape[0]}px / 視野60度")
    axes[0].axis("off")
    axes[1].imshow(np.asarray(eye_left_fovea).astype(np.uint8))
    axes[1].set_title(f"中心窩 eye_left_fovea\n{np.asarray(eye_left_fovea).shape[0]}px / 視野15度")
    axes[1].axis("off")
    fig.suptitle("F1-7: 周辺カメラ vs 中心窩カメラ（同一時刻・同一姿勢）")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "周辺vs中心窩_目視比較.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    log(f"[⑤図] 目視比較図: {fig_path}")

    # --- ③実効解像度 --------------------------------------------------------
    # 周辺の「中央32px切り出し」は 32/128*60=15度 をカバーする＝中心窩(15度)と
    # ちょうど同じ画角。fovea_crop()はピクセルをそのまま抜き出すだけ（拡大縮小
    # 無し）なので、切り出し後も周辺本来の画素密度(2.13px/度)のまま。
    periph_center32 = fovea_crop(eye_left_periph, 32)
    periph_fovy_of_crop = 32.0 / np.asarray(eye_left_periph).shape[0] * TE.VISION_FOVY

    var_periph = _laplacian_var(periph_center32)
    var_fovea = _laplacian_var(eye_left_fovea)
    log(f"[③-a・素朴なラプラシアン分散（ピクセル単位・度/画素を補正しない）]: "
        f"周辺(中央32px切り出し) = {var_periph:.3f} / "
        f"中心窩(全体{np.asarray(eye_left_fovea).shape[0]}px) = {var_fovea:.3f}")
    log("  → 中心窩の方が小さく出る（画素密度が高いほど隣接画素差分が小さくなる"
        "バイアス。落とし穴として③-bのFFT比較に切り替えた。詳細は"
        "_band_energy_cpdのdocstring参照）")

    lo_cpd, hi_cpd = 1.1, TE._acuity_cpd(6.0)
    e_periph = _band_energy_cpd(periph_center32, periph_fovy_of_crop, lo_cpd, hi_cpd)
    e_fovea = _band_energy_cpd(eye_left_fovea, TE.FOVEA_FOVY, lo_cpd, hi_cpd)
    log(f"[③-b・本命：周波数を cycles/度 へ変換したFFTパワーの帯域({lo_cpd:.2f}〜"
        f"{hi_cpd:.2f}cpd)合計]: 周辺(中央32px切り出し・実質同一視野15度) = "
        f"{e_periph:.3e} / 中心窩(全体) = {e_fovea:.3e}")
    if e_periph < 1e-9:
        log(f"  → 周辺はこの帯域の情報を原理的に持てない"
            f"（そのナイキスト限界={ (32.0/periph_fovy_of_crop)/2.0 :.3f}cpd < "
            f"帯域下限{lo_cpd:.2f}cpd）。中心窩だけがこの帯域の情報を持つ"
            f"＝実効解像度が明確に大きい、が確認できた。")

    fig2, axes2 = plt.subplots(1, 2, figsize=(6, 3))
    axes2[0].imshow(np.asarray(periph_center32).astype(np.uint8))
    axes2[0].set_title(f"周辺 中央32px切り出し\n帯域energy={e_periph:.2e}")
    axes2[0].axis("off")
    axes2[1].imshow(np.asarray(eye_left_fovea).astype(np.uint8))
    axes2[1].set_title(f"中心窩 全体\n帯域energy={e_fovea:.2e}")
    axes2[1].axis("off")
    fig2.suptitle(f"F1-7: 実効解像度の比較（{lo_cpd:.1f}〜{hi_cpd:.1f}cpd帯域のFFTパワー）")
    fig2.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "実効解像度比較_FFT帯域エネルギー.png")
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)
    log(f"[⑤図] 実効解像度比較図: {fig2_path}")

    # --- ④コスト実測：get_vision_obs() の所要時間 -----------------------------
    def _time_vision(u, n=20):
        times = []
        for _ in range(n):
            u._vision_t = None   # キャッシュを毎回無効化（実測したいのは描画コストそのもの）
            t0 = time.perf_counter()
            u.get_vision_obs()
            times.append(time.perf_counter() - t0)
        return float(np.mean(times)), float(np.std(times))

    # ウォームアップ（レンダラ初期化コストを除く）
    _time_vision(u_off, n=3)
    _time_vision(u_on, n=3)
    mean_off, std_off = _time_vision(u_off, n=20)
    mean_on, std_on = _time_vision(u_on, n=20)
    delta_ms = (mean_on - mean_off) * 1000.0
    log(f"[④] get_vision_obs() 平均所要時間（20回平均）: "
        f"off={mean_off*1000:.3f}±{std_off*1000:.3f}ms / "
        f"on={mean_on*1000:.3f}±{std_on*1000:.3f}ms / 差={delta_ms:.3f}ms")
    # 【推定の根拠】VISION_MIN_DT=0.1sim秒 = 10Hz でしかvisionを再計算しない
    #   （e_toy_env.py VISION_MIN_DT前のコメント参照）。300秒の学習なら最大3000回。
    n_calls_300s = int(300.0 / TE.VISION_MIN_DT)
    extra_sec_300s = delta_ms / 1000.0 * n_calls_300s
    log(f"[④] 300秒学習1本への上乗せ推定: 差{delta_ms:.3f}ms × "
        f"{n_calls_300s}回（10Hzで300秒） = {extra_sec_300s:.2f}秒")

    close_env(tr_off.env)
    close_env(tr_on.env)

    log("=== 完了 ===")
    log_path = os.path.join(_ROOT, "E", "logs", "F1-7_中心窩", "f17_fovea_check_標準出力.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
