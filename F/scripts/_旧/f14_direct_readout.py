"""F1-4/F1-6 直接読み出しテスト（2026-08-21 初版 → 2026-08-22 改修）。

語（わんわん／ぶーぶー）を聞かせ、連合器(Lexicon)が出す「想像の見え埋め込み」を
取り出し、held-out視点で撮った2つのおもちゃの実物の見え埋め込みと照合して、
正解（実物）を選べるかを、学習済みモデルで測る。

【2026-08-22改修（F1-6 100場面での語彙テスト。仕様＝
  F/docs/仕様_F1-6_100場面での語彙テスト.md 第2部「f14_direct_readout.pyの改修」節）】
  1. `--scenes-dir`：フォルダ内の全JSONを1場面ずつ評価する新モード追加
     （F1-6の100場面／実際に生成できた場面数のマトリクスを想定）。
     この新モードは「1個ずつ提示」（写真の撮り方：片方のおもちゃをrgbaアルファ0で
     透明化してから中心へ視線を合わせて撮る）で視点を撮る＝`capture_views_isolated`。
     【なぜ既存の`--scenes`（A/B）モードと透明化の有無を分けたか】既存2場面モードは
     箱・球が角度差12度で近接しており、32pxフォビア窓（視野15度相当）にほぼ収まらず
     自然に「ほぼ1個ずつ」になっていた実測（2026-08-22時点の既知の事実）があり、
     この視点採取ロジックを変えると2026-08-22時点の既知の結果（3シードとも14/20）が
     再現できなくなる。→ 既存モードの視点採取（`capture_views_legacy`）は
     一切変更せず、透明化は新モード専用の別関数として追加した。
  2. `OFFSETS_DEG` 既定を `[0.0]` のみに変更（場面数そのものが独立性の源。
     角度の微小ずらしは新モードでは使わない）。既存2場面の同値性確認では
     `--offsets -3 -1.5 0 1.5 3` を明示指定して従来と同じ20問を再現する。
  3. **両側引き算をスクリプト内に実装**（従来は素のcosのみ）：
     語側 `w' = w - mean(その model の全語のw)`／
     視点側 `v' = v - mean(採取した全視点・全おもちゃのv)`（1回のみ・model非依存。
     視覚バックエンドはフリーズ済みDINOv2でmodelの重みに依存しないため）／
     `cos(w', v')` の大きい方を答えとする（`two_sided_score`）。
     根拠：F/docs/研究日誌_2026-08.md 2026-08-22 追記
     「両側引き算（実測：素のcos採点だと同じモデルが9〜12/20に見える）」。
  4. `--fovea-px` 既定を 64→**32** に変更（学習時 F1-5_fovea32 と一致させる。
     不一致だと成績が壊れる、既知の罠）。
  5. 既定の判定対象モデルを3シード
     `F/models/F1-5_fovea32_300s_seed{10,11,12}_2026-08-21.pt` に変更
     （旧: trained/blank 2本 → 新: 3シード）。
  6. 出力の拡張：
     - 明細.csv の列を `model, scene, word, sim_toy1, sim_toy2, answer, correct, diff`
       に変更（`view_pair`→`scene`、`diff`=正解-不正解の類似度の差を追加。
       副次②「差の大きさ」に対応）
     - 標準出力にシードごとの「わんわん」正答数/場面数（主判定＝わんわん語のみ。
       ぶーぶーは両側引き算＋2語では数学的に答えが反転するだけで情報がゼロ
       ＝仕様書の指示どおり数えない。ただしCSV明細には両語とも残す）
     - 図1：横軸=場面、縦軸=3シード、正誤（わんわん問題）を色で塗ったマップ
       （`--scenes-dir` モードでのみ作る。既存2場面モードでは作らない＝
       場面数が2しかなく意味を持たないため）
     - 図2（副次①）：シードごとに「わんわん」の想像embeddingに近い上位5枚の
       写真を並べた図。あわせてAUC・上位10枚の内訳を標準出力へ

  ---- 以下は2026-08-21初版のヘッダ・流用元メモ（変更していない箇所に適用中）----

【流用した既存の経路（車輪の再発明はしない・仕様の要求）】
  - 太郎の組み立て　　run/trainer.py Trainer.build()（run/main.py の run.type="train"
    経路と同一。乱数消費順序・環境構築を独自実装しない）
  - 語→トークン列　　taro.hearing.hear(word)（run/trainer.py:260 相当・
    taro_core/src/senses/hearing.py の Hearing.hear、学習時と同一関数）
  - トークン列→チャンク　　taro.lexicon.observe(tokens, confidences, state=None)
    （run/trainer.py:269 と同一呼び出し。state=None を渡すので
    taro_core/src/brain/lexicon.py の Lexicon.observe 内 89-96行の
    state_sum 更新分岐は実行されない＝読み出し専用として安全に使える。
    confidences=[1.0]*len(tokens) も trainer.py:261 と同一）
  - チャンク→想像の見え埋め込み　　taro.lexicon.assoc(chunk)
    （run/trainer.py:281, 311 と同一呼び出し）
  - 視覚→埋め込み　　taro.vision_backend.encode(eye_left, eye_right)
    （run/trainer.py:268, 310 と同一呼び出し。DINOv2バックエンド、
    taro_core/src/senses/vision_backends.py:188-195 の
    DINOv2VisionBackend.encode。学習済みチェックポイントの重みに一切依存しない
    フリーズ済み事前学習モデルなので、想像embeddingの取り出しに使う
    taro（trained/blank）とは別に、視点採取専用の太郎インスタンスで
    まとめて計算しても学習時と同一の埋め込みが得られる＝実装の裁量で
    ビルド回数を削減した）
  - コサイン類似度　　run/trainer.py の _cosine_sim(a, b)（41-54行）をそのまま import
    （orienting.set_recognition へ渡すのと同じ規約の関数を流用）

【新規に書いた部分（このファイル固有・既存に同等の関数が無いことをGrepで確認済み）】
  - held-out視点の直接指定：眼球関節(robot:{left,right}_eye_{horizontal,vertical})の
    qposを直接書き換え→mujoco.mj_forward→env.unwrapped.get_vision_obs()。
    【既知の罠】(a) mj_forwardを呼び忘れると qpos書き換えが視覚に反映されない
    (E/scripts/e_orient_diag.py 診断2の教訓)。(b) get_vision_obs() は
    VISION_MIN_DT(=0.1sim秒)ごとにキャッシュを使い回すので、
    env.unwrapped._vision_t = None にしてから呼ばないと古い画が返る
    （同スクリプトの手法をそのまま踏襲）。
  - 視線をおもちゃの中心へ向ける眼球角度の探索（2自由度・数値ヤコビアンのNewton法）。
    eye_leftカメラのright/up/fwdベクトルから見た目標方向の水平・垂直角度誤差を
    ゼロに近づける。既存コードに同等の「狙った方向を向かせる」関数が無いため新規実装。

書き込み先（指示された範囲のみ）：
  このファイル自身／scratchpad／
  F/logs/F1-4_直接読み出し_2026-08-21/ ／ F/logs/F1-4_直接読み出し/ ／
  F/logs/F1-6_100場面_2026-08-22/ ／ F/logs/F1-6_100場面/ ／
  E/logs配下の新実験ログ（--e-dir で指定した先）
"""
import argparse
import csv
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np                                              # noqa: E402
import cv2                                                       # noqa: E402
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                             # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
matplotlib.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mujoco                                                     # noqa: E402

from run.config import Config                                     # noqa: E402
from run.trainer import Trainer, close_env, _cosine_sim           # noqa: E402
# run.trainer の import 経路で taro_core/src/senses が sys.path に入るので、
#   ここで初めて import できる（run/taro_setup.py 冒頭のsys.path工作に依存）。
from vision_backends import fovea_crop                            # noqa: E402

# ============================================================================
# 設定
# ============================================================================
DEFAULT_MODEL_PATH = "F/models/F1-4h_語彙_2026-08-21.pt"
DEFAULT_SCENES = {
    "A": "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21",
    "B": "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行B_2026-08-21",
}
TOY_BODY = {"toy1": "test_object1", "toy2": "test_object2"}
# parent_labeling.utterances（両シーンとも共通）：toy1="ぶーぶー", toy2="わんわん"
WORD_KEY = {"わんわん": "wanwan", "ぶーぶー": "boobu"}
ANSWER = {"わんわん": "toy2", "ぶーぶー": "toy1"}
PRIMARY_WORD = "わんわん"   # 主判定はこの語のみ（2026-08-22・仕様書の指示）
# 【2026-08-22変更】既定は場面数そのものが独立性の源。角度の微小ずらしは使わない。
#   既存2場面の同値性確認をするときだけ --offsets -3 -1.5 0 1.5 3 を明示指定する。
OFFSETS_DEG = [0.0]

# 【2026-08-22変更】既定の判定対象モデルを3シードに変更
#   （F/docs/仕様_F1-6_100場面での語彙テスト.md 技術付録 手順5）
DEFAULT_MODEL_SPECS = [
    ("seed10", "F/models/F1-5_fovea32_300s_seed10_2026-08-21.pt"),
    ("seed11", "F/models/F1-5_fovea32_300s_seed11_2026-08-21.pt"),
    ("seed12", "F/models/F1-5_fovea32_300s_seed12_2026-08-21.pt"),
]

TARO_BASE = dict(
    actuation="muscle", age_months=6.0, orienting_reflex=True,
    hearing=True, lexicon_vision={"backend": "dinov2_vits14", "fovea_px": 32},
    word_attention={"active_sec": 3.0, "enabled": True}, lr=0.0,
)

# 【2026-08-23・F1-9C】視覚バックエンドを `--backend` で差し替えられるようにした。
#   それまでは "dinov2_vits14" がここに直接書かれており、**学習時と試験時で
#   違う目を使ってしまう**危険があった（F1-8で実際に起きた不備と同型：
#   学習は中心窩208px・試験は周辺32pxという食い違い）。
#   目のアブレーション（設計_F1-9 第2部C節）では、訓練済み／訓練前の器を
#   入れ替えて比べるので、**学習に使ったバックエンドと必ず一致させること**。

_F_DIR_DEFAULT = os.path.join(_ROOT, "F", "logs", "F1-4_直接読み出し_2026-08-21")
_VIEWS_DIR_DEFAULT = os.path.join(_F_DIR_DEFAULT, "views")
_E_DIR_DEFAULT = os.path.join(_ROOT, "E", "logs", "F1-4_直接読み出し")
_FIG_PATH_DEFAULT = os.path.join(_F_DIR_DEFAULT, "埋め込み地図.png")
# 【2026-09-14】以前はここに開発機の絶対パス（Claude Codeのscratchpad）を直書きしていたが、
#   ユーザー名がリポジトリに残るうえ、そのセッションフォルダ自体が既に消えていた。
#   _ROOT基準の F/logs/_scratch に変更（F/logs/ は .gitignore 済み＝出力は公開されない）。
_SCRATCHPAD_DIR = os.path.join(_ROOT, "F", "logs", "_scratch")
os.makedirs(_SCRATCHPAD_DIR, exist_ok=True)
_REPORT_PATH_DEFAULT = os.path.join(_SCRATCHPAD_DIR, "直接読み出しレポート.md")


def _resolve(path, base=_ROOT):
    """相対パスなら_ROOT基準の絶対パスにする（scratchpad始まりはそのまま）。"""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if path.replace("\\", "/").startswith("scratchpad/"):
        return os.path.join(_SCRATCHPAD_DIR, path.split("/", 1)[1])
    return os.path.join(base, path)


def parse_args():
    ap = argparse.ArgumentParser(description="F1-4/F1-6 直接読み出しテスト")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                     help="（後方互換）--models を指定しないときに使う単一モデル")
    ap.add_argument("--e-dir", default=None,
                     help="明細CSV/embeddings.npzの出力先ディレクトリ"
                          f"（既定: {_E_DIR_DEFAULT}）")
    ap.add_argument("--fig", default=None,
                     help=f"埋め込み地図PNGの出力先（既定: {_FIG_PATH_DEFAULT}。"
                          "--scenes-dir モードでは使わない代わりに図1/図2を出す）")
    ap.add_argument("--fig1", default=None,
                     help="図1（場面×シードの正誤マップ）の出力先"
                          "（--scenes-dir モード用。既定はe-dir配下）")
    ap.add_argument("--fig2", default=None,
                     help="図2（副次①：上位5枚）の出力先（--scenes-dir モード用）")
    ap.add_argument("--views-dir", default=None,
                     help=f"視点PNGの出力先ディレクトリ（既定: {_VIEWS_DIR_DEFAULT}）")
    ap.add_argument("--report", default=None,
                     help=f"レポートmdの出力先（既定: {_REPORT_PATH_DEFAULT}）")
    ap.add_argument("--fovea-px", type=int, default=32,
                    help="視覚エンコードのフォビア窓[px]（学習時の設定と一致させること。"
                         "2026-08-22既定を32に変更＝F1-5_fovea32学習と一致）")
    ap.add_argument("--backend", default="dinov2_vits14",
                    help="視覚バックエンド名（taro_core/src/senses/vision_backends.py の"
                         "BACKENDS に登録された名前）。**学習時と必ず一致させること**。"
                         "目のアブレーション（設計_F1-9 C節）では vits14_untrained を使う")
    ap.add_argument("--backend-seed", type=int, default=0,
                    help="vits14_untrained のランダム重みのseed（学習時と一致させること）")
    ap.add_argument("--tag", default="",
                     help="ログ・レポート見出しに付ける表示用ラベル（例: 延長）")
    ap.add_argument("--scenes", nargs="+", default=None,
                     help="（既存2場面モード）視点採取に使うシーン名（1個以上）。"
                          "指定が無くscenes-dirも無指定なら既定の2シーン")
    ap.add_argument("--scenes-dir", default=None,
                     help="（新モード・F1-6）このフォルダ内の全JSONを1場面ずつ評価する。"
                          "1個ずつ提示（アルファ0透明化）で視点を採取する")
    ap.add_argument("--offsets", nargs="+", type=float, default=None,
                     help="視線オフセット[度]のリスト。既定[0.0]のみ。既存2場面の"
                          "同値性確認では -3 -1.5 0 1.5 3 を明示指定する"
                          "（--scenes-dir モードでは常に[0.0]固定・この引数は無視）")
    ap.add_argument("--models", nargs="+", default=None,
                     help="判定対象モデルを label=path の形で複数指定。"
                          "path省略時または'None'で白紙モデル扱い。"
                          "既定は3シード（F1-5_fovea32_300s_seed10/11/12）")
    return ap.parse_args()


def fmt_offset(off):
    sign = "+" if off > 0 else ("-" if off < 0 else "")
    return f"{sign}{abs(off):.1f}"


# ============================================================================
# 太郎の組み立て（run/trainer.py Trainer.build() をそのまま呼ぶ）
# ============================================================================
def build_trainer(scene_name, *, with_model, model_path=None, seed=0):
    taro = dict(TARO_BASE)
    if with_model:
        taro["model"] = model_path or DEFAULT_MODEL_PATH
    spec = {"name": "f14_direct_readout", "scene": scene_name, "taro": taro,
            "run": {"type": "train", "steps": 1, "seed": seed, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    return tr


# ============================================================================
# 眼球関節の直接操作（新規実装。E/scripts/e_orient_diag.py・e_eye_camera_axes.py の
# 「qpos直接書き換え→mj_forward」パターン、VISION_MIN_DTキャッシュの罠を踏襲）
# ============================================================================
def eye_addrs(model):
    out = {}
    for side in ("left", "right"):
        jh = model.joint(f"robot:{side}_eye_horizontal")
        jv = model.joint(f"robot:{side}_eye_vertical")
        out[side] = (int(model.jnt_qposadr[jh.id]), int(model.jnt_qposadr[jv.id]))
    return out


def set_eye_deg(data, addrs, h_deg, v_deg):
    hr, vr = np.radians(h_deg), np.radians(v_deg)
    for _side, (ih, iv) in addrs.items():
        data.qpos[ih] = hr
        data.qpos[iv] = vr


def cam_frame(model, data, cam="eye_left"):
    cid = int(model.camera(cam).id)
    R = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)
    pos = np.array(data.cam_xpos[cid], dtype=float)
    return pos, R[:, 0], R[:, 1], -R[:, 2]     # pos, right, up, fwd


def angle_err_deg(model, data, target_pos):
    """目標方向とカメラ正面(fwd)とのズレを、カメラ右・上ベクトル基準の符号付き角度[度]で返す。"""
    pos, right, up, fwd = cam_frame(model, data)
    v = np.asarray(target_pos, dtype=float) - pos
    v = v / (np.linalg.norm(v) + 1e-12)
    h = np.degrees(np.arctan2(np.dot(v, right), np.dot(v, fwd)))
    vv = np.degrees(np.arctan2(np.dot(v, up), np.dot(v, fwd)))
    return np.array([h, vv])


def solve_gaze(model, data, addrs, target_pos, h0=0.0, v0=0.0, iters=25, eps=0.5, tol=0.02):
    """眼球水平・垂直関節の角度[度]を、視線が target_pos を向くように数値Newton法で解く。"""
    h, v = h0, v0

    def err_at(hh, vv):
        set_eye_deg(data, addrs, hh, vv)
        mujoco.mj_forward(model, data)
        return angle_err_deg(model, data, target_pos)

    e0 = err_at(h, v)
    for _ in range(iters):
        if float(np.max(np.abs(e0))) < tol:
            break
        eh = (err_at(h + eps, v) - e0) / eps
        ev = (err_at(h, v + eps) - e0) / eps
        J = np.stack([eh, ev], axis=1)
        try:
            delta = np.linalg.solve(J, -e0)
        except np.linalg.LinAlgError:
            delta = -e0
        delta = np.clip(delta, -15.0, 15.0)
        h = float(np.clip(h + delta[0], -44.0, 44.0))
        v = float(np.clip(v + delta[1], -46.0, 32.0))
        e0 = err_at(h, v)
    return h, v, e0


def get_vision(env):
    """VISION_MIN_DTキャッシュを無効化してから最新の視覚obsを取る（既知の罠2件対策）。"""
    u = env.unwrapped
    u._vision_t = None
    return u.get_vision_obs()


# ============================================================================
# 想像の見え埋め込み（trainer.pyの経路をそのまま流用。読み出し専用＝state=None）
# ============================================================================
def imagine_embedding(taro, word):
    t = taro
    tokens = t.hearing.hear(word)
    confidences = [1.0] * len(tokens)
    chunk = t.lexicon.observe(tokens, confidences, state=None)
    if chunk is None:
        return None, chunk
    vec = t.lexicon.assoc(chunk)
    if vec is None:
        return None, chunk
    return np.asarray(vec, dtype=np.float64), chunk


# ============================================================================
# 視点採取（1）既存2場面モード：従来のロジックをそのまま（同値性確認のため不変）
# ============================================================================
def capture_views_legacy(scenes, offsets, fovea_px, views_dir, say):
    views = {}          # (toy, scene_key, off) -> np.ndarray(384,)
    view_meta = []
    solve_report = []

    for scene_key, scene_name in scenes.items():
        say(f"\n=== 視点採取: シーン{scene_key} ({scene_name}) ===")
        tr = build_trainer(scene_name, with_model=False)
        env, taro = tr.env, tr.taro
        model, data = env.unwrapped.model, env.unwrapped.data
        addrs = eye_addrs(model)
        for toy in ("toy1", "toy2"):
            body_name = TOY_BODY[toy]
            mujoco.mj_forward(model, data)
            target = np.array(data.body(body_name).xpos, dtype=float).copy()
            h0, v0, err0 = solve_gaze(model, data, addrs, target)
            say(f"  {toy}({body_name}) 中心解: h={h0:+.2f}° v={v0:+.2f}° "
                f"残差=({err0[0]:+.3f}°,{err0[1]:+.3f}°)")
            solve_report.append((scene_key, toy, h0, v0, float(np.max(np.abs(err0)))))
            for off in offsets:
                set_eye_deg(data, addrs, h0 + off, v0)
                mujoco.mj_forward(model, data)
                imgs = get_vision(env)
                # 【2026-08-22・F1-8】中心窩カメラ(eye_left_fovea等)が観測にあれば
                #   そちらを優先してエンコードする。run/trainer.py の
                #   Trainer._vision_backend_encode() と完全に同じロジックを流用
                #   （2重実装しない）。F1-8以前は常にeye_left/eye_rightを直接渡して
                #   おり、fovea_camera:trueのシーンでも中心窩画像が一切使われない
                #   バグだった（学習時の経路と不一致）。旧2場面（fovea_camera無し）
                #   ではhas_foveaがFalseになるため挙動は1ビットも変わらない。
                vec = np.asarray(tr._vision_backend_encode(imgs), dtype=np.float64)
                views[(toy, scene_key, off)] = vec
                crop_src = imgs.get("eye_left_fovea", imgs["eye_left"])
                crop = fovea_crop(crop_src, fovea_px)
                png_path = os.path.join(views_dir, f"{toy}_{scene_key}_{fmt_offset(off)}.png")
                # 【罠】cv2.imwrite はWindowsで非ASCIIパス（日本語フォルダ名）だと
                #   例外を出さずに黙って失敗する。imencode→Python側書き込みで回避。
                ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"png encode失敗: {png_path}")
                buf.tofile(png_path)
                view_meta.append((toy, scene_key, off, png_path))
        close_env(env)
    return views, view_meta, solve_report


# ============================================================================
# 視点採取（2）新モード：--scenes-dir。1個ずつ提示（アルファ0透明化）
# ============================================================================
def capture_views_isolated(scene_paths, fovea_px, views_dir, say):
    """仕様書『写真の撮り方』節どおり：1場面につき2枚（箱のみ・球のみ）を、
    片方をrgbaアルファ0で透明化してから中心へ視線を合わせて撮る。

    シーンJSON自体のrgbaは変えず、build後に model.geom_rgba を直接書き換える
    （E/scripts/e_toy_env.py:726 と同じ配列への上書き。読むだけで済む）。
    """
    views = {}          # (toy, scene_name) -> np.ndarray(384,)
    view_meta = []       # (toy, scene_name, png_path)
    solve_report = []

    for scene_path in scene_paths:
        scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        say(f"\n=== 視点採取(1個ずつ提示): {scene_name} ===")
        tr = build_trainer(scene_path, with_model=False)
        env, taro = tr.env, tr.taro
        model, data = env.unwrapped.model, env.unwrapped.data
        addrs = eye_addrs(model)
        gadr1 = int(model.body("test_object1").geomadr[0])
        gadr2 = int(model.body("test_object2").geomadr[0])
        rgba1_orig = model.geom_rgba[gadr1].copy()
        rgba2_orig = model.geom_rgba[gadr2].copy()

        for toy, body_name, gadr_show, rgba_show, gadr_hide in (
            ("toy1", "test_object1", gadr1, rgba1_orig, gadr2),
            ("toy2", "test_object2", gadr2, rgba2_orig, gadr1),
        ):
            model.geom_rgba[gadr1] = rgba1_orig
            model.geom_rgba[gadr2] = rgba2_orig
            model.geom_rgba[gadr_hide][3] = 0.0
            mujoco.mj_forward(model, data)
            target = np.array(data.body(body_name).xpos, dtype=float).copy()
            h0, v0, err0 = solve_gaze(model, data, addrs, target)
            say(f"  {toy}({body_name}) 中心解: h={h0:+.2f}° v={v0:+.2f}° "
                f"残差=({err0[0]:+.3f}°,{err0[1]:+.3f}°)")
            solve_report.append((scene_name, toy, h0, v0, float(np.max(np.abs(err0)))))
            set_eye_deg(data, addrs, h0, v0)
            mujoco.mj_forward(model, data)
            imgs = get_vision(env)
            # 【2026-08-22・F1-8】capture_views_legacyと同じ理由でfovea優先へ変更
            #   （tr._vision_backend_encode を流用、2重実装しない）。
            vec = np.asarray(tr._vision_backend_encode(imgs), dtype=np.float64)
            views[(toy, scene_name)] = vec
            crop_src = imgs.get("eye_left_fovea", imgs["eye_left"])
            crop = fovea_crop(crop_src, fovea_px)
            png_path = os.path.join(views_dir, f"{scene_name}_{toy}.png")
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"png encode失敗: {png_path}")
            buf.tofile(png_path)
            view_meta.append((toy, scene_name, png_path))
        # 元に戻す（次のtoyループ・念のため）
        model.geom_rgba[gadr1] = rgba1_orig
        model.geom_rgba[gadr2] = rgba2_orig
        close_env(env)
    return views, view_meta, solve_report


# ============================================================================
# 両側引き算（2026-08-22新規実装）
# ============================================================================
def two_sided_scores(imagine, views, view_keys, model_specs):
    """語側・視点側それぞれの平均を引いてからcosを取る。

    語側：w' = w - mean(その model の全語のw)
    視点側：v' = v - mean(全視点・全おもちゃのv)（1回だけ・model非依存）

    戻り値: (rows, v_prime辞書{(toy,scene_key,off または toy,scene_name): np.ndarray},
             w_prime辞書{(model,word): np.ndarray or None})
    """
    v_all = np.stack([views[k] for k in view_keys], axis=0)
    v_mean = v_all.mean(axis=0)
    v_prime = {k: (views[k] - v_mean) for k in view_keys}

    w_prime = {}
    for model_name, _mp in model_specs:
        word_vecs = [imagine[(model_name, w)] for w in ANSWER
                     if imagine[(model_name, w)] is not None]
        if word_vecs:
            w_mean = np.mean(np.stack(word_vecs, axis=0), axis=0)
        else:
            w_mean = None
        for w in ANSWER:
            vec = imagine[(model_name, w)]
            w_prime[(model_name, w)] = None if (vec is None or w_mean is None) \
                else (vec - w_mean)
    return v_prime, w_prime


def score_one(w_prime_vec, v1_prime, v2_prime, correct_toy):
    if w_prime_vec is None:
        return float("nan"), float("nan"), "N/A", False, float("nan")
    s1 = _cosine_sim(w_prime_vec, v1_prime)
    s2 = _cosine_sim(w_prime_vec, v2_prime)
    if s1 == s2:
        answer = "tie"
    else:
        answer = "toy1" if s1 > s2 else "toy2"
    correct = (answer == correct_toy)
    s_correct = s1 if correct_toy == "toy1" else s2
    s_wrong = s2 if correct_toy == "toy1" else s1
    diff = s_correct - s_wrong
    return s1, s2, answer, correct, diff


# ============================================================================
# 副次①：検索方式（AUC・上位10枚・上位5枚の図）
# ============================================================================
def auc_from_scores(pos_scores, neg_scores):
    """Mann-Whitney U 式のAUC（ROC-AUC）。scipy.stats.rankdata でタイを平均順位に。"""
    from scipy.stats import rankdata
    all_scores = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(all_scores)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    rank_sum_pos = ranks[:n_pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binom_threshold(n, alpha=0.05, p=0.5):
    """片側二項検定でp<alphaとなる最小の正答数k（参考情報として出力するだけ。
    合否判断はしない＝仕様書の指示）。"""
    from scipy.stats import binom
    for k in range(n, -1, -1):
        if binom.sf(k - 1, n, p) < alpha:
            continue
        return k + 1
    return 0


# ============================================================================
# メイン
# ============================================================================
def main():
    args = parse_args()
    TARO_BASE["lexicon_vision"]["fovea_px"] = int(args.fovea_px)
    # 【2026-08-23・F1-9C】学習時のバックエンドと一致させる（上のコメント参照）。
    TARO_BASE["lexicon_vision"]["backend"] = str(args.backend)
    if str(args.backend) == "vits14_untrained":
        TARO_BASE["lexicon_vision"]["seed"] = int(args.backend_seed)
    print(f"[視覚バックエンド] {TARO_BASE['lexicon_vision']}")
    e_dir = _resolve(args.e_dir) if args.e_dir else _E_DIR_DEFAULT
    views_dir = _resolve(args.views_dir) if args.views_dir else _VIEWS_DIR_DEFAULT
    report_path = _resolve(args.report) if args.report else _REPORT_PATH_DEFAULT
    csv_path = os.path.join(e_dir, "明細.csv")
    npz_path = os.path.join(e_dir, "embeddings.npz")
    tag = args.tag

    matrix_mode = args.scenes_dir is not None

    if args.models:
        model_specs = []
        for spec in args.models:
            label, _, path = spec.partition("=")
            mp = None if path in ("", "None", "none") else path
            model_specs.append((label, mp))
    else:
        model_specs = list(DEFAULT_MODEL_SPECS)

    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(e_dir, exist_ok=True)

    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say(f"モード: {'scenes-dir(新・F1-6)' if matrix_mode else 'legacy(A/B・同値性確認用)'}")
    say(f"判定対象モデル一覧: {model_specs}")
    say(f"出力先: e_dir={e_dir}  views_dir={views_dir}  report={report_path}")

    # ---- 1) held-out視点の採取 ----------------------------------------------
    if matrix_mode:
        scene_paths = sorted(
            os.path.join(args.scenes_dir, f) for f in os.listdir(args.scenes_dir)
            if f.endswith(".json"))
        say(f"scenes-dir: {args.scenes_dir}  場面数={len(scene_paths)}")
        views, view_meta, solve_report = capture_views_isolated(
            scene_paths, args.fovea_px, views_dir, say)
        view_keys = list(views.keys())
        scene_names = [os.path.splitext(os.path.basename(p))[0] for p in scene_paths]
    else:
        if args.scenes:
            scenes = {chr(ord("A") + i): s for i, s in enumerate(args.scenes)}
        else:
            scenes = dict(DEFAULT_SCENES)
        offsets = args.offsets if args.offsets is not None else OFFSETS_DEG
        say(f"シーン: {scenes}")
        say(f"オフセット: {offsets}")
        views, view_meta, solve_report = capture_views_legacy(
            scenes, offsets, args.fovea_px, views_dir, say)
        view_keys = list(views.keys())
        scene_names = None

    # ---- 2) 想像の見え埋め込み（判定対象モデルごと） -------------------------
    say("\n=== 想像embeddingの取得 ===")
    imagine = {}
    chunks = {}
    if matrix_mode:
        _first_scene = scene_paths[0]
    else:
        _first_scene = next(iter(scenes.values()))
    for model_name, mp in model_specs:
        with_model = mp is not None
        tr = build_trainer(_first_scene, with_model=with_model, model_path=mp, seed=0)
        taro = tr.taro
        say(f"  [{model_name}] hearing.vocab.size={taro.hearing.vocab.size} "
            f"lexicon語彙数={len(taro.lexicon.counts)}")
        for word in ANSWER:
            vec, chunk = imagine_embedding(taro, word)
            imagine[(model_name, word)] = vec
            chunks[(model_name, word)] = chunk
            if vec is None:
                say(f"    {model_name}/{word}: chunk={chunk} → 想像embeddingなし")
            else:
                say(f"    {model_name}/{word}: chunk={chunk} → embedding取得"
                    f"(dim={vec.shape[0]}, norm={np.linalg.norm(vec):.4f})")
        close_env(tr.env)

    # ---- 3) 両側引き算 → 判定・明細CSV ----------------------------------------
    say("\n=== 両側引き算による判定 ===")
    v_prime, w_prime = two_sided_scores(imagine, views, view_keys, model_specs)

    rows = []
    if matrix_mode:
        for model_name, _mp in model_specs:
            for scene_name in scene_names:
                v1p = v_prime[("toy1", scene_name)]
                v2p = v_prime[("toy2", scene_name)]
                for word, correct_toy in ANSWER.items():
                    wp = w_prime[(model_name, word)]
                    s1, s2, answer, correct, diff = score_one(wp, v1p, v2p, correct_toy)
                    rows.append(dict(model=model_name, scene=scene_name, word=word,
                                     sim_toy1=s1, sim_toy2=s2, answer=answer,
                                     correct=correct, diff=diff))
    else:
        for model_name, _mp in model_specs:
            for scene_key in scenes:
                for off in offsets:
                    v1p = v_prime[("toy1", scene_key, off)]
                    v2p = v_prime[("toy2", scene_key, off)]
                    scene_label = f"{scene_key}_{fmt_offset(off)}"
                    for word, correct_toy in ANSWER.items():
                        wp = w_prime[(model_name, word)]
                        s1, s2, answer, correct, diff = score_one(wp, v1p, v2p, correct_toy)
                        rows.append(dict(model=model_name, scene=scene_label, word=word,
                                         sim_toy1=s1, sim_toy2=s2, answer=answer,
                                         correct=correct, diff=diff))

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["model", "scene", "word",
                                           "sim_toy1", "sim_toy2", "answer",
                                           "correct", "diff"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    say(f"明細CSV: {csv_path} ({len(rows)}行)")

    # ---- 4) embeddings.npz -----------------------------------------------------
    npz_data = {}
    for k, vec in views.items():
        key = "_".join(str(x) for x in k)
        npz_data[f"view_{key}"] = vec
    for (model_name, word), vec in imagine.items():
        wk = WORD_KEY[word]
        npz_data[f"imagine_{model_name}_{wk}"] = (
            vec if vec is not None else np.full(384, np.nan))
    np.savez(npz_path, **npz_data)
    say(f"embeddings.npz: {npz_path} ({len(npz_data)}キー)")

    # ---- 5) 集計（主判定=わんわん語のみ。ぶーぶーは数えない・仕様書の指示） --------
    def primary_rows(model_name):
        return [r for r in rows if r["model"] == model_name and r["word"] == PRIMARY_WORD]

    say(f"\n=== 正答数（主判定・{PRIMARY_WORD}のみ） ===")
    summary = {}
    n_scenes = len(scene_names) if matrix_mode else len(scenes) * len(offsets)
    for model_name, _mp in model_specs:
        pr = primary_rows(model_name)
        c = sum(1 for r in pr if r["correct"])
        n = len(pr)
        summary[model_name] = (c, n)
        say(f"  {model_name}: {c}/{n}")
    if matrix_mode and n_scenes > 0:
        k5 = binom_threshold(n_scenes)
        say(f"\n（参考・合否は判断しない）片側二項検定p<0.05の最小正答数: "
            f"{k5}/{n_scenes}（n={n_scenes}）")

    # ---- 6) 図（モード別） ------------------------------------------------------
    fig1_path = fig2_path = None
    auc_report = {}
    if matrix_mode:
        say("\n=== 図1：場面×シードの正誤マップ ===")
        fig1_path = _resolve(args.fig1) if args.fig1 else os.path.join(
            _F_DIR_DEFAULT, "図1_場面別正誤マップ.png")
        model_names = [m for m, _ in model_specs]
        grid = np.zeros((len(model_names), len(scene_names)))
        for i, model_name in enumerate(model_names):
            pr = {r["scene"]: r["correct"] for r in primary_rows(model_name)}
            for j, sn in enumerate(scene_names):
                grid[i, j] = 1.0 if pr.get(sn, False) else 0.0
        fig, ax = plt.subplots(figsize=(max(8, len(scene_names) * 0.3), 3))
        ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel(f"場面（{len(scene_names)}場面）")
        ax.set_title(f"F1-6 場面別正誤マップ（{PRIMARY_WORD}のみ・緑=正解／赤=不正解）")
        if len(scene_names) <= 40:
            ax.set_xticks(range(len(scene_names)))
            ax.set_xticklabels(scene_names, rotation=90, fontsize=6)
        else:
            ax.set_xticks([])
        fig.tight_layout()
        os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
        fig.savefig(fig1_path, dpi=150)
        plt.close(fig)
        say(f"図1: {fig1_path}")

        say("\n=== 副次①：検索方式（AUC・上位10・上位5枚の図） ===")
        fig2_path = _resolve(args.fig2) if args.fig2 else os.path.join(
            _F_DIR_DEFAULT, "図2_副次1_上位5枚.png")
        model_names = [m for m, _ in model_specs]
        fig, axes = plt.subplots(len(model_names), 5,
                                 figsize=(5 * 2.0, len(model_names) * 2.2))
        if len(model_names) == 1:
            axes = axes[None, :]
        # 全200(実際はn_scenes*2)枚のプール：(toy, scene_name) -> v_prime, ラベル
        pool_keys = view_keys   # (toy, scene_name)
        for i, model_name in enumerate(model_names):
            wp = w_prime[(model_name, PRIMARY_WORD)]
            if wp is None:
                say(f"  [{model_name}] {PRIMARY_WORD}の想像embeddingが無いためスキップ")
                for j in range(5):
                    axes[i, j].axis("off")
                continue
            scored = []
            for k in pool_keys:
                toy = k[0]
                sim = _cosine_sim(wp, v_prime[k])
                scored.append((sim, toy, k))
            scored.sort(key=lambda t: -t[0])
            pos_scores = np.array([s for s, toy, _ in scored if toy == "toy2"])
            neg_scores = np.array([s for s, toy, _ in scored if toy == "toy1"])
            auc = auc_from_scores(pos_scores, neg_scores) if len(pos_scores) and len(neg_scores) else float("nan")
            top10 = scored[:10]
            top10_toy2 = sum(1 for _, toy, _ in top10 if toy == "toy2")
            auc_report[model_name] = dict(auc=auc, top10_toy2=top10_toy2,
                                          n_pos=len(pos_scores), n_neg=len(neg_scores))
            say(f"  [{model_name}] AUC(球=正)={auc:.4f}  上位10枚中の球={top10_toy2}/10"
                f"  (プール: 球{len(pos_scores)}枚/箱{len(neg_scores)}枚)")
            top5 = scored[:5]
            meta_by_key = {(m[0], m[1]): m[2] for m in view_meta}
            for j in range(5):
                axes[i, j].axis("off")
                if j >= len(top5):
                    continue
                sim, toy, k = top5[j]
                png_path = meta_by_key.get((toy, k[1]))
                if png_path and os.path.exists(png_path):
                    # 【罠】cv2.imread はWindowsで非ASCIIパス（日本語フォルダ名）だと
                    #   例外を出さずに黙って失敗する（他所のcv2.imwriteと同じ罠、
                    #   実測で発覚）。np.fromfile→cv2.imdecode で回避する。
                    buf = np.fromfile(png_path, dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    axes[i, j].imshow(img)
                label_toy = "球" if toy == "toy2" else "箱"
                axes[i, j].set_title(f"{label_toy} sim={sim:.3f}", fontsize=8)
                if j == 0:
                    axes[i, j].set_ylabel(model_name, fontsize=9)
        fig.suptitle(f"F1-6 副次①：「{PRIMARY_WORD}」の想像に近い上位5枚")
        fig.tight_layout()
        os.makedirs(os.path.dirname(fig2_path), exist_ok=True)
        fig.savefig(fig2_path, dpi=150)
        plt.close(fig)
        say(f"図2: {fig2_path}")
    else:
        say("\n（legacy/同値性確認モードでは図1・図2は作らない）")

    # ---- 7) レポート -----------------------------------------------------------
    lines = []
    lines.append(f"# F1-4/F1-6 直接読み出しテスト 結果{f'（{tag}）' if tag else ''}\n")
    lines.append("モデル: " + "、".join(
        f"{mn}=`{mp}`" if mp else f"{mn}=白紙" for mn, mp in model_specs) + "\n")
    lines.append(f"モード: {'scenes-dir(新)' if matrix_mode else 'legacy(A/B)'}\n")
    lines.append(f"## 正答数（主判定・{PRIMARY_WORD}のみ）\n")
    lines.append("| モデル | 正答数/場面数 |")
    lines.append("|---|---|")
    for model_name, mp in model_specs:
        c, n = summary[model_name]
        label = "白紙" if mp is None else model_name
        lines.append(f"| {label} | {c}/{n} |")
    lines.append("")
    if matrix_mode:
        lines.append("## 副次①：検索方式（AUC・上位10枚）\n")
        for model_name, d in auc_report.items():
            lines.append(f"- {model_name}: AUC={d['auc']:.4f}  "
                         f"上位10枚中の球={d['top10_toy2']}/10  "
                         f"プール球{d['n_pos']}枚/箱{d['n_neg']}枚")
        lines.append("")
    lines.append("## 作ったファイル\n")
    lines.append(f"- 明細CSV: `{csv_path}`")
    lines.append(f"- 埋め込みnpz: `{npz_path}`")
    if fig1_path:
        lines.append(f"- 図1: `{fig1_path}`")
    if fig2_path:
        lines.append(f"- 図2: `{fig2_path}`")
    lines.append(f"- 視点画像(フォビア切り出し、eye_left): `{views_dir}\\` 配下 "
                 f"{len(view_meta)}枚")
    lines.append("")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    say(f"\nレポート: {report_path}")

    return summary


if __name__ == "__main__":
    main()
