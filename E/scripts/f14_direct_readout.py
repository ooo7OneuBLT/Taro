"""F1-4 直接読み出しテスト（2026-08-21）。

語（わんわん／ぶーぶー）を聞かせ、連合器(Lexicon)が出す「想像の見え埋め込み」を
取り出し、held-out視点（学習に使っていない視点）で撮った2つのおもちゃの実物の
見え埋め込みと照合して、正解（実物）を選べるかを、学習済みモデル・白紙モデルの
両方で測る。

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
  - オフセット±1.5°・±3°は、中心解に対して**眼球水平関節の角度に直接加算**する
    （E/scripts/e_eye_camera_axes.py が関節角度を直接動かして視線のズレを測る、
    という前例と同じやり方。ヒントで言及された F1-4e_視線物差し点検 の
    スクリプト本体はリポジトリ中に見当たらなかった＝過去セッションのscratchpadに
    あったとみられ、本スクリプトでは独立に作り直した。垂直方向は中心解のまま固定）。

書き込み先（指示された範囲のみ）：
  このファイル自身／scratchpad／
  F/logs/F1-4_直接読み出し_2026-08-21/ ／ E/logs/F1-4_直接読み出し/ ／
  E/logs配下の新実験ログ（--e-dir で指定した先。継続依頼で許可）

実行:
    .venv\\Scripts\\python.exe -X utf8 E/scripts/f14_direct_readout.py
    .venv\\Scripts\\python.exe -X utf8 E/scripts/f14_direct_readout.py ^
        --model E/models/F1-4_語彙_延長_2026-08-21.pt ^
        --e-dir E/logs/F1-4_直接読み出し_延長 ^
        --fig F/logs/F1-4_直接読み出し_2026-08-21/延長_埋め込み地図.png ^
        --report scratchpad/直接読み出しレポート_延長.md --tag 延長

【2026-08-21追記・継続依頼】モデルパス・出力先をCLI引数化した（コーディネーターの
継続依頼「モデルパスが引数化されていなければ、スクリプトのモデルパス定数の変更だけは
許可する」を受け、定数のみの書き換えでなく引数化にした＝既定値なしで実行すれば前回と
完全に同じ経路・同じファイルに書く。図のPCAも改善：視点embeddingだけでPCAをfitし、
想像の星はそのfit済み空間へ射影する（前回は想像2点が分散を支配して視点の島が潰れて
いた）。
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
# 設定（E/experiments/F1-4h_本走行A.json の taro 欄をそのまま踏襲）
# ============================================================================
# 既定値（引数を何も渡さなければ前回と完全に同じ経路・同じファイルに書く）
DEFAULT_MODEL_PATH = "E/models/F1-4h_語彙_2026-08-21.pt"
DEFAULT_SCENES = {
    "A": "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21",
    "B": "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行B_2026-08-21",
}
TOY_BODY = {"toy1": "test_object1", "toy2": "test_object2"}
# parent_labeling.utterances（両シーンとも共通）：toy1="ぶーぶー", toy2="わんわん"
WORD_KEY = {"わんわん": "wanwan", "ぶーぶー": "boobu"}
ANSWER = {"わんわん": "toy2", "ぶーぶー": "toy1"}
OFFSETS_DEG = [-3.0, -1.5, 0.0, 1.5, 3.0]

TARO_BASE = dict(
    actuation="muscle", age_months=6.0, orienting_reflex=True,
    hearing=True, lexicon_vision={"backend": "dinov2_vits14", "fovea_px": 64},
    word_attention={"active_sec": 3.0, "enabled": True}, lr=0.0,
)

_F_DIR_DEFAULT = os.path.join(_ROOT, "F", "logs", "F1-4_直接読み出し_2026-08-21")
_VIEWS_DIR_DEFAULT = os.path.join(_F_DIR_DEFAULT, "views")
_E_DIR_DEFAULT = os.path.join(_ROOT, "E", "logs", "F1-4_直接読み出し")
_FIG_PATH_DEFAULT = os.path.join(_F_DIR_DEFAULT, "埋め込み地図.png")
_SCRATCHPAD_DIR = os.path.join(
    "C:\\", "Users", "syun5", "AppData", "Local", "Temp", "claude", "C--claude-AI",
    "5f32acd6-3b81-422c-af11-c7a38e83ba47", "scratchpad")
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
    ap = argparse.ArgumentParser(description="F1-4 直接読み出しテスト")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                     help="読み出しテストに使う学習済みモデルのパス（trained側。"
                          f"既定: {DEFAULT_MODEL_PATH}）")
    ap.add_argument("--e-dir", default=None,
                     help="明細CSV/embeddings.npzの出力先ディレクトリ"
                          f"（既定: {_E_DIR_DEFAULT}）")
    ap.add_argument("--fig", default=None,
                     help=f"埋め込み地図PNGの出力先（既定: {_FIG_PATH_DEFAULT}）")
    ap.add_argument("--views-dir", default=None,
                     help=f"視点PNGの出力先ディレクトリ（既定: {_VIEWS_DIR_DEFAULT}。"
                          "視覚バックエンドはモデルの重みに依存しないフリーズ済み"
                          "DINOv2なので既定のままで内容は再現される）")
    ap.add_argument("--report", default=None,
                     help=f"レポートmdの出力先（既定: {_REPORT_PATH_DEFAULT}）")
    ap.add_argument("--fovea-px", type=int, default=64,
                    help="視覚エンコードのフォビア窓[px]（学習時の設定と一致させること。"
                         "既定64=従来動作。F1-5フォビア32検証用に追加 2026-08-21）")
    ap.add_argument("--tag", default="",
                     help="ログ・レポート見出しに付ける表示用ラベル（例: 延長）")
    ap.add_argument("--scenes", nargs="+", default=None,
                     help="視点採取に使うシーン名（1個以上、スペース区切り）。"
                          "既定は従来の2シーン（±1.5/3度×2シーン＝10視点/おもちゃ）。"
                          "1個だけ渡すと5視点/おもちゃになる（2026-08-21・学習場面での"
                          "切り分け依頼で追加。既定値なしなら従来動作のまま）")
    ap.add_argument("--models", nargs="+", default=None,
                     help="判定対象モデルを label=path の形で複数指定（例: "
                          "orig=E/models/F1-4h_語彙_2026-08-21.pt "
                          "ext=E/models/F1-4_語彙_延長_2026-08-21.pt）。"
                          "path省略時または'None'で白紙モデル扱い。既定は従来どおり"
                          "trained(--modelの値)+blankの2本（2026-08-21・複数モデル比較の"
                          "依頼で追加。--model単体は引き続きそのまま使える）")
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
    """眼球水平・垂直関節の角度[度]を、視線が target_pos を向くように数値Newton法で解く。

    既存コードに「狙った方向を向かせる」関数が無いための新規実装（仕様の対象外＝
    trainer.pyの語→想像embedding経路とは無関係の、視点採取専用の道具）。
    """
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
# メイン
# ============================================================================
def main():
    args = parse_args()
    model_path = args.model
    # --fovea-px：テスト側の視点エンコードを学習時の窓幅に合わせる（不一致だと
    #   照合が壊れる）。既定64は従来と完全一致。
    TARO_BASE["lexicon_vision"]["fovea_px"] = int(args.fovea_px)
    e_dir = _resolve(args.e_dir) if args.e_dir else _E_DIR_DEFAULT
    fig_path = _resolve(args.fig) if args.fig else _FIG_PATH_DEFAULT
    views_dir = _resolve(args.views_dir) if args.views_dir else _VIEWS_DIR_DEFAULT
    report_path = _resolve(args.report) if args.report else _REPORT_PATH_DEFAULT
    csv_path = os.path.join(e_dir, "明細.csv")
    npz_path = os.path.join(e_dir, "embeddings.npz")
    tag = args.tag

    if args.scenes:
        scenes = {chr(ord("A") + i): s for i, s in enumerate(args.scenes)}
    else:
        scenes = dict(DEFAULT_SCENES)

    if args.models:
        model_specs = []
        for spec in args.models:
            label, _, path = spec.partition("=")
            mp = None if path in ("", "None", "none") else path
            model_specs.append((label, mp))
    else:
        model_specs = [("trained", model_path), ("blank", None)]

    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(e_dir, exist_ok=True)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    log = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    say(f"モデル: {model_path}")
    say(f"シーン: {scenes}")
    say(f"判定対象モデル一覧: {model_specs}")
    say(f"出力先: e_dir={e_dir}  fig={fig_path}  views_dir={views_dir}  report={report_path}")

    # ---- 1) held-out視点の採取（視点採取専用の太郎＝白紙構成で十分。
    #      DINOv2バックエンドはフリーズ済みでモデルの重みに依存しないため、
    #      trained/blank どちらの taro で撮っても同じ埋め込みになる） -----------
    views = {}          # (toy, scene_key, off) -> np.ndarray(384,)
    view_meta = []       # 明細用の行の元データ
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
            for off in OFFSETS_DEG:
                set_eye_deg(data, addrs, h0 + off, v0)
                mujoco.mj_forward(model, data)
                imgs = get_vision(env)
                vec = np.asarray(taro.vision_backend.encode(
                    imgs["eye_left"], imgs["eye_right"]), dtype=np.float64)
                views[(toy, scene_key, off)] = vec
                crop = fovea_crop(imgs["eye_left"], 64)
                png_path = os.path.join(views_dir, f"{toy}_{scene_key}_{fmt_offset(off)}.png")
                # 【罠】cv2.imwrite はWindowsで非ASCIIパス（日本語フォルダ名）だと
                #   例外を出さずに黙って失敗する（戻り値Falseのみ、実測で発覚）。
                #   imencode→Python側でファイル書き込みにすれば回避できる。
                ok, buf = cv2.imencode(".png", cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f"png encode失敗: {png_path}")
                buf.tofile(png_path)
                view_meta.append((toy, scene_key, off, png_path))
        close_env(env)

    # ---- 2) 想像の見え埋め込み（判定対象モデルごと） -------------------------
    say("\n=== 想像embeddingの取得 ===")
    imagine = {}   # (model_name, word) -> vec or None
    chunks = {}
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
                say(f"    {model_name}/{word}: chunk={chunk} → 想像embeddingなし"
                    f"（lexicon.state_sumにこの語の蓄積が無い）")
            else:
                say(f"    {model_name}/{word}: chunk={chunk} → embedding取得"
                    f"(dim={vec.shape[0]}, norm={np.linalg.norm(vec):.4f})")
        close_env(tr.env)

    # ---- 3) 判定・明細CSV ----------------------------------------------------
    say("\n=== 判定 ===")
    rows = []
    for model_name, _mp in model_specs:
        for word, correct_toy in ANSWER.items():
            vec = imagine[(model_name, word)]
            for scene_key in scenes:
                for off in OFFSETS_DEG:
                    v1 = views[("toy1", scene_key, off)]
                    v2 = views[("toy2", scene_key, off)]
                    if vec is None:
                        s1 = s2 = float("nan")
                        answer = "N/A"
                        correct = False
                    else:
                        s1 = _cosine_sim(vec, v1)
                        s2 = _cosine_sim(vec, v2)
                        if s1 == s2:
                            answer = "tie"
                        else:
                            answer = "toy1" if s1 > s2 else "toy2"
                        correct = (answer == correct_toy)
                    view_pair = f"{scene_key}_{fmt_offset(off)}"
                    rows.append(dict(model=model_name, word=word, view_pair=view_pair,
                                     sim_toy1=s1, sim_toy2=s2, answer=answer,
                                     correct=correct))

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["model", "word", "view_pair",
                                           "sim_toy1", "sim_toy2", "answer", "correct"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    say(f"明細CSV: {csv_path} ({len(rows)}行)")

    # ---- 4) embeddings.npz（明細と対応するキー。wordは英字化：{WORD_KEY}） --------
    npz_data = {}
    for (toy, scene_key, off), vec in views.items():
        npz_data[f"view_{toy}_{scene_key}_{fmt_offset(off)}"] = vec
    for (model_name, word), vec in imagine.items():
        wk = WORD_KEY[word]
        npz_data[f"imagine_{model_name}_{wk}"] = (
            vec if vec is not None else np.full(384, np.nan))
    np.savez(npz_path, **npz_data)
    say(f"embeddings.npz: {npz_path} ({len(npz_data)}キー)")

    # ---- 5) 集計 --------------------------------------------------------------
    def count_correct(model_name, word=None):
        sub = [r for r in rows if r["model"] == model_name
               and (word is None or r["word"] == word)]
        return sum(1 for r in sub if r["correct"]), len(sub)

    summary = {}
    for model_name, _mp in model_specs:
        c, n = count_correct(model_name)
        summary[model_name] = (c, n)
        for word in ANSWER:
            summary[(model_name, word)] = count_correct(model_name, word)

    say("\n=== 正答数 ===")
    for model_name, _mp in model_specs:
        c, n = summary[model_name]
        say(f"  {model_name}: {c}/{n}")
        for word in ANSWER:
            cw, nw = summary[(model_name, word)]
            say(f"    {word}: {cw}/{nw}")

    # ---- 6) 図：PCAで2次元に落として1枚 ---------------------------------------
    #   【2026-08-21改善・継続依頼】PCAは「視点embeddingだけ」でfitし、想像の星は
    #   そのfit済み空間へ射影する（想像2点を分散計算に混ぜると、視点同士より遠い
    #   ぶん分散を支配してしまい、視点の島が潰れて見える＝前回の図の問題点）。
    say("\n=== 図の作成 ===")
    view_keys = list(views.keys())
    Xv = np.stack([views[k] for k in view_keys], axis=0)
    mean = Xv.mean(axis=0, keepdims=True)
    Xvc = Xv - mean
    _U, _S, Vt = np.linalg.svd(Xvc, full_matrices=False)
    components = Vt[:2]                       # (2, dim)：視点embeddingだけでfit

    def project(vec):
        return (vec - mean.ravel()) @ components.T

    keys = [("view",) + k for k in view_keys]
    P_list = [project(views[k]) for k in view_keys]
    for (model_name, word), v in imagine.items():
        if v is not None:
            keys.append(("imagine", model_name, word))
            P_list.append(project(v))
    P = np.stack(P_list, axis=0)

    # モデルごとにマーカーを割り当てる（blank=mp is Noneは既定どおり×固定。
    #   それ以外の「実モデル」は登場順に★▲■...を割り当てる＝2本以上の実モデルを
    #   同じ図で比較できるようにする2026-08-21拡張）。
    _real_markers = ["*", "^", "s", "P", "D", "v"]
    _marker_for_model = {}
    _next_real = 0
    for _mn, _mp in model_specs:
        if _mp is None:
            _marker_for_model[_mn] = "x"
        else:
            _marker_for_model[_mn] = _real_markers[_next_real % len(_real_markers)]
            _next_real += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, key in enumerate(keys):
        if key[0] == "view":
            _tag, toy, scene_key, off = key
            color = "#c0392b" if toy == "toy1" else "#7f8c8d"
            ax.scatter(P[i, 0], P[i, 1], c=color, s=40, alpha=0.75,
                      marker="o", edgecolors="none")
        else:
            _tag, model_name, word = key
            color = "#c0392b" if ANSWER[word] == "toy1" else "#7f8c8d"
            marker = _marker_for_model.get(model_name, "*")
            size = 200 if marker == "x" else 380
            ax.scatter(P[i, 0], P[i, 1], c=color, s=size, marker=marker,
                      edgecolors="black" if marker != "x" else color,
                      linewidths=1.2, zorder=5)
            right_half = P[i, 0] > (P[:, 0].min() + P[:, 0].max()) / 2
            ax.annotate(f"{model_name}:{word}", (P[i, 0], P[i, 1]),
                        textcoords="offset points",
                        xytext=(-8, 6) if right_half else (6, 6),
                        ha="right" if right_half else "left", fontsize=9)
    n_missing = sum(1 for (mn, wd) in imagine if imagine[(mn, wd)] is None)
    if n_missing:
        missing_words = [f"{mn}:{wd}" for (mn, wd) in imagine if imagine[(mn, wd)] is None]
        ax.text(0.02, 0.02, "想像embeddingなし: " + ", ".join(missing_words),
               transform=ax.transAxes, fontsize=8, color="#555555")
    ax.margins(x=0.18)
    ax.set_xlabel("PC1（視点embeddingのみでfit）")
    ax.set_ylabel("PC2（視点embeddingのみでfit）")
    title_tag = f"[{tag}] " if tag else ""
    legend_bits = "　".join(f"{m}={_marker_for_model[m]}" for m, _mp in model_specs)
    ax.set_title(f"{title_tag}F1-4 直接読み出し：見え埋め込みの地図（視点embeddingでPCA fit）\n"
                 f"丸=held-out視点（赤=toy1／灰=toy2）　想像embeddingの記号：{legend_bits}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    say(f"図: {fig_path}")

    # ---- 7) レポート -----------------------------------------------------------
    lines = []
    lines.append(f"# F1-4 直接読み出しテスト 結果{f'（{tag}）' if tag else ''}\n")
    lines.append("モデル: " + "、".join(
        f"{mn}=`{mp}`" if mp else f"{mn}=白紙" for mn, mp in model_specs) + "\n")
    lines.append(f"シーン: " + "、".join(f"{k}=`{v}`" for k, v in scenes.items()) + "\n")
    lines.append("## 正答数\n")
    lines.append("| モデル | 全体 | わんわん | ぶーぶー |")
    lines.append("|---|---|---|---|")
    for model_name, mp in model_specs:
        c, n = summary[model_name]
        cw1, nw1 = summary[(model_name, "わんわん")]
        cw2, nw2 = summary[(model_name, "ぶーぶー")]
        label = "白紙" if mp is None else model_name
        lines.append(f"| {label}({model_name}) | {c}/{n} | {cw1}/{nw1} | {cw2}/{nw2} |")
    lines.append("")
    lines.append("## 類似度の代表値（cos類似度、model毎・N/A行を除く）\n")
    for model_name, _mp in model_specs:
        m_rows = [r for r in rows if r["model"] == model_name
                  and not (isinstance(r["sim_toy1"], float) and np.isnan(r["sim_toy1"]))]
        if not m_rows:
            continue
        correct_sims = [max(r["sim_toy1"], r["sim_toy2"]) for r in m_rows]
        wrong_sims = [min(r["sim_toy1"], r["sim_toy2"]) for r in m_rows]
        lines.append(f"- [{model_name}] 選ばれた方（大きい方）: "
                     f"平均{np.mean(correct_sims):.4f} 中央値{np.median(correct_sims):.4f}"
                     f"／選ばれなかった方（小さい方）: "
                     f"平均{np.mean(wrong_sims):.4f} 中央値{np.median(wrong_sims):.4f}")
    lines.append("")
    lines.append("## 想像embeddingの有無\n")
    for (model_name, word), vec in imagine.items():
        state = "あり" if vec is not None else "なし（lexicon.state_sumに蓄積なし）"
        lines.append(f"- {model_name}/{word}: {state}  chunk={chunks[(model_name, word)]}")
    lines.append("")
    lines.append("## 眼球方向の中心解の残差角度（最大成分、solve_gazeの収束確認）\n")
    for scene_key, toy, h0, v0, resid in solve_report:
        lines.append(f"- {scene_key}/{toy}: h={h0:+.2f}° v={v0:+.2f}° 残差={resid:.4f}°")
    lines.append("")
    lines.append("## 作ったファイル\n")
    lines.append(f"- 明細CSV: `{csv_path}`")
    lines.append(f"- 埋め込みnpz: `{npz_path}`")
    lines.append(f"- 図: `{fig_path}`")
    lines.append(f"- 視点画像(フォビア切り出し、eye_left): `{views_dir}\\` 配下 "
                 f"{len(view_meta)}枚（例: {view_meta[0][3] if view_meta else '(なし)'}）")
    lines.append(f"- 新規スクリプト: `E/scripts/f14_direct_readout.py`")
    lines.append("")
    lines.append("## 流用した関数（車輪の再発明はしていない）\n")
    lines.append("- `run/trainer.py` Trainer.build()（136-190行）：env・taro構築")
    lines.append("- `taro_core/src/senses/hearing.py` Hearing.hear()："
                 "run/trainer.py:260, 306 と同一呼び出し")
    lines.append("- `taro_core/src/brain/lexicon.py` Lexicon.observe()（78-97行）："
                 "run/trainer.py:269 と同一呼び出し（state=Noneで読み出し専用）")
    lines.append("- `taro_core/src/brain/lexicon.py` Lexicon.assoc()（99-107行）："
                 "run/trainer.py:281, 311 と同一呼び出し")
    lines.append("- `taro_core/src/senses/vision_backends.py` "
                 "DINOv2VisionBackend.encode()（188-195行）："
                 "run/trainer.py:268, 310 と同一呼び出し")
    lines.append("- `run/trainer.py` _cosine_sim()（41-54行）をそのままimport")
    lines.append("")
    lines.append("## 新規実装（このスクリプト固有・止まる条件には該当しない）\n")
    lines.append("- 眼球関節qposの直接操作＋mj_forward＋"
                 "get_vision_obs()キャッシュ無効化（VISION_MIN_DT対策）")
    lines.append("- 視線を目標へ向ける2自由度Newton法の角度探索（solve_gaze）")
    lines.append("")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    say(f"\nレポート: {report_path}")

    return summary


if __name__ == "__main__":
    main()
