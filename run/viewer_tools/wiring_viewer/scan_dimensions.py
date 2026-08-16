"""太郎の各層（感覚エンコーダ5個・融合層・脳）を実際にインスタンス化し、
パラメータ数・出力次元を実測する道具（TaroMap 配線図の「実測ベース化」工程1）。

【なぜ実測か、仕様より】
`run/wiring_map.py` のNODES/EDGESは「概念上の機構名」しか持たず、量（パラメータ数・
次元）が無い。手書きの表は腐る（実例：`touch_adaptation` が2週間で表から欠落した）。
⇒ 太郎の層を**実際に組み立てて**（学習は一切回さない・1ステップも進めない）測る。

【代表値であって実測実験値ではないことに注意】
ここで作るのは「新生児相当の体を仮定した代表値」（`ProprioceptionEncoder(226)`,
`TouchEncoder(1908)` 等をデフォルト引数で直接インスタンス化）。実際に選んだ実験の
体格（実際のMuJoCo/MIMo環境の observation_space）とは多少ずれることがある
（捏造しない原則。仕様より）。

【Qt非依存の純粋関数】既存の `scan_config.py` / `scan_docstrings.py` と同じく、
QApplicationを一切構築せず呼べる。`measure_dimensions()` がこのモジュールの唯一の
公開関数。

【契約、仕様より】戻り値の形は工程2（`page_prediction.py`）がそのまま使う前提。
キー名・値の意味を変えるときは呼び出し側と合わせること。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone


def _taro_root() -> str:
    """このファイルの位置（run/viewer_tools/wiring_viewer/scan_dimensions.py）から
    Taroリポジトリのルートを逆算する（scan_docstrings.pyと同じ考え方）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))


_ROOT = _taro_root()
_CACHE_DIR = os.path.join(_ROOT, "run", "viewer_tools", "wiring_viewer", "_cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "寸法実測.json")

# 指紋（キャッシュの有効期限）の元にする依存ファイル群。仕様に列挙された7つ＋自分自身。
# 【なぜこの7つだけか】測定対象（感覚エンコーダ5個・融合層・脳）を実際に組み立てる
#   コード自身の定義ファイル。これらのどれか1つでも変わったら測り直す。
# 【なぜ自分自身（scan_dimensions.py）も入れるか、2026-08-15 実装担当の指摘で追加】
#   最初の一覧に、このファイル自身が含まれていなかった。構築引数の定数
#   （_FUSION_VISION_RES 等）や _measure_module の出力次元の推定ロジックはこの
#   ファイルの中にしかなく、それを変えても依存ファイル側は無傷なので指紋が変わらず、
#   古いキャッシュが黙って使われ続けてしまう。実際、2026-08-15 の _FUSION_VISION_RES
#   修正（64→256）はこのファイル自身の変更であり、旧一覧では検知できなかった。
_DEP_FILES_REL = [
    os.path.join("taro_core", "src", "senses", "insula.py"),
    os.path.join("taro_core", "src", "senses", "sensory_encoders.py"),
    os.path.join("taro_core", "src", "senses", "vision_encoder.py"),
    os.path.join("taro_core", "src", "senses", "fusion.py"),
    os.path.join("taro_core", "src", "senses", "somatosensory_cortex.py"),
    os.path.join("taro_core", "src", "brain", "taro_brain_motor.py"),
    os.path.join("taro_core", "src", "brain", "taro_brain.py"),
    os.path.abspath(__file__),
]

# 脳の代表値（仕様より）。sensory_dim=320=64(embedding_dim)×5感覚（全感覚ON）、
# n_actuators=90=関節モード既定と一致。vocab_size=3はrun/taro_setup.py 428行目が
# 実際に渡している値（Vocabularyの特殊トークン3つぶん）をそのまま流用。
_BRAIN_VOCAB_SIZE = 3
_BRAIN_SENSORY_DIM = 320
_BRAIN_N_ACTUATORS = 90

# 融合層の構築引数。感覚エンコーダ5個を単体測定したときと**完全に同じ**引数にする
# （image_sizeがずれるとVisionEncoderの内部Linear次元が変わり、「融合層は5感覚の合計」
# という説明と数字が食い違ってしまう。2026-08-15 工程1差し戻しで発覚・修正）。
# 【なぜ、2026-08-15】以前は vision_res=64 だった。これは試作スクリプト
# （寸法実測.py試作版の `MinimalFusion(1908, vision_res=64, proprio_dim=226)`）を
# 確認せずそのまま流用した値で、64は試作が便宜的に置いた数字にすぎず、
# taro_setup.pyの実際の運用値でも、単体測定に使ったVisionEncoder()の
# デフォルト(image_size=256)でもなかった。単体測定と揃えるため256に直した。
_FUSION_TOUCH_DIM = 1908  # 単体測定のTouchEncoder(1908)と同じ
_FUSION_VISION_RES = 256  # 単体測定のVisionEncoder()のデフォルトimage_size=256と同じ
_FUSION_PROPRIO_DIM = 226  # 単体測定のProprioceptionEncoder(226)と同じ

# 感覚エンコーダ5個の定義。(id, 日本語, 英語, 入力の説明)
# 【なぜ id が insula/prop/vest/touch/vision か】run/wiring_map.py のNODESのidと
#   完全に一致させる指定（工程2がこのidでis_on()を呼ぶ契約）。
#   注意（想定外・作業記録に記載）：run/wiring_map.py の実際のNODESでは、
#   「感覚（sense）」列のidは prop/vest/touch/vision/intero であり、"insula" id は
#   感覚列ではなく「まとめる（fuse）」列の島皮質ノードに使われている
#   （内受容感覚そのもののノードidは "intero"）。仕様の指定どおり "insula" を
#   使ったが、wiring_map.py 側の意味づけと食い違うため、工程2の担当者と実装担当へ
#   上げること（このファイルの report にも明記する）。
_SENSE_SPECS = [
    ("insula", "内受容", "insula", "体の内側の信号（空腹・眠気など）"),
    ("prop", "固有感覚", "proprioception", "関節の角度・速度（代表値：226次元）"),
    ("vest", "前庭感覚", "vestibular", "頭の傾きと回転"),
    ("touch", "触覚", "touch", "皮膚の圧力（代表値：1908点）"),
    ("vision", "視覚（両眼）", "vision", "左右の目の映像（代表値：256x256x3 x2）"),
]


def _ensure_sys_path():
    """taro_core/src の senses / brain を sys.path に足す（run/taro_setup.py 20-25行目と
    同じ考え方の部分流用。フラットな `from insula import Insula` 形式のimportを
    そのまま使うために必要）。"""
    core_src = os.path.join(_ROOT, "taro_core", "src")
    for sub in ("senses", "brain"):
        p = os.path.join(core_src, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def _compute_fingerprint() -> str:
    """依存ファイル群の "path:mtime:size" を集めてsha256する。

    【なぜmtime+sizeか、仕様より】ファイル読み込み無しで「変わったらしい」を
    素早く判定できる（内容のsha256だと全部読む必要がある）。誤検出（内容同じでも
    mtimeだけ変わる）はあるが、そちらは「安全側に倒れて測り直すだけ」で実害が無い。
    """
    parts = []
    for rel in _DEP_FILES_REL:
        abs_path = os.path.join(_ROOT, rel)
        try:
            st = os.stat(abs_path)
            mtime = st.st_mtime
            size = st.st_size
        except OSError:
            # ファイルが無い（想定外）：指紋に「無い」ことを刻んで、キャッシュが
            #   誤って有効と判定されないようにする。
            mtime, size = -1, -1
        rel_posix = rel.replace(os.sep, "/")
        parts.append(f"{rel_posix}:{mtime}:{size}")
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _load_cache():
    """キャッシュJSONを読む。壊れている・無い場合はNoneを返す（例外を外に出さない）。"""
    try:
        with open(_CACHE_PATH, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


def _save_cache(data: dict):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp_path = _CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)
    os.replace(tmp_path, _CACHE_PATH)


def _measure_module(mod) -> tuple:
    """nn.Module一つの (パラメータ数, 出力次元) を実測する。

    出力次元は「最後に登録されたnn.Linear相当（out_featuresを持つ層）」から推定する
    （寸法実測.py の試作と同じ考え方。VisionEncoderのように複数のLinearを持つ場合、
    最後（self.fuse）が実際の出力次元になる＝modules()はモジュール登録順で辿るため）。
    """
    n_params = sum(p.numel() for p in mod.parameters())
    out_dims = [m.out_features for m in mod.modules() if hasattr(m, "out_features")]
    out_dim = out_dims[-1] if out_dims else None
    return n_params, out_dim


def _efference_copy_enabled(cfg) -> bool:
    """cfg（run/config.pyのConfig、Noneなら既定を作る）からefference_copyの値だけを読む。"""
    if cfg is None:
        config_mod = importlib.import_module("run.config")
        cfg = config_mod.Config()
    return bool(cfg.efference_copy)


def _build_and_measure() -> dict:
    """太郎の各層を実際に組み立てて測る（学習は一切回さない・1ステップも進めない）。

    戻り値は measure_dimensions() の戻り値のうち、"生成時刻"/"所要秒"/"キャッシュから"/
    "指紋" を除いた部分（＝キャッシュに保存する本体）。
    """
    _ensure_sys_path()
    from insula import Insula
    from sensory_encoders import ProprioceptionEncoder, VestibularEncoder, TouchEncoder
    from vision_encoder import VisionEncoder
    from fusion import MinimalFusion
    from taro_brain_motor import TaroBrainWithMotor

    constructors = {
        "insula": lambda: Insula(),
        "prop": lambda: ProprioceptionEncoder(226),
        "vest": lambda: VestibularEncoder(),
        "touch": lambda: TouchEncoder(1908),
        "vision": lambda: VisionEncoder(),
    }

    senses = {}
    for nid, ja, en, input_desc in _SENSE_SPECS:
        mod = constructors[nid]()
        n_params, out_dim = _measure_module(mod)
        senses[nid] = {
            "日本語": ja,
            "英語": en,
            "入力の説明": input_desc,
            "出力次元": out_dim,
            "パラメータ数": n_params,
        }

    # ---- 融合層（run/taro_setup.py 393-401行目の作り方を参考にした代表値）----
    fusion = MinimalFusion(touch_dim=_FUSION_TOUCH_DIM, vision_res=_FUSION_VISION_RES,
                            proprio_dim=_FUSION_PROPRIO_DIM)
    target_fusion = MinimalFusion(touch_dim=_FUSION_TOUCH_DIM, vision_res=_FUSION_VISION_RES,
                                   proprio_dim=_FUSION_PROPRIO_DIM).freeze()
    fusion_params_train = sum(p.numel() for p in fusion.parameters())
    fusion_params_target = sum(p.numel() for p in target_fusion.parameters())
    fusion_info = {
        "統合する感覚id": ["insula", "prop", "vest", "touch", "vision"],
        "出力次元": 64 * 5,
        "パラメータ数_学習側": fusion_params_train,
        "パラメータ数_凍結正解側": fusion_params_target,
    }

    # ---- 脳 ----
    brain = TaroBrainWithMotor(vocab_size=_BRAIN_VOCAB_SIZE, sensory_dim=_BRAIN_SENSORY_DIM,
                                n_actuators=_BRAIN_N_ACTUATORS)
    brain_params = sum(p.numel() for p in brain.parameters())
    brain_info = {
        "入力次元": _BRAIN_SENSORY_DIM,
        "出力次元": _BRAIN_N_ACTUATORS,
        "パラメータ数": brain_params,
        "代表値の前提": "n_actuators=90（関節モード既定の代表値。拮抗筋モードでは実際は倍の180次元を"
                      "環境に渡すが、脳が出す行動そのものは90次元のまま）",
    }

    prediction_info = {
        "学習側": "fusion.encode(観測) → GRU(遠心性コピー) → 予測符号化の潜在変数z → 順モデルの予測",
        "正解側": "target_fusion.encode(観測).detach() → 予測の「正解」（ターゲット）",
        "なぜ分けるか": "予測側と正解側が同じ学習中の層だと「出力を平坦にすれば当たる」抜け道で"
                      "崩壊する（目標Cで実際に踏んだ、実測で確認済み）",
        "根拠ファイル": ["run/taro_setup.py:393-401", "taro_core/src/senses/fusion.py"],
    }

    return {
        "感覚": senses,
        "融合層": fusion_info,
        "脳": brain_info,
        "予測構造": prediction_info,
    }


def measure_dimensions(cfg=None, use_cache: bool = True) -> dict:
    """太郎の各層（感覚エンコーダ5個・融合層・脳）を実測して辞書で返す。

    Args:
        cfg: run/config.py の Config。省略時はデフォルトのConfig()を作る。
             "遠心性コピー"."有効か" にだけ反映される（感覚・融合層・脳の構造自体は
             cfgに依らず常に全部測る＝wiring_mapの「実装したもの全部を出す」方針と揃える）。
        use_cache: True なら run/viewer_tools/wiring_viewer/_cache/寸法実測.json の
             キャッシュを使う（依存ファイル群の指紋が一致する場合のみ）。

    Returns:
        仕様「戻り値の形」節で定義された辞書（生成時刻・所要秒・キャッシュから・指紋・
        感覚・融合層・脳・予測構造・遠心性コピー）。
    """
    fingerprint = _compute_fingerprint()
    efference_copy_enabled = _efference_copy_enabled(cfg)

    if use_cache:
        cached = _load_cache()
        if cached is not None and cached.get("指紋") == fingerprint:
            result = json.loads(json.dumps(cached))  # 浅いdeepcopy代わり（呼び出し側の書換から守る）
            result["キャッシュから"] = True
            # 遠心性コピーの「有効か」だけはcfgに依るので、キャッシュ命中でも都度上書きする。
            result["遠心性コピー"] = {
                "有効か": efference_copy_enabled,
                "説明": "直前に自分が出した行動を、次の瞬間の内部表現の推論に渡す仕組み"
                        "（z = infer_latent(いまの感覚, 直前の行動, 正解, 前回の内部状態)）",
                "根拠ファイル": ["run/trainer.py:687", "run/config.py:93"],
            }
            return result

    t0 = time.time()
    built = _build_and_measure()
    elapsed = time.time() - t0

    result = {
        "生成時刻": datetime.now(timezone.utc).astimezone().isoformat(),
        "所要秒": elapsed,
        "キャッシュから": False,
        "指紋": fingerprint,
        "感覚": built["感覚"],
        "融合層": built["融合層"],
        "脳": built["脳"],
        "予測構造": built["予測構造"],
        "遠心性コピー": {
            "有効か": efference_copy_enabled,
            "説明": "直前に自分が出した行動を、次の瞬間の内部表現の推論に渡す仕組み"
                    "（z = infer_latent(いまの感覚, 直前の行動, 正解, 前回の内部状態)）",
            "根拠ファイル": ["run/trainer.py:687", "run/config.py:93"],
        },
    }
    _save_cache(result)
    return result


if __name__ == "__main__":
    # 自己テスト：実測して要点を表示する（cp932コンソールでも落ちない記号だけ使う）。
    r1 = measure_dimensions(use_cache=False)
    print(f"[キャッシュ無し] 所要秒={r1['所要秒']:.3f} キャッシュから={r1['キャッシュから']}")
    for nid, info in r1["感覚"].items():
        print(f"  {nid}: 出力次元={info['出力次元']} パラメータ数={info['パラメータ数']}")
    print(f"  融合層 学習側={r1['融合層']['パラメータ数_学習側']} "
          f"凍結正解側={r1['融合層']['パラメータ数_凍結正解側']}")
    print(f"  脳 パラメータ数={r1['脳']['パラメータ数']}")

    r2 = measure_dimensions(use_cache=True)
    print(f"[キャッシュ有り] 所要秒={r2['所要秒']:.3f} キャッシュから={r2['キャッシュから']}")
