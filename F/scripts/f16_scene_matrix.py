"""F1-6 100場面マトリクス生成（2026-08-22、改訂版）。

設計＝`F/docs/仕様_F1-6_100場面での語彙テスト.md` 第2部（改訂2＝確定版）。

【前任者版との違い】前任者の版は角度×距離×床明るさ（10×5×2=100）だったが、
床の明るさは仕様書の「既知の事実」表で **変化0.0%＝効果ゼロと確定済み** なので
因子として使わない（仕様書どおり）。代わりに **高さ(elev_deg)** を因子に追加し、
3因子（elev_deg・dist・angle_deg）とも「5%ルール」（本ファイルの `measure_*` 系）で
実測してから水準を決める。水準を適当に等間隔で刻まない（仕様書の指示）。

【罠1: state キー】シーンJSONに `state`（qpos）があると `dist`/`angle_deg`/`elev_deg`
の変更が `run/scene_tools/scene_io.py:634 build()` の `apply_state`（`scene.get("state")`
が真のときだけ呼ばれる）で上書きされて効かない。→ 生成するJSONからは
**`state` キーを削除する**（`_strip_state` 参照）。stateが無ければ姿勢は
setup（neck_tone/limb_tone/body_support）から毎回作り直される
（`scene_io.py:634` `if scene.get("state"):` を確認済み＝キー自体が無ければ
このブロックは実行されない）。
【罠2: enabled キー】`toy2.enabled=false` で消すと、残した側の角度計算まで
無効化されて中央寄せに戻る（`e_toy_env.py`）。→ 使わない。片方を消すときは
rgbaのアルファを0にする（本ファイルでは測定用に `model.geom_rgba` を実行時に
直接書き換える。シーンJSON生成物には両方とも不透明(alpha=1)で残す＝
「1個ずつ提示」はf14_direct_readout.py側の撮影時にアルファ0で行う）。
"""
import argparse
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

BASE_SCENE_PATH = os.path.join(
    _ROOT, "run", "scenes",
    "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21.json")
OUT_DIR_DEFAULT = os.path.join(_ROOT, "run", "scenes", "F1-6_100場面_2026-08-22")
E_LOG_DIR_DEFAULT = os.path.join(_ROOT, "E", "logs", "F1-6_100場面")
TOY_BODY = {"toy1": "test_object1", "toy2": "test_object2"}

# ============================================================================
# 0) 基準シーンの読み書き
# ============================================================================
def load_base(path=BASE_SCENE_PATH):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _strip_state(sc):
    """罠1回避：state（qpos）を削除する。fingerprintも条件変更で無効になるので落とす。"""
    sc.pop("state", None)
    sc["fingerprint"] = None
    return sc


# ============================================================================
# 1) 水準の刻み方（5%ルール）の実測
# ============================================================================
# 【なぜ、2026-08-22】仕様書：「隣り合う水準どうしで5%以上変化することを条件に
#   水準を刻む」「全域でも5%未満ならその因子ごと捨てる」。適当な等間隔を禁止
#   されているので、まず候補グリッドで32px窓（フォビア窓と同じ切り出し）を撮り、
#   隣接ペアの「変化画素の割合」を実測してから、貪欲法で水準を選ぶ。
#
# 変化画素の判定閾値＝チャンネル最大絶対差 > 10（0-255階調中）。
#   【Tier3・実装側の裁量】仕様書に具体的な画素閾値の指定が無いため、
#   「5%ルールの5%」と同程度の粒度（レンダリングは決定論的＝ノイズはほぼ0のため、
#   小さい閾値でも取りこぼしは起きない一方、丸め誤差の1階調差だけを拾わないよう
#   10という小さな値で丸める）で選んだ。作業記録に明記する。
PIX_THRESH = 10
FOVEA_PX = 32

ELEV_CANDIDATES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
DIST_SCALE_CANDIDATES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                          1.05, 1.10, 1.15, 1.20, 1.25]
ANGLE_MAG_CANDIDATES = [7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]

# 他の2因子を固定するときの基準値（基準シーンの値と一致）
BASE_ELEV = 10.0
BASE_DIST_SCALE = 1.0
BASE_ANGLE_MAG = 12.0


def build_variant(base, *, elev=BASE_ELEV, dist_scale=BASE_DIST_SCALE,
                   angle_mag=BASE_ANGLE_MAG, angle_sign=1.0):
    """基準シーンをコピーし、指定した3因子の値だけ書き換えた辞書を返す（測定・生成共通）。"""
    sc = copy.deepcopy(base)
    _strip_state(sc)
    sc["world"]["toy"]["elev_deg"] = float(elev)
    sc["world"]["toy2"]["elev_deg"] = float(elev)
    base_toy_dist = base["world"]["toy"]["dist"]
    base_toy2_dist = base["world"]["toy2"]["dist"]
    sc["world"]["toy"]["dist"] = round(base_toy_dist * dist_scale, 6)
    sc["world"]["toy2"]["dist"] = round(base_toy2_dist * dist_scale, 6)
    sc["world"]["toy2"]["angle_deg"] = float(angle_sign * angle_mag)
    return sc


# 【2026-09-14】以前はここに開発機の絶対パス（Claude Codeのscratchpad）を直書きしていたが、
#   ユーザー名がリポジトリに残るうえ、そのセッションフォルダ自体が既に消えていた。
#   _ROOT基準の F/logs/_scratch に変更（F/logs/ は .gitignore 済み＝出力は公開されない）。
_SCRATCH_SCENE_DIR = os.path.join(_ROOT, "F", "logs", "_scratch", "f16_tmp_scenes")


def _write_tmp_scene(sc, tag="tmp"):
    """scene.py の build() は scene_io.load() 経由でファイルパスしか受け付けない
    （辞書を直接渡すと "シーンが見つからない" で落ちる＝実測で確認済み）ので、
    一時ファイルへ書き出してパスを渡す。"""
    os.makedirs(_SCRATCH_SCENE_DIR, exist_ok=True)
    path = os.path.join(_SCRATCH_SCENE_DIR, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(sc, fp, ensure_ascii=False)
    return path


def _capture_toy2_window(sc, fovea_px=FOVEA_PX, tag="measure"):
    """1場面を組み立て、toy1を透明化してtoy2中心へ視線を合わせ、32px窓を返す。

    測定専用（本走行の撮影ロジックとは独立）。実行時に model.geom_rgba を
    直接書き換えて透明化する＝シーンJSON上のrgbaは変えない
    （e_toy_env.py:726 `self.model.geom_rgba[gadr] = self._toy_rgba_off` と
    同じ配列を、build後にもう一段だけ上書きする形。読むだけで済み、
    e_toy_env.py 自体は変更していない）。
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import mujoco
    from run.config import Config
    from run.trainer import Trainer, close_env
    from f14_direct_readout import (eye_addrs, set_eye_deg, cam_frame,  # noqa: E402
                                     angle_err_deg, solve_gaze, get_vision)
    from vision_backends import fovea_crop  # noqa: E402

    scene_path = _write_tmp_scene(sc, tag)
    taro = dict(actuation="muscle", age_months=6.0, orienting_reflex=False,
                hearing=False, lr=0.0)
    spec = {"name": "f16_measure", "scene": scene_path, "taro": taro,
            "run": {"type": "train", "steps": 1, "seed": 0, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    env = tr.env
    try:
        model, data = env.unwrapped.model, env.unwrapped.data
        # toy1(test_object1)を透明化
        gadr1 = model.body("test_object1").geomadr[0]
        model.geom_rgba[gadr1][3] = 0.0
        mujoco.mj_forward(model, data)
        addrs = eye_addrs(model)
        target = np.array(data.body("test_object2").xpos, dtype=float).copy()
        h0, v0, _err = solve_gaze(model, data, addrs, target)
        set_eye_deg(data, addrs, h0, v0)
        mujoco.mj_forward(model, data)
        imgs = get_vision(env)
        crop = np.asarray(fovea_crop(imgs["eye_left"], fovea_px), dtype=np.int32)
        return crop
    finally:
        close_env(env)


# ============================================================================
# 1b) 宿題A：対象おもちゃ自身の見えの変化（輝度）で水準を選ぶ（F1-8新設・2026-08-22）
# ============================================================================
# 【なぜ、2026-08-22】仕様＝`F/docs/仕様_F1-8_中心窩の目での再学習と前後比較.md`
#   第2部(2-c)節。前回（上のsweep_and_capture系）は「窓全体の変化が5%以上」で
#   水準を選んだ結果、距離（見かけの大きさ）が81.5%で支配し、照明・陰影による
#   おもちゃ自身の明るさ変化を拾えなかった（水準決定.json実測：elev 5.18%・
#   angle 5.66%・dist 81.5%）。今回は「窓全体」ではなく「おもちゃ自身が占める
#   画素」だけを対象に平均輝度を測る。マスクの作り方＝仕様書の指示どおり
#   「物体を透明化した画像との差分」：同じ視線方向で(a)対象おもちゃのみ表示
#   (b)対象おもちゃも透明化＝背景のみ、の2枚を撮り、差分が閾値を超える画素を
#   対象おもちゃの領域とする。
#
#   候補範囲は前回（elev最大20度・angle最大25度）より広く振る（仕様書の指示
#   「elev_degとangle_degを広めに振って候補を作る」）。elevはsolve_gazeの
#   垂直可動域（v∈[-46,32]度、f14_direct_readout.py:solve_gaze）を踏まえ
#   0〜40度、angleは0〜45度まで振った（それ以上は乳児の視野内に収まらず
#   solve_gazeの残差が大きくなる＝実測で確認しながら選ぶ）。
HW_ELEV_CANDIDATES = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0]
HW_ANGLE_CANDIDATES = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
# 距離は水準数3までという制約（仕様書）。前回採用済みの実測（dist差が支配的＝
# 十分効くことは分かっている）ので、広い範囲から3点だけ均等に取る。
HW_DIST_LEVELS = [0.8, 1.0, 1.2]

MASK_MIN_FRAC = 0.005   # 対象おもちゃの画素占有率がこれ未満なら測定不能（画面外相当）とみなす


def _luminance_mean(img):
    """RGB画像(H,W,3)の単純平均輝度（0-255階調の(R+G+B)/3の画素平均）。"""
    import numpy as np
    a = np.asarray(img, dtype=np.float64)
    return float(a.mean())


def _capture_toy_masked_luminance(sc, toy="toy1", fovea_px=FOVEA_PX, tag="hw"):
    """対象おもちゃ(toy)自身が占める画素だけの平均輝度を測る（宿題A）。

    手順：(a)対象おもちゃだけ表示（他方は透明化）で視線を対象へ合わせて撮影、
    (b)同じ視線方向のまま対象おもちゃも透明化（=背景のみ）して撮影、
    (c) (a)-(b) の画素ごとの最大チャンネル差がPIX_THRESHを超える画素を
    対象おもちゃのマスクとし、(a)画像のそのマスク内平均輝度を返す。

    戻り値: (luminance or None, mask_frac)。mask_frac が MASK_MIN_FRAC 未満
    なら対象がほぼ画面外＝luminance は None（「測れなかった」であり0ではない、
    項78 nanは「ダメ」でなく「測れなかった」）。
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import mujoco
    from run.config import Config
    from run.trainer import Trainer, close_env
    from f14_direct_readout import eye_addrs, set_eye_deg, solve_gaze, get_vision  # noqa: E402

    scene_path = _write_tmp_scene(sc, tag)
    taro = dict(actuation="muscle", age_months=6.0, orienting_reflex=False,
                hearing=False, lr=0.0)
    spec = {"name": "f16_measure_hw", "scene": scene_path, "taro": taro,
            "run": {"type": "train", "steps": 1, "seed": 0, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    env = tr.env
    try:
        model, data = env.unwrapped.model, env.unwrapped.data
        body_name = TOY_BODY[toy]
        other = "toy2" if toy == "toy1" else "toy1"
        gadr_target = model.body(body_name).geomadr[0]
        gadr_other = model.body(TOY_BODY[other]).geomadr[0]
        rgba_target_orig = model.geom_rgba[gadr_target].copy()

        # (a) 対象のみ表示
        model.geom_rgba[gadr_other][3] = 0.0
        model.geom_rgba[gadr_target] = rgba_target_orig
        mujoco.mj_forward(model, data)
        addrs = eye_addrs(model)
        target_pos = np.array(data.body(body_name).xpos, dtype=float).copy()
        h0, v0, _err = solve_gaze(model, data, addrs, target_pos)
        set_eye_deg(data, addrs, h0, v0)
        mujoco.mj_forward(model, data)
        imgs_with = get_vision(env)
        img_with = imgs_with.get("eye_left_fovea", imgs_with["eye_left"])
        img_with = np.asarray(img_with, dtype=np.float64)

        # (b) 対象も透明化（=背景のみ）。視線方向は据え置き（再solveしない）。
        model.geom_rgba[gadr_target][3] = 0.0
        mujoco.mj_forward(model, data)
        imgs_without = get_vision(env)
        img_without = imgs_without.get("eye_left_fovea", imgs_without["eye_left"])
        img_without = np.asarray(img_without, dtype=np.float64)

        # 元に戻す処理は不要（このenvはこの関数のfinallyでclose_envされ捨てる専用）。
        diff = np.max(np.abs(img_with - img_without), axis=-1)
        mask = diff > PIX_THRESH
        mask_frac = float(np.mean(mask))
        if mask_frac < MASK_MIN_FRAC:
            return None, mask_frac
        lum = float(img_with[mask].mean())
        return lum, mask_frac
    finally:
        close_env(env)


def sweep_luminance(base, factor, candidates, toy="toy1", verbose=True):
    """1因子（elev/angle）を広い候補で振り、対象おもちゃの平均輝度を測る。"""
    out = {}
    for val in candidates:
        if factor == "elev":
            sc = build_variant(base, elev=val)
        elif factor == "angle":
            sc = build_variant(base, angle_mag=val, angle_sign=1.0)
        else:
            raise ValueError(factor)
        lum, mask_frac = _capture_toy_masked_luminance(sc, toy=toy, tag=f"hw_{factor}_{val}")
        out[val] = (lum, mask_frac)
        if verbose:
            lum_s = f"{lum:.2f}" if lum is not None else "測れず"
            print(f"  [{factor}={val}] 輝度={lum_s}  マスク占有率={mask_frac*100:.2f}%", flush=True)
    return out


def select_hw_levels(lum_dict, want_n=4):
    """輝度の実測から、最大・最小を必ず含む形でwant_n個までの水準を選ぶ。

    測れなかった(None)候補は除外して選ぶ。want_nが候補数を超える場合は
    全採用。中間はできるだけ輝度が均等に離れるように等間隔インデックスで選ぶ
    （最大/最小コントラストを優先する仕様書の指示に沿い、極端値を必ず残す）。
    """
    valid = [(v, lum) for v, (lum, _mf) in lum_dict.items() if lum is not None]
    valid.sort(key=lambda t: t[1])   # 輝度の昇順
    if len(valid) <= want_n:
        chosen = valid
    else:
        idxs = sorted(set(int(round(i * (len(valid) - 1) / (want_n - 1))) for i in range(want_n)))
        chosen = [valid[i] for i in idxs]
    return [v for v, _lum in chosen]


# 【なぜ、2026-08-22・実測結果を受けての追加】elev単独・angle単独の1次元sweepは
#   それぞれ最大/最小輝度比 1.12・1.18 止まり（1.5倍未達）だった（`sweep_luminance`
#   の実測。MIMoの光源は`benchmarkv2_scene.xml`で2灯とも`dir="0 0 -1"`固定の
#   常に真上から真下向き＝おもちゃを横や高さ方向へ動かしても入射角があまり
#   変わらないため）。**3因子を同時に動かした角（コーナー）** では、
#   (elev=0,dist_scale=1.25,angle_mag=5,sign=+1)=82.9 と
#   (elev=40,dist_scale=0.75,angle_mag=45,sign=+1)=53.2 の実測で比1.558を確認
#   （距離が近いほど窓内でおもちゃの陰影面の割合が増えて暗く写る効果が乗る）。
#   仕様書は「距離の水準数は3まで」を許しており（=距離を要因として使うこと自体は
#   禁止していない、ただし前回のように距離だけで9水準刻んで支配させないことが
#   条件）、この3因子同時使用は仕様に沿う。
HW_FINAL_ELEV_LEVELS = [0.0, 20.0, 40.0]
HW_FINAL_DIST_LEVELS = [0.75, 1.0, 1.25]
HW_FINAL_ANGLE_MAG_LEVELS = [5.0, 25.0, 45.0]


def run_hw_measurement(out_json=None, toy="toy1", verbose=True):
    """宿題A：まずelev・angle単独の広い範囲sweepで様子を見て（記録用）、
    そのあと実際に採用する3因子同時グリッドの両コーナー（最も明るくなる
    はずの組と最も暗くなるはずの組）を実測し、最大/最小比を確認する。
    E_LOG_DIR_DEFAULT/宿題A_水準決定.json へ保存。"""
    base = load_base()
    result = {"single_factor_sweep": {}}
    for factor, candidates in (("elev", HW_ELEV_CANDIDATES), ("angle", HW_ANGLE_CANDIDATES)):
        if verbose:
            print(f"\n=== 宿題A輝度測定(単独sweep・参考): {factor} "
                  f"(候補{len(candidates)}点, toy={toy}) ===", flush=True)
        lum_dict = sweep_luminance(base, factor, candidates, toy=toy, verbose=verbose)
        valid_lums = [lum for lum, _mf in lum_dict.values() if lum is not None]
        ratio = (max(valid_lums) / min(valid_lums)) if (valid_lums and min(valid_lums) > 0) else None
        result["single_factor_sweep"][factor] = dict(
            candidates=candidates,
            measured={str(v): dict(luminance=lum, mask_frac=mf) for v, (lum, mf) in lum_dict.items()},
            max_min_ratio=ratio)
        if verbose:
            print(f"  単独sweepの最大/最小輝度比={ratio}（1.5倍未達なら参考情報のみ）",
                  flush=True)

    if verbose:
        print(f"\n=== 宿題A輝度測定(3因子同時コーナー確認、toy={toy}) ===", flush=True)
    corners = {
        "bright": dict(elev=min(HW_FINAL_ELEV_LEVELS), dist_scale=max(HW_FINAL_DIST_LEVELS),
                       angle_mag=min(HW_FINAL_ANGLE_MAG_LEVELS), angle_sign=1.0),
        "dark": dict(elev=max(HW_FINAL_ELEV_LEVELS), dist_scale=min(HW_FINAL_DIST_LEVELS),
                     angle_mag=max(HW_FINAL_ANGLE_MAG_LEVELS), angle_sign=1.0),
    }
    corner_meas = {}
    for tag, kw in corners.items():
        sc = build_variant(base, **kw)
        lum, mf = _capture_toy_masked_luminance(sc, toy=toy, tag=f"hw_corner_{tag}")
        corner_meas[tag] = dict(params=kw, luminance=lum, mask_frac=mf)
        if verbose:
            lum_s = f"{lum:.2f}" if lum is not None else "測れず"
            print(f"  [{tag}] {kw} → 輝度={lum_s}  マスク占有率={mf*100:.2f}%", flush=True)
    bright_lum = corner_meas["bright"]["luminance"]
    dark_lum = corner_meas["dark"]["luminance"]
    final_ratio = (bright_lum / dark_lum) if (bright_lum and dark_lum) else None
    result["corner_check"] = corner_meas
    result["final_max_min_ratio"] = final_ratio
    result["elev"] = dict(kept_levels=HW_FINAL_ELEV_LEVELS)
    result["angle"] = dict(kept_levels=HW_FINAL_ANGLE_MAG_LEVELS)
    result["dist_levels"] = HW_FINAL_DIST_LEVELS
    result["toy_measured"] = toy
    if verbose:
        ratio_s = f"{final_ratio:.4f}" if final_ratio else "N/A"
        print(f"\n採用する3因子グリッド: elev={HW_FINAL_ELEV_LEVELS} "
              f"dist_scale={HW_FINAL_DIST_LEVELS} angle_mag={HW_FINAL_ANGLE_MAG_LEVELS}(±両側)")
        print(f"グリッド内コーナー実測の最大/最小輝度比={ratio_s}"
              f"（1.5倍以上なら宿題Aの最低条件を満たす）")
    out_json = out_json or os.path.join(E_LOG_DIR_DEFAULT, "宿題A_水準決定.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\n宿題A水準決定の記録: {out_json}", flush=True)
    return result


def generate_hw_matrix(elev_values, dist_scales, angle_values, out_dir,
                        base_path=BASE_SCENE_PATH, fovea_camera=False, verbose=True):
    """宿題A：確定した水準（輝度ベース）から総当たりで場面JSONを書き出す。

    generate_matrix()とほぼ同じだが、fovea_camera引数を追加（2-c試験は新目の
    テストにfovea_camera=true、旧モデルの対照にはfovea_camera=false、両方
    同じ物理配置で作れるようにするため）。"""
    base = load_base(base_path)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for e_idx, elev in enumerate(elev_values):
        for d_idx, dscale in enumerate(dist_scales):
            for a_idx, angle in enumerate(angle_values):
                sc = build_variant(base, elev=elev, dist_scale=dscale,
                                    angle_mag=abs(angle), angle_sign=(1.0 if angle >= 0 else -1.0))
                if fovea_camera:
                    sc["body"]["fovea_camera"] = True
                suffix = "_fovea" if fovea_camera else ""
                name = f"F1-8_hwA_e{e_idx}_d{d_idx}_a{a_idx}{suffix}"
                sc["name"] = name
                sc["note"] = (
                    "F1-8宿題A（難所入り新セット）の生成物（f16_scene_matrix.py "
                    "generate_hw_matrix、2026-08-22）。基準シーン "
                    "座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21 のコピーに、"
                    f"world.toy.elev_deg=world.toy2.elev_deg={elev}度、"
                    f"world.toy.dist={sc['world']['toy']['dist']}・"
                    f"world.toy2.dist={sc['world']['toy2']['dist']}（基準×{dscale}）、"
                    f"world.toy2.angle_deg={angle}度（toy1は自動で-{angle}度）を適用。"
                    "水準は対象おもちゃ(箱)自身のマスク平均輝度の実測で選定"
                    "（F/logs/F1-6_100場面/宿題A_水準決定.json）。"
                    + ("body.fovea_camera=trueを追加（新モデル用）。" if fovea_camera
                       else "fovea_cameraは既定false（旧モデル対照用、旧目のまま）。")
                    + "stateキーは削除済み。fingerprintも無効化。")
                sc["created"] = "2026-08-22"
                path = os.path.join(out_dir, f"{name}.json")
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(sc, fp, ensure_ascii=False, indent=2)
                written.append(path)
                if verbose:
                    print(f"書いた: {path}  elev={elev} dist_scale={dscale} angle={angle}")
    return written


def diff_frac(crop_a, crop_b, pix_thresh=PIX_THRESH):
    """2つの窓画像を比べ、閾値を超えて変化した画素の割合[0-1]を返す。"""
    import numpy as np
    a = np.asarray(crop_a, dtype=np.int32)
    b = np.asarray(crop_b, dtype=np.int32)
    diff = np.max(np.abs(a - b), axis=-1)   # (H,W)：チャンネル最大絶対差
    changed = diff > pix_thresh
    return float(np.mean(changed))


def sweep_and_capture(base, factor, candidates, verbose=True):
    """1因子を候補グリッド全域で振り、各候補の32px窓を撮って辞書で返す。"""
    crops = {}
    for val in candidates:
        if factor == "elev":
            sc = build_variant(base, elev=val)
        elif factor == "dist":
            sc = build_variant(base, dist_scale=val)
        elif factor == "angle":
            sc = build_variant(base, angle_mag=val, angle_sign=1.0)
        else:
            raise ValueError(factor)
        crop = _capture_toy2_window(sc)
        crops[val] = crop
        if verbose:
            print(f"  [{factor}] val={val} 撮影完了 (窓平均輝度={crop.mean():.1f})",
                  flush=True)
    return crops


def select_levels(candidates, crops, min_change=0.05):
    """貪欲法で水準を選ぶ：最小値から始め、直前に選んだ水準との窓差が
    min_change(既定5%)以上になる次の候補を選び続ける。

    戻り値: (選ばれた値のリスト, 隣接ペアの実測変化率のリスト,
             全域(最小↔最大)の変化率)
    """
    cands = sorted(candidates)
    kept = [cands[0]]
    pair_changes = []
    i = 1
    while i < len(cands):
        change = diff_frac(crops[kept[-1]], crops[cands[i]])
        if change >= min_change:
            kept.append(cands[i])
            pair_changes.append(change)
        i += 1
    full_range_change = diff_frac(crops[cands[0]], crops[cands[-1]])
    return kept, pair_changes, full_range_change


def run_5pct_measurement(out_json=None, verbose=True):
    """3因子すべてについて5%ルールの実測→採否→水準確定を行う。

    採否・水準は E_LOG_DIR_DEFAULT/水準決定.json に保存する。
    """
    base = load_base()
    result = {}

    for factor, candidates in (
        ("elev", ELEV_CANDIDATES),
        ("dist", DIST_SCALE_CANDIDATES),
        ("angle", ANGLE_MAG_CANDIDATES),
    ):
        if verbose:
            print(f"\n=== 因子測定: {factor} (候補{len(candidates)}点) ===", flush=True)
        crops = sweep_and_capture(base, factor, candidates, verbose=verbose)
        kept, pair_changes, full_range = select_levels(candidates, crops)
        adopt = full_range >= 0.05
        result[factor] = dict(
            candidates=candidates, kept_levels=kept,
            pair_changes=pair_changes, full_range_change=full_range,
            adopted=adopt)
        if verbose:
            print(f"  全域変化率(最小↔最大): {full_range*100:.2f}%  "
                  f"→ {'採用' if adopt else '因子ごと除外'}", flush=True)
            print(f"  選ばれた水準: {kept}", flush=True)
            print(f"  隣接ペアの変化率: {[f'{c*100:.2f}%' for c in pair_changes]}",
                  flush=True)

    # 角度の符号反転（左右入替）が5%ルールを満たすかも実測で確認する
    # （最小の大きさで確認＝一番差が付きにくいはずの条件で確かめる、が方針）。
    if verbose:
        print("\n=== 角度の符号反転（左右入替）の変化率確認 ===", flush=True)
    min_mag = min(ANGLE_MAG_CANDIDATES)
    sc_pos = build_variant(base, angle_mag=min_mag, angle_sign=1.0)
    sc_neg = build_variant(base, angle_mag=min_mag, angle_sign=-1.0)
    crop_pos = _capture_toy2_window(sc_pos)
    crop_neg = _capture_toy2_window(sc_neg)
    sign_change = diff_frac(crop_pos, crop_neg)
    result["angle_sign_flip_check"] = dict(magnitude=min_mag, change=sign_change,
                                            adopted=sign_change >= 0.05)
    if verbose:
        print(f"  大きさ{min_mag}度での符号反転の変化率: {sign_change*100:.2f}%  "
              f"→ {'採用' if sign_change >= 0.05 else '不採用（符号は水準に使わない）'}",
              flush=True)

    out_json = out_json or os.path.join(E_LOG_DIR_DEFAULT, "水準決定.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\n水準決定の記録: {out_json}", flush=True)
    return result


# ============================================================================
# 2) 場面の生成（実測で確定した水準を使う）
# ============================================================================
def all_combos(n_elev, n_dist, n_angle):
    for e in range(n_elev):
        for d in range(n_dist):
            for a in range(n_angle):
                yield e, d, a


def generate_matrix(elev_values, dist_scales, angle_values, out_dir,
                     base_path=BASE_SCENE_PATH, verbose=True):
    """確定した3因子の水準リストから総当たりで場面JSONを書き出す。"""
    base = load_base(base_path)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for e_idx, elev in enumerate(elev_values):
        for d_idx, dscale in enumerate(dist_scales):
            for a_idx, angle in enumerate(angle_values):
                sc = build_variant(base, elev=elev, dist_scale=dscale,
                                    angle_mag=abs(angle), angle_sign=(1.0 if angle >= 0 else -1.0))
                name = f"F1-6_e{e_idx}_d{d_idx}_a{a_idx}"
                sc["name"] = name
                sc["note"] = (
                    f"F1-6_100場面_2026-08-22の生成物（f16_scene_matrix.py 改訂版）。"
                    f"基準シーン 座位_6ヶ月_2おもちゃ_F1-4h_テスト12試行A_2026-08-21 の"
                    f"コピーに、world.toy.elev_deg=world.toy2.elev_deg={elev}度、"
                    f"world.toy.dist={sc['world']['toy']['dist']}・"
                    f"world.toy2.dist={sc['world']['toy2']['dist']}"
                    f"（基準×{dscale}）、world.toy2.angle_deg={angle}度"
                    f"（toy1は自動で-{angle}度）を適用。3因子は仕様書の5%ルールの"
                    f"実測で確定した水準（F/logs/F1-6_100場面/水準決定.json）。"
                    f"stateキーは削除済み（apply_stateの罠回避）。fingerprintも無効化。")
                sc["created"] = "2026-08-22"
                path = os.path.join(out_dir, f"{name}.json")
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(sc, fp, ensure_ascii=False, indent=2)
                written.append(path)
                if verbose:
                    print(f"書いた: {path}  elev={elev} dist_scale={dscale} angle={angle}")
    return written


# ============================================================================
# 3) 目視確認用レンダリング（写真A/B方式：仕様書の撮り方に合わせて2枚撮る）
# ============================================================================
def render_pair(scene_path_or_dict, out_dir, tag):
    """1場面から写真A（箱のみ）・写真B（球のみ）を撮ってPNG保存する。

    仕様書「写真の撮り方」節どおり：アルファ0で片方を透明化→中心へ視線→撮影。
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import cv2
    import mujoco
    from run.config import Config
    from run.trainer import Trainer, close_env
    from f14_direct_readout import eye_addrs, set_eye_deg, solve_gaze, get_vision  # noqa: E402

    if isinstance(scene_path_or_dict, str):
        scene_arg = scene_path_or_dict
    else:
        scene_arg = _write_tmp_scene(scene_path_or_dict, tag)

    taro = dict(actuation="muscle", age_months=6.0, orienting_reflex=False,
                hearing=False, lr=0.0)
    spec = {"name": "f16_render_check", "scene": scene_arg, "taro": taro,
            "run": {"type": "train", "steps": 1, "seed": 0, "K": 10}}
    cfg = Config.from_spec(spec)
    tr = Trainer(cfg, verbose=False)
    tr.build()
    env = tr.env
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    try:
        model, data = env.unwrapped.model, env.unwrapped.data
        gadr1 = model.body("test_object1").geomadr[0]
        gadr2 = model.body("test_object2").geomadr[0]
        rgba1_orig = model.geom_rgba[gadr1].copy()
        rgba2_orig = model.geom_rgba[gadr2].copy()
        addrs = eye_addrs(model)

        # 写真A：箱（toy1）のみ。toy2を透明化。
        model.geom_rgba[gadr1] = rgba1_orig
        model.geom_rgba[gadr2][3] = 0.0
        mujoco.mj_forward(model, data)
        target1 = np.array(data.body("test_object1").xpos, dtype=float).copy()
        h1, v1, _ = solve_gaze(model, data, addrs, target1)
        set_eye_deg(data, addrs, h1, v1)
        mujoco.mj_forward(model, data)
        imgs = get_vision(env)
        img_a = np.asarray(imgs["eye_left"])
        pa = os.path.join(out_dir, f"{tag}_写真A箱.png")
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"png encode失敗: {pa}")
        buf.tofile(pa)
        paths["A"] = pa

        # 写真B：球（toy2）のみ。toy1を透明化・toy2を戻す。
        model.geom_rgba[gadr1][3] = 0.0
        model.geom_rgba[gadr2] = rgba2_orig
        mujoco.mj_forward(model, data)
        target2 = np.array(data.body("test_object2").xpos, dtype=float).copy()
        h2, v2, _ = solve_gaze(model, data, addrs, target2)
        set_eye_deg(data, addrs, h2, v2)
        mujoco.mj_forward(model, data)
        imgs = get_vision(env)
        img_b = np.asarray(imgs["eye_left"])
        pb = os.path.join(out_dir, f"{tag}_写真B球.png")
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"png encode失敗: {pb}")
        buf.tofile(pb)
        paths["B"] = pb
    finally:
        close_env(env)
    return paths


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="F1-6 100場面マトリクス生成（改訂版）")
    ap.add_argument("--measure", action="store_true",
                     help="5%ルールの実測（水準決定）だけを行う")
    ap.add_argument("--generate", action="store_true",
                     help="水準決定.json を読み、確定した水準で場面を総当たり生成する")
    ap.add_argument("--levels-json", default=None,
                     help="--generate 用。既定: F/logs/F1-6_100場面/水準決定.json")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--render-check", nargs="*", default=None,
                     help="指定した場面JSON(複数可)について写真A/Bを撮って目視確認用に保存する")
    ap.add_argument("--render-out", default=None)
    ap.add_argument("--measure-hw", action="store_true",
                     help="宿題A：対象おもちゃ自身のマスク輝度で水準を実測する")
    ap.add_argument("--hw-toy", default="toy1",
                     help="宿題Aで輝度を測る対象（既定toy1=箱）")
    ap.add_argument("--generate-hw", action="store_true",
                     help="宿題A_水準決定.json を読み、輝度ベースの水準で場面を総当たり生成する"
                          "（fovea版・非fovea版の両方を書き出す）")
    ap.add_argument("--hw-levels-json", default=None,
                     help="--generate-hw 用。既定: F/logs/F1-6_100場面/宿題A_水準決定.json")
    ap.add_argument("--hw-out-dir-fovea", default=None,
                     help="--generate-hw 用（新モデル・fovea_camera=true版）の出力先")
    ap.add_argument("--hw-out-dir-plain", default=None,
                     help="--generate-hw 用（旧モデル対照・fovea_camera=false版）の出力先")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.measure:
        run_5pct_measurement()
        return

    if args.generate:
        levels_json = args.levels_json or os.path.join(E_LOG_DIR_DEFAULT, "水準決定.json")
        with open(levels_json, encoding="utf-8") as fp:
            result = json.load(fp)
        elev_values = result["elev"]["kept_levels"] if result["elev"]["adopted"] else None
        dist_values = result["dist"]["kept_levels"] if result["dist"]["adopted"] else None
        angle_mags = result["angle"]["kept_levels"] if result["angle"]["adopted"] else None
        if elev_values is None or dist_values is None or angle_mags is None:
            raise RuntimeError("いずれかの因子が5%ルール未達で除外されています。"
                                "水準決定.json を確認してください。")
        sign_ok = result.get("angle_sign_flip_check", {}).get("adopted", False)
        if sign_ok:
            angle_values = sorted([-m for m in angle_mags] + [m for m in angle_mags])
        else:
            angle_values = sorted(angle_mags)
        print(f"確定水準: elev={elev_values} ({len(elev_values)}) "
              f"dist_scale={dist_values} ({len(dist_values)}) "
              f"angle={angle_values} ({len(angle_values)})")
        n_total = len(elev_values) * len(dist_values) * len(angle_values)
        print(f"総場面数: {n_total}")
        generate_matrix(elev_values, dist_values, angle_values, args.out_dir)
        return

    if args.measure_hw:
        run_hw_measurement(toy=args.hw_toy)
        return

    if args.generate_hw:
        hw_levels_json = args.hw_levels_json or os.path.join(
            E_LOG_DIR_DEFAULT, "宿題A_水準決定.json")
        with open(hw_levels_json, encoding="utf-8") as fp:
            result = json.load(fp)
        elev_values = result["elev"]["kept_levels"]
        angle_mags = result["angle"]["kept_levels"]
        dist_values = result.get("dist_levels", HW_DIST_LEVELS)
        angle_values = sorted([-m for m in angle_mags] + [m for m in angle_mags])
        n_total = len(elev_values) * len(dist_values) * len(angle_values)
        print(f"宿題A確定水準: elev={elev_values} ({len(elev_values)}) "
              f"dist_scale={dist_values} ({len(dist_values)}) "
              f"angle={angle_values} ({len(angle_values)})")
        print(f"総場面数: {n_total}")
        out_fovea = args.hw_out_dir_fovea or os.path.join(
            _ROOT, "run", "scenes", "F1-8_難所入り_fovea_2026-08-22")
        out_plain = args.hw_out_dir_plain or os.path.join(
            _ROOT, "run", "scenes", "F1-8_難所入り_旧目対照_2026-08-22")
        print(f"fovea版を書き出し: {out_fovea}")
        generate_hw_matrix(elev_values, dist_values, angle_values, out_fovea,
                            fovea_camera=True)
        print(f"旧目対照版を書き出し: {out_plain}")
        generate_hw_matrix(elev_values, dist_values, angle_values, out_plain,
                            fovea_camera=False)
        return

    if args.render_check is not None:
        render_out = args.render_out or os.path.join(
            _ROOT, "F", "logs", "F1-6_100場面_2026-08-22", "目視確認")
        targets = args.render_check if args.render_check else [BASE_SCENE_PATH]
        for p in targets:
            tag = os.path.splitext(os.path.basename(p))[0]
            paths = render_pair(p, render_out, tag)
            print(f"{tag}: {paths}")
        return

    print("何もしませんでした。--measure か --generate か --render-check を指定してください。")


if __name__ == "__main__":
    main()
