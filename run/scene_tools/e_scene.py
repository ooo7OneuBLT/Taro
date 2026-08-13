"""シーン方式 — 実験環境を「1つのファイル」で決める唯一の入口。

【なぜ要るか、2026-07-29】環境の条件が4か所に散らばっていた
（コードの定数／環境変数／Viewerのプリセット／保存ファイル）。
さらに `ToySupineEnv(...)` を呼ぶファイルが **45個**あり、それぞれが独自に
「体を作る → 首を設定 → 実験者が支える」を書いていた。実際にずれていた例：

                     Viewer   e_saved_pos_check   e_own_body_in_view
    四肢の屈曲        False        True                True
    おもちゃ          True         True                False
    反射              True         False               True
    首のバネ          毎tick上書き  保存値で1回          呼ばない

`E_FLEXION` の既定は "0" なので、Viewer は False、測定側は `kw["flexion"]=True` と
手で上書きしていた。＝**Viewerで見ていた太郎と、測っていた太郎は手足の姿勢が違う体**。

注意：同じ型の事故は 2026-07-25 にも起き、`body_kwargs_from_env` はその対策だった。
  しかし**返り値を呼び出し側が上書きできる**ので、同じ罠が `flexion` で再発した。
  ⇒ 入口を1つにするだけでは足りない。**上書きできない形**にする必要がある。
  このファイルの `build()` は環境変数も呼び出し側の引数も見ず、
  **シーンの辞書だけ**から環境を組み立てる。

設計の全体は `E/docs/シーン方式_設計.md`。

【使い方】

    import e_scene
    scene = e_scene.load("リーチング_リクライニング60度")
    env, hands = e_scene.build(scene, orient=True)     # 反射のON/OFFは実験の条件
    e_scene.verify(scene, env)                          # 指紋が合わなければ止まる

【シーンに入れないもの】
反射のON/OFF・前庭動眼反射・乱数の種・測定時間は入れない。
同じシーンで反射ON/OFFを比べるのが実験だから、シーンに焼き込むと比較できなくなる。
"""
import os
import io
import sys
import json
import copy
import contextlib
import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
# 【なぜ、2026-08-05】このファイルは元々 E/scripts/ にあり、_HERE（自分のフォルダ）が
#   暗黙に e_toy_env・e_head_hold・e_visibility（いずれも E/scripts 直下）を
#   解決していた。run/scene_tools/ へ移動すると _HERE はそちらを指さなくなるため、
#   E/scripts を明示的にsys.pathへ足す必要がある（設計には無い追加対応。
#   移動後に import e_toy_env が ModuleNotFoundError になることで実際に発覚した）。
for _p in [os.path.join(_ROOT, "D", "scripts"), os.path.join(_ROOT, "MIMo"),
           os.path.join(_ROOT, "taro_core"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "E", "scripts"), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import mujoco       # noqa: E402

SCENE_DIR = os.path.join(_ROOT, "run", "scenes")
SCHEMA = 1

# 指紋の許容ずれ。ここを超えたら「別の環境」と判定して止める。
# 注意：[Tier3] 文献値ではない。物理の再現性から決めた実務的な値：
#   同じ qpos を入れて mj_forward するだけなら誤差は 1e-9 のオーダーなので、
#   下の値はどれも桁違いに緩い。それでも引っかかるなら本当に別物。
TOL = {
    "mass_kg": 1e-4,        # 体が違う
    "angle_deg": 0.5,       # 首・体幹の角度
    "pos_m": 0.002,         # 目・おもちゃの位置（2mm）
    "dir": 0.01,            # 視線の向き（単位ベクトルの差）
    "ratio": 0.02,          # 肩からの距離 ÷ 腕の長さ
}

# 姿勢が「保てる」と判定する上限。保存時の安定確認で使う。
# 注意：[Tier3・ARBITRARY] 2026-07-28 の実測（首が30度→63度に倒れた／30秒で0.52cm ずれた）
#   を踏まえ、「実験の結論が変わらない程度」として置いた目安。
SETTLE_LIMIT = {"angle_deg": 3.0, "pos_m": 0.01}


# ============================================================================
# シーンの雛形
# ============================================================================
def default_scene(name="無題"):
    """既定のシーン。ここに無い項目は build() で使われない（＝設定の全量がここ）。"""
    return {
        "name": name,
        "note": "",
        "created": datetime.date.today().isoformat(),
        "schema": SCHEMA,

        # ① 体そのもの。モデルを作るときに決まる（実行中は変えられない）
        "body": {
            "age_months": 4.0,
            "shape": True,               # 新生児体型の補正
            "shape_scales": None,        # None なら core の既定値
            "head_elongation": None,     # None なら core の既定値
            "flexion": False,            # 四肢の屈曲姿勢
            "flexion_stiffness": None,
            "limb_scale": 1.0,
            "limb_fix": True,
            "distal_mass": 1.0,
            "neck_fix": True,            # 首の筋力補正
            "eye_rest_vertical_deg": 0.0,
            # 視線誘導反射の実装バージョン（2026-07-29 追加）。
            #   注意：`e_toy_env` の既定は "1"（旧版）で、16個の測定スクリプトが
            #     それぞれ `E_ORIENT_V=2` を指定していた＝設定の散らばりそのもの。
            #     指定を忘れると**黙って旧版で走る**（実際に踏んだ）。
            #   v1 は構造的な穴が6つ見つかっている旧版。比較・アブレーション用に残す。
            "orienting_version": 2,
        },

        # ② 環境のオブジェクト。これもモデル構築時
        "world": {
            "recline_deg": 0.0,
            "seat_friction": 2.0,
            "fence": True,
            "plain": True,
            "floor_dim": 1.0,
            "static_tex": False,         # 背景のテクスチャを固定するか
            # 床の物理。注意体の滑り方が変わるので実験条件として効く
            "floor_roll": None,          # None なら環境の既定
            "floor_condim": None,
            "toy": {
                "enabled": True,
                "shape": "box",
                "radius": 0.020,
                "mode": "hold",          # hold / tether / free
                "dist": 0.086,           # 目からおもちゃまで（位置を自動で決めるとき）
                "delay_sec": 0.0,        # 0 なら最初から定位置に置く
                # 登場を遅らせるとき（delay_sec > 0）だけ効く運び方
                "approach_sec": 0.5,
                "approach_from": "above",
            },
            "parent_intervene": False,
            "parent_wait_sec": 2.0,      # 見失ってから差し出し直すまで
            "parent_lost_deg": 25.0,     # 「見失った」と判定する角度
        },

        # ③ 実験開始前の設定。モデルを作ったあとに効かせる
        "setup": {
            # 首の筋緊張（弱いバネ）。注意減衰は apply_neck_tone が臨界減衰を
            #   自分で計算するので、ここでは指定しない（文献値が無いため）。
            "neck_tone": None,           # {"target_deg":30, "stiffness":0.6}
            # 四肢の筋緊張（2026-07-29 追加）。脱力しても腕が体の前に保たれる。
            #   {"hold_deg":20, "groups":["arm","leg"]}
            #   target_deg を書かなければ**シーンの姿勢が戻る先**になる
            #   ＝Viewer で作った姿勢がそのまま筋緊張の落ち着き先。
            #   注意：乳児の四肢の筋緊張の実測値は文献に存在しない（Tier3）。
            #     詳細と間接根拠は `taro_core/src/body/infant_limbs.py` の
            #     `apply_limb_tone` 冒頭。
            "limb_tone": None,
            "head_hold": None,           # {"stiffness":200, "target_deg":{"head_tilt":60}}
            # 体を支える範囲（2026-07-29 追加）。
            #   {"stiffness":200, "free":["arm","finger"]}
            #   ＝「肩から手先」と指と眼球は自由、それ以外は椅子とベルトが支える。
            #   人間のリーチ実験は乳児を椅子に固定して行う（von Hofsten 1982）。
            #   注意：脚の固定は人間からの逸脱（実験の測定条件）。逸脱リスト参照。
            #   "pin_root": true を足すと、体そのものも空間に留める
            #   （関節のバネでは体が椅子から転がり落ちるのを止められないため）。
            "body_support": None,
        },

        # ④ 最初の姿勢。qpos が正、joints_readable は表示専用
        "state": None,

        # ⑤ 指紋（照合用）。save() が自動で作る
        "fingerprint": None,
    }


def _merge(base, over):
    """既定の雛形に、読み込んだ内容を重ねる（項目が増えても古いシーンが読める）。"""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


# ============================================================================
# 読み書き
# ============================================================================
def scene_path(name, ext=".json"):
    return os.path.join(SCENE_DIR, f"{name}{ext}")


def list_scenes():
    """使えるシーンの名前を返す。"""
    if not os.path.isdir(SCENE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SCENE_DIR) if f.endswith(".json"))


def load(name):
    """シーンを読む。名前でもファイルパスでもよい。"""
    path = name if os.path.isfile(str(name)) else scene_path(name)
    if not os.path.isfile(path):
        avail = "、".join(list_scenes()) or "（1つも無い）"
        raise FileNotFoundError(
            f"シーンが見つからない: {path}\n  使えるシーン: {avail}")
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    scene = _merge(default_scene(), raw)
    scene["_path"] = path
    return scene


def save(scene, name=None, env=None, hands=None, settle_seconds=3.0,
         image=True, verbose=True):
    """シーンを書く。env を渡すと、今の姿勢・指紋・左目の絵も一緒に保存する。

    Args:
        env: 渡すと `state`（qpos 丸ごと）と `fingerprint` を今の環境から作る
        settle_seconds: 保存前に物理を進めて「姿勢が保てるか」を確かめる秒数。
            0 なら確認しない。
    Returns:
        (保存したシーン, 安定確認の結果 or None)
    """
    scene = copy.deepcopy(scene)
    scene.pop("_path", None)
    nm = name or scene.get("name") or "無題"
    scene["name"] = nm
    scene["schema"] = SCHEMA

    drift = None
    if env is not None:
        # 注意：実験者の手を「今の角度で支える」設定にしている場合、目標角は
        #   build した直後（リセット直後の姿勢）で決まっている。そのまま保存すると
        #   **保存する姿勢と、支える目標角がずれる**。読み込み側は保存された姿勢で
        #   支え直すので、指紋が一致しなくなる（2026-07-29 の自己テストで検出。差0.13度）。
        #   ⇒ 保存する姿勢に合わせて支え直してから記録する。
        # 【2026-07-29】眼球は**基準位置に戻してから**保存する。
        #
        # 【なぜ】シーンは qpos を丸ごと記録するので、眼球の向きもそのまま残る。
        #   Viewer で姿勢を整えているあいだに反射やVORが眼球を動かしているので、
        #   保存されるのは「たまたまその瞬間に向いていた方向」になる。
        #   実測（2026-07-29）：視線誘導反射の測定で、視野の右端（+4.0cm）の対象が
        #   **移行前は見えていたのに移行後は見えなくなった**。原因は開始時の眼球が
        #   2.3秒ぶん動いた位置だったこと。
        #
        # 【人間ではどうか】定位の実験は**中心を固視した状態から**始める。
        #   Hunter & Richards (2003) は「中心を見ていない試行は除外」と明記している。
        #   ＝眼球を基準位置にそろえるのは、人間の実験の作法に合わせることでもある。
        #
        # 注意：`eye_rest_vertical_deg`（顎を引く代わりに眼球を下げる設定）は意図的な
        #   条件なので、その角度を基準として使う。
        try:
            from infant_body import center_eyes
            _u = env.unwrapped
            center_eyes(_u.model, _u.data,
                        vertical_deg=float(scene["body"]["eye_rest_vertical_deg"]))
            mujoco.mj_forward(_u.model, _u.data)
        except Exception as e:
            print(f"[scene] 注意眼球を基準位置に戻せなかった: {e}")
        # 注意：支え直すのは**眼球を戻したあと**。支えの減衰は今の姿勢から計算するので、
        #   順序が違うと読み込み側とわずかにずれる（2026-07-29：0.023 の差で
        #   再現性テストが落ちた）。頭だけでなく椅子も取り直すこと。
        _rehold(env, scene, hands)
        if settle_seconds and settle_seconds > 0:
            drift = settle_check(env, settle_seconds, restore=True)
        scene["state"] = snapshot_state(env)
        scene["fingerprint"] = fingerprint(env, scene)
        scene["fingerprint"]["settle"] = drift

    os.makedirs(SCENE_DIR, exist_ok=True)
    path = scene_path(nm)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(scene, fp, ensure_ascii=False, indent=2)

    if env is not None and image:
        try:
            capture_eye(env, scene_path(nm, ".png"))
        except Exception as e:      # 絵が撮れなくても保存自体は成立させる
            print(f"[scene] 左目の絵が撮れなかった: {e}")

    if verbose:
        print(f"[scene] 保存しました → {os.path.relpath(path, _ROOT)}")
        if drift is not None:
            print("        " + drift["summary"])

    # 【なぜ、2026-08-05】シーンを保存するたびに E/docs/シーン一覧.md を
    #   自動で作り直す（手書きの一覧が更新漏れで古くなる問題への対策）。
    #   一覧の自動更新に失敗しても、シーンの保存という主目的は必ず成立させる
    #   ため try/except で包む。
    try:
        import catalog
        catalog.write_catalog()
    except Exception as e:
        print(f"[scene] 注意一覧の自動更新に失敗（シーン保存自体は成立している）: {e}")

    return scene, drift


# ============================================================================
# 組み立て（ここでしか環境を作らない）
# ============================================================================
def build(scene, orient=None, vor=True, seed=0, verbose=False, actuation_model=None,
          vision=True):
    """シーンから環境を作る。**組み立てはここだけ**。

    Args:
        scene: load() が返した辞書
        orient: 視線誘導反射のON/OFF（実験の条件なのでシーンには入れない）
        vor: 前庭動眼反射のON/OFF（同上）
        seed: リセットの乱数の種（同上）
        vision: 環境に視覚センサを持たせるか。False なら vision_params に
            None を渡し、LeanMimoEnv.strip_textures（D/scripts/mimo_lean.py）を
            発動させてテクスチャ（顔・服 約977MB）を単色化する（2026-08-13）。
        actuation_model: 筋の駆動モデル。None なら MuscleModel（従来どおり）。
            【なぜ渡せるようにしたか、2026-07-30】学習ループ
            （`e_growth_train.py`）は既定で SpringDamperModel を使うのに、
            ここが MuscleModel 固定だったため、シーンで学習しようとすると
            **行動の次元が食い違って学習済みモデルを読み込めなかった**。
            注意：駆動モデルは「体の性質」なので本来シーンに入れるべきだが、
            既存シーン3つを壊さないため、まず引数で受ける形にした。
            → やることリストに「シーンに actuation を入れる」を積むこと。
    Returns:
        (env, hands)  hands は実験者の手。使わない設定なら None
    """
    b, w, s = scene["body"], scene["world"], scene["setup"]
    toy = w["toy"]

    # 【なぜ、2026-08-10】body.flexion=True のとき、infant_body.apply_runtime_corrections が
    #   環境構築の途中で apply_limb_tone(profile="newborn_flexor") を隠れて呼び、
    #   四肢の筋緊張バネを起動する（setup.limb_tone とは完全に別の入口）。
    #   Viewerの「四肢の筋緊張」チェックボックスは setup.limb_tone だけを見ているため、
    #   setup.limb_tone を設定しないシーンでは、GUI表示OFFのまま物理にはバネが常時入る
    #   （既知の型バグ。監査 作業記録（非公開）
    #   2026-08-10_run系システムとViewerの型バグ横断監査.md「中1」参照）。
    #   現存シーンは両方のキーが整合しているため実害は無いが、今後の新規シーンで
    #   食い違いが混入するのを早期に発見できるよう、ここで検出して警告する。
    #   verboseに関係なく常に出す＝学習ログにも出て見つけやすくするため
    #   （constraint_summaryの[scene]ログと同じ扱い）。
    if b.get("flexion") and not s.get("limb_tone"):
        print("[scene] 注意 body.flexion=True ですが setup.limb_tone が設定されていません。"
              "Viewerの「四肢の筋緊張」チェックボックスはOFF表示のままですが、"
              "infant_body.apply_runtime_corrections 経由で四肢の筋緊張バネが物理には"
              "常時入ります（既知の型バグ。監査 作業記録（非公開）"
              "2026-08-10_run系システムとViewerの型バグ横断監査.md『中1』参照）。"
              "GUIと物理を一致させたい場合は setup.limb_tone を明示的に設定してください。")

    # --- 1. モジュールの定数として読まれる値を先に決める ---------------------
    #   注意：`e_toy_env` は import した瞬間に環境変数を読んで定数を固める作りなので、
    #     import より前に環境変数を置き、import 後にも定数を直接上書きする
    #     （既に import 済みでも効くようにするため）。
    os.environ["E_RECLINE"] = str(w["recline_deg"])
    os.environ["E_SEAT_FRICTION"] = str(w["seat_friction"])
    os.environ["E_FLOOR_DIM"] = str(w["floor_dim"])
    os.environ["E_PLAIN"] = "1" if w["plain"] else "0"
    os.environ["E_FENCE"] = "1" if w["fence"] else "0"
    os.environ["E_TOY_OBJ"] = "1" if toy["enabled"] else "0"
    os.environ["E_TOY_SHAPE"] = str(toy["shape"])
    os.environ["E_TOY_RADIUS"] = str(toy["radius"])
    os.environ["E_TOY_MODE"] = str(toy["mode"])
    os.environ["E_TOY_DIST"] = str(toy["dist"])
    os.environ["E_TOY_DELAY"] = str(toy["delay_sec"])
    os.environ["E_PARENT"] = "1" if w["parent_intervene"] else "0"
    os.environ["E_PARENT_WAIT"] = str(w["parent_wait_sec"])
    os.environ["E_PARENT_LOST"] = str(w["parent_lost_deg"])
    os.environ["E_TOY_APPROACH"] = str(toy["approach_sec"])
    os.environ["E_TOY_FROM"] = str(toy["approach_from"])
    os.environ["E_STATIC_TEX"] = "1" if w["static_tex"] else "0"
    if w["floor_roll"] is not None:
        os.environ["E_FLOOR_ROLL"] = str(w["floor_roll"])
    if w["floor_condim"] is not None:
        os.environ["E_FLOOR_CONDIM"] = str(w["floor_condim"])
    os.environ["E_EYE_REST_V"] = str(b["eye_rest_vertical_deg"])
    os.environ["E_ORIENT_V"] = str(b["orienting_version"])
    os.environ["E_NECK"] = "1" if b["neck_fix"] else "0"
    os.environ["E_LIMBS"] = "1" if b["limb_fix"] else "0"

    import e_toy_env as TE
    import infant_body as IB
    if actuation_model is None:      # 既定は従来どおり MuscleModel
        from mimoActuation.muscle import MuscleModel
        actuation_model = MuscleModel

    TE.RECLINE_DEG = float(w["recline_deg"])
    TE.SEAT_FRICTION = float(w["seat_friction"])
    TE.FLOOR_DIM = float(w["floor_dim"])
    TE.TOY_SHAPE = str(toy["shape"])
    TE.TOY_RADIUS = float(toy["radius"])
    TE.TOY_MODE = str(toy["mode"])
    TE.TOY_DISTANCE = float(toy["dist"])
    TE.TOY_APPEAR_DELAY = float(toy["delay_sec"])
    TE.PARENT_INTERVENE = bool(w["parent_intervene"])
    TE.PARENT_WAIT_SEC = float(w["parent_wait_sec"])
    TE.PARENT_LOST_DEG = float(w["parent_lost_deg"])
    TE.TOY_APPROACH_SEC = float(toy["approach_sec"])
    TE.TOY_APPROACH_FROM = str(toy["approach_from"])
    # 眼球の基準角はモジュール定数として読まれるので直接入れる
    IB.EYE_REST_VERTICAL_DEG = float(b["eye_rest_vertical_deg"])

    # --- 2. 身体の設定を**シーンから直接**作る -------------------------------
    #   注意：`body_kwargs_from_env` は使わない。環境変数を見る関数を通すと、
    #     呼び出し側が返り値を上書きできる余地が残り、2026-07-25／07-28 と
    #     同じ事故（Viewer と測定で別の体）が再発する。
    kw = _body_kwargs(b, verbose=verbose)

    # --- 3. 環境を作る -------------------------------------------------------
    buf = io.StringIO()
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
    with ctx:
        env = TE.ToySupineEnv(
            actuation_model=actuation_model,
            # 【なぜ、2026-08-13】vision=False のとき None を渡すと、LeanMimoEnv
            #   （taro_core/src/senses/mimo_lean.py。実体はD/scripts/mimo_lean.pyにあり
            #   目標B/C/Dと共有）のstrip_texturesが自動発動し、顔・服テクスチャ
            #   （約977MB）を単色化する。目標B/C/Dは既にこの経路に乗っていたが、
            #   目標Eだけ常に非Noneを渡していたため発動していなかった
            #   （測定の実測：作業記録（非公開） 24節）。
            vision_params=(TE.infant_vision_params(acuity_age=b["age_months"])
                           if vision else None),
            age=float(b["age_months"]),
            toy=bool(toy["enabled"]),
            vor=bool(vor),
            orient=(bool(orient) if orient is not None else False),
            recline_deg=float(w["recline_deg"]),
            fence=bool(w["fence"]),
            toy_radius=float(toy["radius"]),
            toy_dist=float(toy["dist"]),
            newborn_neck=bool(b["neck_fix"]),
            newborn_limbs=bool(b["limb_fix"]),
            **kw)
        env.reset(seed=seed)

        # --- 4. 実験開始前の設定 --------------------------------------------
        hands = _apply_setup(env, s, age=b["age_months"], verbose=verbose)

        # --- 5. 最初の姿勢を流し込む ----------------------------------------
        if scene.get("state"):
            apply_state(env, scene["state"])
            # 姿勢を入れ替えたので、実験者の手の目標角を取り直す
            #   （目標角を指定していない場合は「今の角度で支える」意味だから）
            _rehold(env, scene, hands)
        _apply_limb_tone(env, scene, verbose=verbose)
        _repin(env, scene, verbose=verbose)

        # ---- 6. 「これ以降の reset で戻る先」を、いま作った姿勢に更新する --------
        # 【なぜ、2026-08-03】SupineMimoEnv.reset_model()（d_supine_env.py）は
        #   self.init_position + jitter に戻る。ところが self.init_position は
        #   super().__init__() の中、settle 直後・上のstep2〜5（リクライニング／
        #   シーンの state.qpos／筋緊張／固定）より**前**に1度だけ記録される。
        #   これを更新しないと、この build() が返した直後は正しい姿勢に見えても、
        #   次に env.reset() が呼ばれた瞬間（Taro.__init__ の2回目のreset、
        #   学習中にエピソードが終わって trainer.reset_state() が呼ばれる場面）に
        #   シーンで作り込んだ姿勢が消え、settle直後の既定姿勢へ戻ってしまう。
        #   実測（run/tools/check_scene_reset_consistency.py、2026-08-03）：
        #     新生児_仰向け_柵なし_伸展版  肘 -108.41度 → +5.00度（差+113.41度）
        #     リーチング_リクライニング60度  指・脚に60〜97度規模の不一致
        #   pin_root/pin_joints で毎step固定される関節は影響を受けない
        #   （そちらは reset の有無に関係なく毎stepで書き戻されるため）が、
        #   自由なまま（腕・指など）の関節はここを直さないと戻らない。
        u = env.unwrapped
        if hasattr(u, "init_position"):
            u.init_position = u.data.qpos.copy()
    summary = constraint_summary(scene)
    print(f"[scene] 「{scene['name']}」 固定={summary['pinned_groups']} "
          f"自由={summary['free_groups']} root_pinned={summary['root_pinned']} "
          f"おもちゃ={'あり' if summary['toy_enabled'] else 'なし'}")
    return env, hands


def _apply_limb_tone(env, scene, verbose=False):
    """四肢の筋緊張を効かせる。**姿勢を入れたあとに呼ぶ**。

    注意：目標角を書かなければ「今の姿勢」が落ち着き先になる。だから
      シーンの姿勢を復元する前に呼ぶと、リセット直後の**伸びきった腕**が
      戻る先になってしまい、まったく意味が逆になる。
    """
    tone = (scene.get("setup") or {}).get("limb_tone")
    if not tone:
        return None
    from infant_limbs import apply_limb_tone
    u = env.unwrapped
    return apply_limb_tone(
        u.model, u.data, age=float(scene["body"]["age_months"]),
        target=tone.get("target_deg"),
        stiffness=tone.get("stiffness"),
        hold_deg=float(tone.get("hold_deg", 20.0)),
        groups=tuple(tone.get("groups") or ("arm", "leg")),
        verbose=verbose)


def _rehold(env, scene, hands=None):
    """支えの目標角を、いまの姿勢で取り直す。

    注意：【なぜ要るか】実験者の手も椅子も「今の角度で支える」設定のとき、目標角は
      **支え始めた時点**で決まる。シーンの姿勢を流し込んだあとに取り直さないと、
      バネが**復元前の角度へ引き戻し続ける**。
      実測（2026-07-29）：椅子の取り直しを忘れていたため、リクライニング60度で
      体幹が30秒かけて 59度 → 73度 へ動き、視線が横を向いた。
      ⇒ 頭と椅子の**両方**を取り直す。片方だけ直したのがこの取りこぼしの原因。
    """
    s = scene.get("setup") or {}
    u = env.unwrapped
    hold = s.get("head_hold") or {}
    if hands is not None and not hold.get("target_deg"):
        hands.hold(stiffness=hold.get("stiffness"))
    sup = s.get("body_support") or {}
    chair = getattr(u, "_chair_support", None)
    if chair is not None and sup and not sup.get("target_deg"):
        chair.hold(stiffness=sup.get("stiffness"))


def constraint_summary(scene):
    """setup/body/world の辞書だけから「何が固定され、何が自由か」を計算する。
    副作用なし。env が無くても呼べる（load() 直後でも catalog.py でも使える）。
    """
    s, w = scene["setup"], scene["world"]
    all_groups = {"arm", "finger", "leg", "trunk", "head"}
    # 【なぜ、2026-08-10】以前は `sup = s.get("body_support") or {}` のあと
    #   `free = set(sup.get("free") or ("arm","finger"))` としていた。
    #   body_support が None（未設定）のときも空dictへ読み替えられ、
    #   "arm","finger" だけ自由という**既定値**にフォールバックしていた。
    #   ところが実際に固定処理を行う _apply_setup()・_repin() はどちらも
    #   body_support が None/falsy なら**即return し、何も固定しない**
    #   （＝実際の物理は全関節が自由）。表示だけが「腕・指以外は固定」という
    #   誤った要約を返していた（監査 2026-08-10横断監査 既知③）。
    #   ⇒ _apply_setup()・_repin() と同じ真偽判定（body_support がfalsyなら
    #   何も固定しない）に合わせ、実際の物理と表示を一致させる。
    sup = s.get("body_support")
    if not sup:
        return {
            "root_pinned": False,
            "pinned_groups": [],
            "free_groups": sorted(all_groups),
            "toy_enabled": bool(w["toy"]["enabled"]),
            "flexion": bool(scene["body"]["flexion"]),
            "recline_deg": w["recline_deg"],
        }
    free = set(sup.get("free") or ("arm", "finger"))
    pinned = sorted(all_groups - free) if sup.get("pin_joints", True) else []
    return {
        "root_pinned": bool(sup.get("pin_root", True)),
        "pinned_groups": pinned,          # 例：["leg", "trunk"]
        "free_groups": sorted(free),      # 例：["arm", "finger"]
        "toy_enabled": bool(w["toy"]["enabled"]),
        "flexion": bool(scene["body"]["flexion"]),
        "recline_deg": w["recline_deg"],
    }


def _repin(env, scene, verbose=False):
    """体そのものを、いまの位置・向きで空間に留める。

    注意：必ず**姿勢を復元したあと**に呼ぶ。先に留めるとリセット直後の位置で固まり、
      シーンの姿勢とずれる（＝体だけ別の場所にいる状態になる）。
    """
    sup = (scene.get("setup") or {}).get("body_support")
    if not sup:
        return None
    u = env.unwrapped
    q = u.pin_root(True) if sup.get("pin_root", True) else None
    if verbose and q is not None:
        print("[support] 体そのものも留めた（ベルト相当）"
              " 注意工学的な固定＝人間からの逸脱。逸脱リスト参照")
    # 体幹・脚も完全固定する（2026-07-29、ユーザーの判断）。
    #   バネ（200N·m/rad）では体幹がゆっくり動き続け、VOR が打ち消そうとして
    #   **眼球が可動域の上限に張り付いた**。視線の実験が成立しないため。
    if sup.get("pin_joints", True):
        from e_head_hold import joints_to_support
        free = tuple(sup.get("free") or ("arm", "finger"))
        if (scene.get("setup") or {}).get("head_hold") and "head" not in free:
            free = free + ("head",)      # 首は実験者の手が担当
        n = u.pin_joints(joints_to_support(u.model, free=free), on=True)
        if verbose and n:
            print(f"[support] 体幹・脚を完全固定した（{n}関節）"
                  " 注意工学的な固定＝人間からの逸脱。逸脱リスト参照")
    return q


def _body_kwargs(b, verbose=False):
    """シーンの body から `ToySupineEnv` に渡す身体の設定を作る。"""
    from infant_body import (NEWBORN_SHAPE_DEFAULTS, HEAD_ELONGATION,
                             body_scale_custom)
    kw = {
        "limb_scale": float(b["limb_scale"]),
        "limb_fix": bool(b["limb_fix"]),
        "distal_mass": float(b["distal_mass"]),
        "flexion": bool(b["flexion"]),
        "flexion_stiffness": (None if b["flexion_stiffness"] is None
                              else float(b["flexion_stiffness"])),
    }
    if b["shape"]:
        scales = dict(NEWBORN_SHAPE_DEFAULTS)
        scales.update(b.get("shape_scales") or {})
        kw["head_elongation"] = float(HEAD_ELONGATION if b["head_elongation"] is None
                                      else b["head_elongation"])
        custom = body_scale_custom(float(b["age_months"]), scales, verbose=verbose)
        if custom:
            kw["custom_measurements"] = custom
    else:
        # 体型の補正なし＝素の mimoGrowth（アブレーション）
        kw["head_elongation"] = 1.0
    return kw


def _apply_setup(env, s, age, verbose=False):
    """首のバネと実験者の手を効かせる。"""
    from infant_body import apply_neck_tone
    from e_head_hold import CaregiverHands
    u = env.unwrapped
    tone = s.get("neck_tone")
    if tone:
        apply_neck_tone(u.model, float(age), target=tone.get("target_deg"),
                        stiffness=tone.get("stiffness"),
                        data=u.data, verbose=verbose)
    # 体を支える（椅子とベルト）。首より先に効かせる
    #   ＝首の目標角を「支えたあとの姿勢」で取れるようにするため。
    sup = s.get("body_support")
    if sup:
        from e_head_hold import joints_to_support, GROUP_JP
        free = tuple(sup.get("free") or ("arm", "finger"))
        # 注意：首は「実験者の手」が担当するので、二重に支えない（役割を分ける）。
        #   両方が同じ `jnt_stiffness` を書くので、混ざると
        #   「どちらの設定が効いているのか」が分からなくなる（落とし穴 項62 の型）。
        if s.get("head_hold") and "head" not in free:
            free = free + ("head",)
        names = joints_to_support(u.model, free=free)
        if names:
            chair = CaregiverHands(u.model, u.data, joints=tuple(names),
                                   stiffness=sup.get("stiffness"))
            chair.hold(target=sup.get("target_deg"), verbose=False)
            u._chair_support = chair          # 後から緩められるように持っておく
            if verbose:
                jp = "・".join(GROUP_JP.get(g, g) for g in free)
                print(f"[support] 椅子とベルトが体を支える: {len(names)}関節 "
                      f"強さ={chair.stiffness:.1f}N·m/rad  自由なのは {jp}と眼球 "
                      f"[人間のリーチ実験も乳児を椅子に固定する＝von Hofsten 1982]")
        # 注意：体そのものを留めるのは**姿勢を復元したあと**（`_repin`）。
        #   ここで留めるとリセット直後の位置で固まり、シーンの姿勢とずれる。
    hold = s.get("head_hold")
    if not hold:
        return None
    # 注意：四肢の筋緊張は `apply_state` の**あと**に効かせる（`_apply_limb_tone`）。
    #   目標角を「今の姿勢」にするので、シーンの姿勢を入れる前だと
    #   リセット直後の伸びきった姿勢が落ち着き先になってしまう。
    hands = CaregiverHands(u.model, u.data, stiffness=hold.get("stiffness"))
    hands.hold(target=hold.get("target_deg"), verbose=verbose)
    return hands


# ============================================================================
# 姿勢の記録と復元
# ============================================================================
def snapshot_state(env):
    """今の姿勢を記録する。qpos が正、joints_readable は人が読むための写し。"""
    u = env.unwrapped
    m, d = u.model, u.data
    readable = {}
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                      int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue           # free / ball は角度で表せないので qpos に任せる
        nm = (m.joint(j).name or "").split(":")[-1]
        if not nm or "eye" in nm:
            continue
        val = float(d.qpos[int(m.jnt_qposadr[j])])
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            val = float(np.degrees(val))
        readable[nm] = round(val, 2)
    st = {"qpos": [float(x) for x in d.qpos],
          "joints_readable": readable}
    if getattr(u, "_toy", False):
        st["toy_pos"] = [float(x) for x in d.qpos[u._toy_qadr:u._toy_qadr + 3]]
    return st


def reset_to_scene(env, scene, hands=None, seed=0, toy_offset=None):
    """シーンの状態に戻す。測定が `env.reset()` を呼ぶ代わりにこれを使う。

    注意：【なぜ要るか】`env.reset()` だけだと**シーンの姿勢が失われる**。
      測定スクリプトは条件ごとに reset するので（反射ON/OFF、シードごと）、
      そのたびにシーンへ戻さないと「1回目だけシーンの姿勢、2回目からは既定の姿勢」
      という状態になる。これは Viewer と測定の食い違いと同じ型のバグ。

    Args:
        hands: 渡すと実験者の手を支え直す（目標角はシーンの設定から取る）
        toy_offset: おもちゃをシーンの位置から動かす[m]（実験の条件として振る用）
    """
    env.reset(seed=seed)
    if scene.get("state"):
        apply_state(env, scene["state"])
    # 注意：実験者の手も椅子も、いまの姿勢で支え直す（片方だけだと崩れる）
    hold = scene["setup"].get("head_hold") or {}
    if hands is not None and hold.get("target_deg"):
        hands.hold(target=hold["target_deg"], stiffness=hold.get("stiffness"))
    else:
        _rehold(env, scene, hands)
    if toy_offset is not None:
        place_toy(env, np.asarray(toy_offset, dtype=float), relative=True)
    _apply_limb_tone(env, scene)   # 注意姿勢を戻したので、筋緊張の目標も取り直す
    _repin(env, scene)             # 注意留める位置も取り直す
    u = env.unwrapped
    mujoco.mj_forward(u.model, u.data)
    return env


def place_toy(env, pos, relative=False):
    """おもちゃを置く。「親が手に持っている」モードでも固定位置ごと動かす。

    注意：`d.qpos` を書くだけでは足りない。`hold` モードは毎ステップ `_rest_pos` の位置へ
      戻すので、次の step で元の場所に引き戻される（2026-07-28 に踏んだ）。
    """
    u = env.unwrapped
    m, d = u.model, u.data
    if not getattr(u, "_toy", False):
        return None
    p = np.asarray(pos, dtype=float)
    if relative:
        p = np.asarray(d.qpos[u._toy_qadr:u._toy_qadr + 3], dtype=float) + p
    d.qpos[u._toy_qadr:u._toy_qadr + 3] = p
    d.qvel[u._toy_dadr:u._toy_dadr + 6] = 0.0
    u._rest_pos = p.copy()
    u._anchor = p + np.array([0.0, 0.0, u._tether_len])
    u._toy_pending = False
    u._toy_arriving = False
    mujoco.mj_forward(m, d)
    return p


def apply_state(env, state):
    """記録した姿勢を書き戻す。**qpos から復元する**（joints_readable は使わない）。"""
    u = env.unwrapped
    m, d = u.model, u.data
    q = np.asarray(state["qpos"], dtype=float)
    if q.shape[0] != m.nq:
        raise ValueError(
            f"姿勢の長さが合わない（記録 {q.shape[0]} ≠ 今のモデル {m.nq}）。\n"
            f"  ＝体か環境の条件が違うので、この姿勢は使えない。\n"
            f"  シーンの body / world を、保存したときと同じにすること。")
    d.qpos[:] = q
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    if state.get("toy_pos") is not None and getattr(u, "_toy", False):
        pos = np.asarray(state["toy_pos"], dtype=float)
        d.qpos[u._toy_qadr:u._toy_qadr + 3] = pos
        d.qvel[u._toy_dadr:u._toy_dadr + 6] = 0.0
        # 「親が手に持っている」モードは _rest_pos の位置に固定し続けるので、
        # そこも同じ場所にしないと、次の step で元の場所へ戻されてしまう。
        u._rest_pos = pos.copy()
        u._anchor = pos + np.array([0.0, 0.0, u._tether_len])
        u._toy_pending = False
        u._toy_arriving = False
    mujoco.mj_forward(m, d)


# ============================================================================
# 指紋（食い違いの検知）
# ============================================================================
def fingerprint(env, scene=None):
    """この環境の状態を表す数値。読み込みが正しく効いたかの照合に使う。

    注意：どれか1つでも合わなければ「別の環境」＝測定を始めてはいけない。
      2026-07-28 に実際に起きた食い違い（首30度で保存したのに60度で測っていた／
      Viewer と測定で四肢の屈曲が違った）は、いずれもこの指紋で止まる。

    【2026-07-29・自己テストで判明した穴】最初の版は「今の姿勢から計算した量」
      （首の角度・目の位置・おもちゃが見えるか）しか見ていなかった。
      ところがシーンは **qpos を丸ごと流し込む**ので、
      四肢の屈曲を反転しても・リクライニングを20度変えても・首を支える角度を
      30度ずらしても、**最終的な姿勢は同じ**になり、照合をすり抜けた（3件見逃し）。
      ⇒ qpos に上書きされない「モデルそのものの値」（バネ・可動域・板の位置）を
        指紋に入れる。`_model_fingerprint` がそれ。
    """
    import e_visibility as VIS
    u = env.unwrapped
    m, d = u.model, u.data
    mujoco.mj_forward(m, d)

    def bpos(n):
        return np.array(d.xpos[int(m.body(n).id)], dtype=float)

    fp = {"total_mass_kg": round(float(np.sum(m.body_mass)), 6),
          "nq": int(m.nq), "ngeom": int(m.ngeom),
          "model": _model_fingerprint(m)}
    if scene is not None:
        fp["config_hash"] = config_hash(scene)

    # 首の3軸
    head = {}
    for j in range(m.njnt):
        nm = (m.joint(j).name or "").split(":")[-1]
        if nm in ("head_swivel", "head_tilt", "head_tilt_side"):
            head[nm] = round(float(np.degrees(d.qpos[int(m.jnt_qposadr[j])])), 2)
    fp["head_angles_deg"] = head

    # 体幹の傾き（リクライニングが効いているか）
    #   骨盤から胸へ向かうベクトルが水平から何度上がっているか
    try:
        axis = bpos("upper_body") - bpos("hip")
        n = float(np.linalg.norm(axis))
        fp["trunk_tilt_deg"] = round(float(np.degrees(np.arcsin(
            np.clip(axis[2] / max(n, 1e-9), -1, 1)))), 2)
    except Exception:
        fp["trunk_tilt_deg"] = None

    # 目の位置と視線
    eyes, fwds = [], []
    for nm in ("eye_left", "eye_right"):
        cid = int(m.camera(nm).id)
        eyes.append(np.array(d.cam_xpos[cid], dtype=float))
        fwds.append(-np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)[:, 2])
    eye = np.mean(eyes, axis=0)
    fwd = np.mean(fwds, axis=0)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    fp["eye_pos"] = [round(float(x), 5) for x in eye]
    fp["gaze_dir"] = [round(float(x), 5) for x in fwd]

    # おもちゃ（あれば）
    if getattr(u, "_toy", False):
        pos = np.array(d.qpos[u._toy_qadr:u._toy_qadr + 3], dtype=float)
        fp["toy_pos"] = [round(float(x), 5) for x in pos]
        v = pos - eye
        r = float(np.linalg.norm(v))
        fp["toy_dist_cm"] = round(r * 100, 2)
        fp["toy_angle_deg"] = round(float(np.degrees(np.arccos(
            np.clip(float(np.dot(v / max(r, 1e-9), fwd)), -1, 1)))), 2)
        # 遮蔽込みで実際に見えるか
        #   注意：幾何的な角度だけで「見える」と言ってはいけない。2026-07-28 に
        #     角度だけで判定して「見えて届く」と報告した12件が、実際は
        #     すべて自分の体に隠れていた。
        try:
            bid = int(m.body("test_object1").id)
            sv = VIS.visible_by_segment(m, d, bid, "eye_left", size=96)
            fp["toy_visible_left"] = bool(sv["seen"])
            fp["toy_view_xy"] = ([round(float(sv["cx"]), 3), round(float(sv["cy"]), 3)]
                                 if sv["seen"] else None)
        except Exception as e:
            fp["toy_visible_left"] = None
            fp["toy_visible_error"] = str(e)
        # 手が届くか（肩からの距離 ÷ 腕の長さ）
        try:
            sh_r, sh_l = bpos("right_upper_arm"), bpos("left_upper_arm")
            arm = (float(np.linalg.norm(bpos("right_lower_arm") - sh_r))
                   + float(np.linalg.norm(bpos("right_hand") - bpos("right_lower_arm"))))
            fp["arm_len_cm"] = round(arm * 100, 2)
            fp["toy_reach_ratio"] = round(min(
                float(np.linalg.norm(pos - sh_r)),
                float(np.linalg.norm(pos - sh_l))) / max(arm, 1e-9), 3)
        except Exception:
            fp["toy_reach_ratio"] = None
    return fp


def _model_fingerprint(m):
    """モデルそのものの特徴。**姿勢（qpos）を入れ替えても変わらない値だけ**を見る。

    | 見るもの | 捕まえる食い違い |
    |---|---|
    | バネの本数・強さの合計 | 四肢の屈曲・首の筋緊張・実験者の手 |
    | バネの目標角の合計 | 頭を支える角度（30度で保存し60度で測る事故） |
    | 可動域の合計 | 四肢の屈曲（伸展側の壁） |
    | 背もたれの板の位置・向き・摩擦 | リクライニング角 |
    | 体の自由関節の初期姿勢 qpos0 | リクライニング（体を起こす回転） |
    """
    out = {
        "spring_n": int(np.sum(np.asarray(m.jnt_stiffness) > 0)),
        "spring_k_sum": round(float(np.sum(m.jnt_stiffness)), 4),
        "spring_target_sum_deg": round(float(np.sum(np.degrees(m.qpos_spring))), 3),
        "jnt_range_sum": round(float(np.sum(m.jnt_range)), 4),
        "damping_sum": round(float(np.sum(m.dof_damping)), 4),
    }
    # 背もたれの板（リクライニングのときだけ存在する）
    try:
        gid = int(m.geom("recline_back").id)
        out["seat"] = {
            "pos": [round(float(x), 5) for x in m.geom_pos[gid]],
            "quat": [round(float(x), 5) for x in m.geom_quat[gid]],
            "friction": round(float(m.geom_friction[gid][0]), 4),
        }
    except Exception:
        out["seat"] = None
    # 体の自由関節の初期姿勢（リクライニングはここを書き換えて体を起こす）
    # 注意：【2026-07-29 修正】**関節名**に "mimo_location" が含まれるかで探していたが、
    #   MIMo では body 名が "mimo_location" で**関節名は "mimo_orientation"** なので
    #   一度も一致せず、この指紋は常に None だった＝**照合項目が1つ死んでいた**。
    #   （リクライニングの違いは背もたれの板の位置で検知できていたので気づかなかった）
    out["body_qpos0"] = None
    try:
        from e_toy_env import ROOT_BODY
        for j in range(m.njnt):
            if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            if (m.body(int(m.jnt_bodyid[j])).name or "") != ROOT_BODY:
                continue
            a = int(m.jnt_qposadr[j])
            out["body_qpos0"] = [round(float(x), 5) for x in m.qpos0[a:a + 7]]
            break
    except Exception:
        pass
    return out


def config_hash(scene):
    """シーンの設定（体・環境・実験前の設定）の指紋。

    注意：シーンファイルを手で書き換えたのに指紋を作り直し忘れた場合を捕まえる。
      数値を直接編集する運用を想定しているので、これが無いと
      「ファイルには新しい値、指紋には古い値」という状態が黙って通る。
    """
    import hashlib
    payload = json.dumps({k: scene.get(k) for k in ("body", "world", "setup")},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def compare(recorded, current):
    """指紋を比べて、ずれた項目の一覧を返す（空なら一致）。"""
    diffs = []

    def add(key, a, b, tol, unit=""):
        if a is None or b is None:
            if a != b:
                diffs.append(f"{key}: 記録={a} / 今={b}")
            return
        if abs(float(a) - float(b)) > tol:
            diffs.append(f"{key}: 記録={float(a):.3f}{unit} / 今={float(b):.3f}{unit}"
                         f"  （差 {abs(float(a)-float(b)):.3f}{unit}）")

    if recorded.get("nq") != current.get("nq"):
        diffs.append(f"関節の数: 記録={recorded.get('nq')} / 今={current.get('nq')}"
                     f"  ← 体か環境の条件が違う")
        return diffs        # ここが違うと以降の比較は意味を持たない
    # 注意：柵・背もたれは geom を足すだけで関節を増やさないので、nq では捕まらない
    #   （2026-07-29 の自己テストで「柵の有無を反転」がすり抜けて判明）。
    if recorded.get("ngeom") != current.get("ngeom"):
        diffs.append(f"物体の数: 記録={recorded.get('ngeom')} / 今={current.get('ngeom')}"
                     f"  ← 柵・背もたれ・おもちゃの有無が違う")

    add("体の総質量", recorded.get("total_mass_kg"), current.get("total_mass_kg"),
        TOL["mass_kg"], "kg")
    for k, a in (recorded.get("head_angles_deg") or {}).items():
        add(f"首 {k}", a, (current.get("head_angles_deg") or {}).get(k),
            TOL["angle_deg"], "度")
    add("体幹の傾き", recorded.get("trunk_tilt_deg"), current.get("trunk_tilt_deg"),
        TOL["angle_deg"], "度")

    for key, tol in (("eye_pos", TOL["pos_m"]), ("gaze_dir", TOL["dir"]),
                     ("toy_pos", TOL["pos_m"])):
        a, b = recorded.get(key), current.get(key)
        if a is None or b is None:
            if (a is None) != (b is None):
                diffs.append(f"{key}: 記録={a} / 今={b}")
            continue
        dv = float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))
        if dv > tol:
            nm = {"eye_pos": "目の位置", "gaze_dir": "視線の向き",
                  "toy_pos": "おもちゃの位置"}[key]
            diffs.append(f"{nm}: 記録={_v(a)} / 今={_v(b)}  （差 {dv:.4f}）")

    add("肩からおもちゃ（腕に対する割合）", recorded.get("toy_reach_ratio"),
        current.get("toy_reach_ratio"), TOL["ratio"])
    if recorded.get("toy_visible_left") != current.get("toy_visible_left"):
        diffs.append(
            f"おもちゃが見えるか: 記録={_seen(recorded.get('toy_visible_left'))}"
            f" / 今={_seen(current.get('toy_visible_left'))}")

    # --- モデルそのもの（姿勢を入れ替えても変わらない値）---------------------
    #   注意：ここが無いと、四肢の屈曲・リクライニング角・頭を支える角度の食い違いが
    #     すり抜ける（2026-07-29 の自己テストで3件見逃した）。
    ra, ca = recorded.get("model") or {}, current.get("model") or {}
    if ra or ca:
        # 注意：許容値は項目ごとに変える。**一律 1e-3 では厳しすぎる**：
        #   減衰は「今の姿勢から臨界減衰を計算」して入るので、Viewer で保存した
        #   瞬間と読み込んだ瞬間で末尾がわずかに変わる。2026-07-29 に実測 0.016
        #   （0.014%）の差で測定が止まった＝実害のない差で実験を妨げていた。
        #   一方、意味のある違い（首を支える角度を30度ずらす＝合計が30変わる）は
        #   下の値で確実に捕まる。
        labels = {"spring_n": ("バネの入った関節の数", 0.0),
                  "spring_k_sum": ("バネの強さの合計", 0.5),
                  "spring_target_sum_deg": ("バネの目標角の合計", 0.5),
                  "jnt_range_sum": ("可動域の合計", 0.01),
                  "damping_sum": ("減衰の合計", 0.5)}
        for k, (lab, tol) in labels.items():
            a, b = ra.get(k), ca.get(k)
            if a is None or b is None:
                continue
            if abs(float(a) - float(b)) > tol:
                diffs.append(f"{lab}: 記録={float(a):.3f} / 今={float(b):.3f}"
                             f"  ← 体か実験前の設定が違う")
        sa, sb = ra.get("seat"), ca.get("seat")
        if (sa is None) != (sb is None):
            diffs.append(f"背もたれの板: 記録={'あり' if sa else 'なし'}"
                         f" / 今={'あり' if sb else 'なし'}  ← リクライニングの有無が違う")
        elif sa and sb:
            for key, lab, tol in (("pos", "背もたれの位置", TOL["pos_m"]),
                                  ("quat", "背もたれの向き", TOL["dir"])):
                dv = float(np.linalg.norm(np.array(sa[key], dtype=float)
                                          - np.array(sb[key], dtype=float)))
                if dv > tol:
                    diffs.append(f"{lab}: 記録={_v(sa[key])} / 今={_v(sb[key])}"
                                 f"  （差 {dv:.4f}）← リクライニング角が違う")
            if abs(float(sa["friction"]) - float(sb["friction"])) > 1e-4:
                diffs.append(f"背もたれの摩擦: 記録={sa['friction']} / 今={sb['friction']}")
        qa, qb = ra.get("body_qpos0"), cb_get(ca, "body_qpos0")
        if qa and qb:
            dv = float(np.linalg.norm(np.array(qa, dtype=float)
                                      - np.array(qb, dtype=float)))
            if dv > TOL["dir"]:
                diffs.append(f"体の初期姿勢: 差 {dv:.4f}  ← リクライニング角が違う")

    # --- 設定そのもの（ファイルを手で書き換えたのに指紋が古い場合）-----------
    ha, hb = recorded.get("config_hash"), current.get("config_hash")
    if ha and hb and ha != hb:
        diffs.append(
            f"設定の指紋: 記録={ha} / 今={hb}\n"
            f"      ← シーンファイルの body / world / setup が書き換えられています。\n"
            f"        Viewer で開き直して保存すると指紋が作り直されます。")
    return diffs


def cb_get(d, k):
    return (d or {}).get(k)


def _v(a):
    return "(" + ", ".join(f"{float(x):+.3f}" for x in a) + ")"


def _seen(v):
    return {True: "見える", False: "見えない", None: "不明"}[v]


def verify(scene, env, strict=True, verbose=True):
    """シーンの指紋と、いま作った環境が一致するか確かめる。

    Args:
        strict: True なら食い違ったときに例外で止める（測定を始めさせない）
    Returns:
        ずれた項目の一覧（空なら一致）
    """
    rec = scene.get("fingerprint")
    if not rec:
        if verbose:
            print("[scene] 指紋が記録されていないので照合できない（古いシーン）")
        return []
    cur = fingerprint(env, scene)
    diffs = compare(rec, cur)
    if not diffs:
        if verbose:
            print(f"[scene] 照合OK — 「{scene.get('name')}」と同じ環境です")
        return []
    msg = ["=" * 74,
           f" 食い違いを検知しました — シーン「{scene.get('name')}」",
           "=" * 74]
    msg += [f"  ・{t}" for t in diffs]
    msg += ["", "  ＝Viewerで保存した状態と、いま作った環境が違います。",
            "    測定を始めても、見ていたものとは別の実験になります。",
            f"    保存時の左目の絵: {os.path.relpath(scene_path(scene.get('name'), '.png'), _ROOT)}"]
    text = "\n".join(msg)
    if strict:
        raise RuntimeError("\n" + text)
    if verbose:
        print(text)
    return diffs


# ============================================================================
# 安定確認（保存した姿勢が実験開始時まで保つか）
# ============================================================================
def settle_check(env, seconds=3.0, restore=True):
    """物理を進めて、姿勢がどれだけ崩れるかを測る。

    【なぜ要るか、2026-07-29】首を30度に整えて保存しても、測定が物理を進めると
    重力で 63度まで倒れる（2026-07-28 の実測）。保存した瞬間は正しいのに
    実験開始時にはもう別物、という食い違いを保存の時点で捕まえる。

    Args:
        restore: True なら測ったあと元の姿勢に戻す（保存する姿勢を変えない）
    Returns:
        {"seconds":, "head_drift_deg":, "toy_drift_m":, "ok":, "summary":}
    """
    u = env.unwrapped
    m, d = u.model, u.data
    before_q = d.qpos.copy()
    before_v = d.qvel.copy()
    fp0 = fingerprint(env)
    dt = float(m.opt.timestep) * int(u.frame_skip)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for _ in range(max(1, int(seconds / dt))):
            env.step(a)
    fp1 = fingerprint(env)

    head_drift = {k: round(abs(v - (fp1["head_angles_deg"] or {}).get(k, v)), 2)
                  for k, v in (fp0["head_angles_deg"] or {}).items()}
    # 注意：ここは `abs(a or 0.0 - (b or 0.0))` と書くと `or` の方が `-` より弱いため
    #   `a or (0.0 - b)` になり、**差ではなく傾きそのもの**が入る（最初の版のバグ）。
    t0 = fp0.get("trunk_tilt_deg")
    t1 = fp1.get("trunk_tilt_deg")
    trunk_drift = (abs(float(t0) - float(t1)) if (t0 is not None and t1 is not None)
                   else 0.0)
    worst_ang = max([trunk_drift] + list(head_drift.values()) or [0.0])
    toy_drift = None
    if fp0.get("toy_pos") and fp1.get("toy_pos"):
        toy_drift = float(np.linalg.norm(np.array(fp1["toy_pos"])
                                         - np.array(fp0["toy_pos"])))
    eye_drift = float(np.linalg.norm(np.array(fp1["eye_pos"])
                                     - np.array(fp0["eye_pos"])))

    # 注意：おもちゃを「紐で吊るす」モード（tether）は**揺れるのが仕様**なので、
    #   動いたことを崩れと判定しない（2026-07-29：新生児のシーンが
    #   おもちゃの揺れ20.9cm だけを理由に「保てません」と出ていた）。
    try:
        import e_toy_env as _TE
        _toy_fixed = (str(_TE.TOY_MODE) == "hold")
    except Exception:
        _toy_fixed = True
    ok = (worst_ang <= SETTLE_LIMIT["angle_deg"]
          and eye_drift <= SETTLE_LIMIT["pos_m"]
          and (toy_drift is None or not _toy_fixed
               or toy_drift <= SETTLE_LIMIT["pos_m"]))
    parts = [f"{seconds:g}秒進めた結果："]
    parts.append(f"首の最大ずれ {worst_ang:.2f}度")
    parts.append(f"目の移動 {eye_drift*100:.2f}cm")
    if toy_drift is not None:
        parts.append(f"おもちゃ {toy_drift*100:.2f}cm"
                     + ("" if _toy_fixed else "（紐で吊るす設定なので揺れて当然）"))
    parts.append("→ 姿勢は保てます" if ok else "→ 注意この姿勢は保てません")
    out = {"seconds": seconds, "head_drift_deg": head_drift,
           "trunk_tilt_before": fp0.get("trunk_tilt_deg"),
           "trunk_tilt_after": fp1.get("trunk_tilt_deg"),
           "eye_drift_m": round(eye_drift, 5),
           "toy_drift_m": (None if toy_drift is None else round(toy_drift, 5)),
           "worst_angle_deg": round(worst_ang, 2),
           "ok": bool(ok), "summary": "  ".join(parts)}

    if restore:
        d.qpos[:] = before_q
        d.qvel[:] = before_v
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
    return out


# ============================================================================
# 左目の絵（見ていた景色を残す）
# ============================================================================
def capture_eye(env, path, cam="eye_left", size=None):
    """左目に映っている景色を1枚保存する。

    注意：落とし穴チェックリスト 項70「位置・姿勢の問題は数値より先に絵を撮る」を
      仕組みにしたもの。2026-07-28 は数値だけで1時間・7回失敗し、
      画像1枚で即座に解決した。

    【2026-07-29】`size` の既定を「モデルが持っている描画バッファに収める」に変えた。
      【なぜ】以前は 256 を要求し、足りなければ `m.vis.global_.offwidth/offheight` を
      **その場で広げていた**。Viewer が動いている最中にこれをやると、既に作られた
      OpenGL のコンテキストと食い違ってプロセスごと落ちうる
      （Viewer が Segmentation fault で終了。2026-07-29）。
      ⇒ 実行中はバッファを広げず、収まる大きさで撮る。
    """
    from PIL import Image
    u = env.unwrapped
    m, d = u.model, u.data
    lim = min(int(m.vis.global_.offwidth), int(m.vis.global_.offheight))
    size = lim if size is None else min(int(size), lim)
    if size < 16:
        raise RuntimeError(f"描画バッファが小さすぎて絵が撮れない（{lim}px）")
    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, size, size)
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FIXED
    c.fixedcamid = int(m.camera(cam).id)
    ren.update_scene(d, c)
    img = ren.render().copy()
    ren.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)
    return path


# ============================================================================
# 測定スクリプトからの入口
# ============================================================================
def from_env_var(default=None):
    """環境変数 E_SCENE でシーンを選ぶ。測定スクリプトはこれを呼ぶだけでよい。"""
    name = os.environ.get("E_SCENE") or default
    if not name:
        avail = "、".join(list_scenes()) or "（1つも無い）"
        raise RuntimeError(
            "シーンが指定されていない。E_SCENE=名前 を付けて実行すること。\n"
            f"  使えるシーン: {avail}")
    return load(name)


def open_scene(name=None, orient=None, vor=True, seed=0, strict=True, verbose=False):
    """シーンを読んで環境を作り、指紋を照合するところまでを1回で行う。

    測定スクリプトはこれ1行で始められる::

        env, hands, scene = e_scene.open_scene(orient=True)
    """
    scene = load(name) if name else from_env_var()
    env, hands = build(scene, orient=orient, vor=vor, seed=seed, verbose=verbose)
    verify(scene, env, strict=strict, verbose=True)
    return env, hands, scene


if __name__ == "__main__":
    names = list_scenes()
    print("=" * 74)
    print(" 使えるシーン")
    print("=" * 74)
    if not names:
        print("  （1つも無い。Viewer で作って保存してください）")
    for n in names:
        sc = load(n)
        fp = sc.get("fingerprint") or {}
        st = "姿勢あり" if sc.get("state") else "姿勢なし"
        se = (fp.get("settle") or {}).get("ok")
        print(f"  {n}")
        print(f"      {sc['body']['age_months']:g}ヶ月 / "
              f"リクライニング{sc['world']['recline_deg']:g}度 / "
              f"柵{'あり' if sc['world']['fence'] else 'なし'} / {st}"
              f"{'' if se is None else ('  姿勢は保てる' if se else '  注意姿勢が崩れる')}")
        if sc.get("note"):
            print(f"      {sc['note']}")
