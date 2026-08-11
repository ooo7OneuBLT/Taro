"""視線誘導反射（Orienting Reflex）が発動した瞬間の、太郎の両目カメラ映像をmp4で保存する。

仕様：作業記録（非公開）
（この仕様のファイル名は依頼元の管理都合。実体は本セッションで受けた指示）

【何をするか】
`run/scenes/視線誘導反射_仰向け_頭を支える.json`（4ヶ月・仰向け・頭固定・おもちゃ距離8.6cm）
を、run/main.py を経由せず `Config.from_spec` + `run.plugins.common.scene.build` で
直接組み立て、ランダム行動（探索ノイズの代わり）で env.step() を回しながら
両目のカメラ画像（`env.unwrapped.get_vision_obs()` の "eye_left"/"eye_right"）を
低頻度でローリングバッファへ蓄積し続ける。`env.unwrapped._orienting.n_saccades`
（視線誘導反射がサッケードを撃った回数）の増分を検知した瞬間の前後を切り出してmp4にする。

【なぜ方策（学習済みモデル）を使わないか】
視線誘導反射は皮質下反射（`E/scripts/e_toy_env.py` の `step()` で、方策の出力より
先に `self._orienting.apply(action)` として首・目の指令に直接加算される）であり、
方策を介さない。スモークテスト（本ツール作成前に実施）で、ランダム行動
（`env.action_space.sample()`）だけで2000ステップ中3回発火することを実測済み
（探索ノイズ相当の自発的な首・目の動きが、視野内の模様を動かして見せるため）。
学習済みモデルは不要と判断した。

【なぜ常時（低頻度で）描画し続けるか】
発火の**3秒前**からの映像が要る。物理シミュレーションはリプレイ可能な描画バッファを
持たないため、発火してから遡って描画することはできない。常に低頻度で描画して
ローリングバッファ（`collections.deque`、上限長）に貯め続けるしかない
（`run/tools/record_self_touch_clip.py` と同じ設計）。

【record_self_touch_clip.pyとの違い】
自己接触デモは `env.render()`（三人称視点）を使っていたが、こちらは太郎の目の
カメラそのもの（`get_vision_obs()` の "eye_left"/"eye_right"）を使う。左右の画像を
`np.hstack` で横に並べて1フレームにする。

【描画の頻度について】
`get_vision_obs()` は内部で `VISION_MIN_DT=0.1`（sim秒）ごとにしか実際には
再描画しない（それより短い間隔で呼んでもキャッシュを返すだけ）。そこで
RENDER_EVERY を「物理1tickの秒数(DT) × RENDER_EVERY = 0.1秒」に一致させ、
呼ぶたびに新しい画を確実に取れるようにした（重複フレームを避ける）。

【触ってよいファイル】このファイルのみ（仕様の指定通り）。
"""
import collections
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 【なぜ】Windows既定のcp932コンソールでは、シーン構築時のprint（濁点付き記号や
#   「・」以外の特殊文字）でUnicodeEncodeErrorが起き、途中で落ちる
#   （スモークテストで実際に踏んだ）。utf-8へ差し替えて防ぐ。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2                                              # noqa: E402
import numpy as np                                       # noqa: E402

from run.config import Config                            # noqa: E402
from run.plugins.common import scene as scene_mod         # noqa: E402

SCENE = "視線誘導反射_仰向け_頭を支える"
OUT_PATH = os.path.join(_ROOT, "E", "docs", "figures",
                         "orienting_reflex_both_eyes_multi_2026-08-04.mp4")

DT = 0.01                     # run/trainer.py と同じ（MuJoCoの1物理ステップの秒数）
RENDER_EVERY = 10             # DT*RENDER_EVERY=0.1秒＝get_vision_obsの再描画間隔(VISION_MIN_DT)に一致
FPS = (1.0 / DT) / RENDER_EVERY     # =10
SEC_BEFORE = 3.0
SEC_AFTER = 45.0  # 測定が実測(下調べ)に基づき変更・複数発火を収めるため延長
MAX_STEPS = 15000             # 測定が実測に基づき変更。下調べ(scratchpad)でtick10,238,1124,2269,3489に
                               # 5回発火を確認済み。SEC_AFTER=45sでtick~4510まで録れるので余裕を持って収まる
SEED = 0


def main():
    spec = {
        "name": "orienting_reflex_clip",
        "scene": SCENE,
        "taro": {
            "orienting_reflex": True,
            "vor": True,
        },
        "run": {},
    }
    cfg = Config.from_spec(spec)

    taro_spec = dict(cfg._taro)
    if cfg.age_months is not None:
        taro_spec["age_months"] = float(cfg.age_months)
    # 注意：hybrid=False（内臓感覚は不要。学習済み脳を通さないので観測次元は無関係）
    env, scene_dict, hands = scene_mod.build(
        cfg.scene, taro=taro_spec, seed=SEED, verbose=True, hybrid=False)

    u = env.unwrapped
    assert u._orienting is not None, \
        "[想定外] 反射が有効化されていない（想定外として報告すること）"
    print(f"[診断] 反射オブジェクト: {type(u._orienting).__name__}", flush=True)

    env.reset(seed=SEED)

    # ---- 0コスト診断：両目の画像形状を確認する -------------------------------
    imgs = u.get_vision_obs()
    assert "eye_left" in imgs and "eye_right" in imgs, \
        f"[想定外] eye_left/eye_rightが無い（キー: {list(imgs.keys())}）"
    left0, right0 = imgs["eye_left"], imgs["eye_right"]
    print(f"[診断] eye_left形状={left0.shape} eye_right形状={right0.shape} "
          f"dtype={left0.dtype}", flush=True)
    if left0.shape != right0.shape:
        raise RuntimeError(
            f"[想定外] 左右の目の画像サイズが異なる（left={left0.shape} "
            f"right={right0.shape}）。np.hstackできないので実装を止めて報告する")

    # ---- ローリングバッファの用意 --------------------------------------------
    maxlen = int(SEC_BEFORE * FPS) + 2
    buf = collections.deque(maxlen=maxlen)

    n_before = u._orienting.n_saccades
    hit_tick = None
    hit_info = None
    after_frames = []
    after_needed = int(SEC_AFTER * FPS)

    t0 = time.time()
    tick = 0
    n_fires = 0
    done = False
    for i in range(MAX_STEPS):
        a = env.action_space.sample()
        _obs, _r, te, tr, _info = env.step(a)
        tick += 1

        if tick % RENDER_EVERY == 0:
            imgs = u.get_vision_obs()
            fr = np.hstack([imgs["eye_left"], imgs["eye_right"]])
            if hit_tick is None:
                buf.append(fr)
            else:
                after_frames.append(fr)

        cur_sacc = u._orienting.n_saccades
        if cur_sacc > n_before + n_fires:
            n_fires = cur_sacc - n_before
            if hit_tick is None:
                hit_tick = tick
                hit_info = {"tick": tick, "step": i, "n_saccades": cur_sacc,
                            "strength": float(getattr(u._orienting, "strength", float("nan")))}
                print(f"[hit] tick={tick} step={i} n_saccades={cur_sacc}", flush=True)

        if hit_tick is not None and len(after_frames) >= after_needed:
            done = True
            break
        if te or tr:
            env.reset()
            # 注意：resetでn_saccadesがリセットされる可能性がある
            #   （e_toy_env.py の reset_model は self._orienting.reset() を呼ぶ）。
            #   発火前ならn_beforeを取り直し、発火後（hit_tick確定後）なら継続する
            #   （切り出しに必要な after_frames を集めきるまではresetしても問題ない）。
            if hit_tick is None:
                n_before = u._orienting.n_saccades
                n_fires = 0

    elapsed = time.time() - t0
    print(f"探索にかかった時間: {elapsed:.1f}秒（{tick}物理tick、"
          f"1tickあたり{elapsed / max(tick, 1):.5f}秒）", flush=True)

    if hit_tick is None:
        print("[想定外] MAX_STEPS以内に視線誘導反射のサッケードが"
              "見つからなかった。", flush=True)
        print(f"  探索した物理tick数: {tick} / {MAX_STEPS}", flush=True)
        env.close()
        return

    frames = list(buf) + after_frames
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    h, w, _c = frames[0].shape
    vw = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for fr in frames:
        arr = fr
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"RECORDED {OUT_PATH} ({len(frames)}フレーム, {FPS:.0f}fps, "
          f"約{len(frames) / FPS:.1f}秒, フレーム形状={frames[0].shape})", flush=True)
    print(f"[hit_info] {hit_info}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
