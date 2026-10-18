"""目だけのテストの問題集を作る（段階3 作り直し・2026-10-16）。

太郎の脳は回さない。段階2bの場面で、目を乱数の向きへ跳ばしながら左右の目の画像を撮り、
各コマで「3つの物がどの画素にあるか」の正解を、物を1つずつ消して撮り直した差から作る。
目が跳んだ直後のコマ（指令の次の回）も含める（jump_age=1）。

使い方：
  python G/scripts/make_eye_dataset.py [--seeds 5] [--seed-start 0] [--ticks 20] [--out G/data/目のテスト]
  別の問題集（決めすぎの確かめ用）：--seed-start 5 --ticks 10 --out G/data/目のテスト_確かめ
出力：<out>/<場面>_seed<種>_t<回>.npz（img_l, img_r, masks[3,224,224], jump_age, ang）と index.json
"""

from __future__ import annotations

import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)
import core_v2  # noqa: E402,F401  負荷の決まり（core_v2/perf.py：スレッド1本など）を numpy・torch より先に効かせる
import numpy as np
os.chdir(REPO)

from core_v2 import run_v2  # noqa: E402

SCENES = ["段階2b_3物体_ボール左_2026-10-15", "段階2b_3物体_ボール右_2026-10-15"]
NAMES = ["ball", "cup", "distractor"]
DIFF_MIN = 30          # 消す前後で RGB の差の和がこれより大きい画素を、その物の画素とする
JUMP_EVERY = 4         # 何回ごとに目を跳ばすか
JUMP_RANGE = 0.6       # 跳ばす向きの幅（[-1,1] の方向、0.6 ≒ 視野の6割）


def grab(u) -> tuple[np.ndarray, np.ndarray]:
    u._vision_cache = None
    obs = u._get_obs()
    return np.array(obs["eye_left"], dtype=np.uint8), np.array(obs["eye_right"], dtype=np.uint8)


def object_masks(u, base_l: np.ndarray) -> np.ndarray:
    masks = np.zeros((len(NAMES), base_l.shape[0], base_l.shape[1]), dtype=bool)
    for k, name in enumerate(NAMES):
        bid = u.model.body(run_v2.OBJECTS[name]).id
        gids = [g for g in range(u.model.ngeom) if u.model.geom_bodyid[g] == bid]
        old = u.model.geom_rgba[gids].copy()
        u.model.geom_rgba[gids, 3] = 0.0
        hid, _ = grab(u)
        u.model.geom_rgba[gids] = old
        masks[k] = np.abs(base_l.astype(np.int16) - hid.astype(np.int16)).sum(-1) > DIFF_MIN
    return masks


def main() -> None:
    args = sys.argv[1:]
    def opt(name, default):
        return type(default)(args[args.index(name) + 1]) if name in args else default
    n_seeds, n_ticks = opt("--seeds", 5), opt("--ticks", 20)
    seed0 = opt("--seed-start", 0)
    out = os.path.join(REPO, opt("--out", "G/data/目のテスト"))
    os.makedirs(out, exist_ok=True)
    index = []
    for scene in SCENES:
        env = run_v2.build_env(scene, None, seed=0, silent=True)
        u = env.unwrapped
        act = run_v2.zero_action(env)
        try:
            for seed in range(seed0, seed0 + n_seeds):
                rng = np.random.default_rng(1000 + seed)
                env.reset(seed=seed)
                u._orienting._map_target = None
                jump_age = 99
                for t in range(n_ticks):
                    if t % JUMP_EVERY == 0:
                        h, v = rng.uniform(-JUMP_RANGE, JUMP_RANGE, size=2)
                        u._orienting.set_map_target(float(h), float(v))
                        jump_age = 0
                    for _ in range(run_v2.K):
                        env.step(act)
                    jump_age += 1
                    img_l, img_r = grab(u)
                    masks = object_masks(u, img_l)
                    ang = run_v2.gaze_angles(env)
                    name = f"{scene}_seed{seed}_t{t:02d}.npz"
                    np.savez_compressed(os.path.join(out, name), img_l=img_l, img_r=img_r,
                                        masks=masks, jump_age=jump_age,
                                        ang=np.array([ang[k] for k in NAMES]))
                    index.append({"file": name, "scene": scene, "seed": seed, "t": t,
                                  "jump_age": jump_age,
                                  "visible": [int(m.sum()) for m in masks]})
                print(f"{scene} seed{seed}: {n_ticks}枚", flush=True)
        finally:
            run_v2.close_env(env)
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"合計 {len(index)} 枚 → {out}")


if __name__ == "__main__":
    main()
