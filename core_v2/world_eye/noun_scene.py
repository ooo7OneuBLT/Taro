"""段階9：目だけの場面で「名詞を厚く」を回すための部屋と置き方と親（体なしの試験場 tools/noun_bed.py の場面を、体ありへ移したもの）。

noun_bed の親の場面がそのまま仕様（仕様書 `2026-10-18_段階9_仕様書_名詞を厚く.md` §2・§4）：
* 場面1つ＝部屋を1つ選び、その部屋の語から3つ、覚える個体（個体0〜2）を1つずつ並べる。12秒。場面と場面のあいだは3秒、何も無い
* 太郎が物を見始めたら（noun_bed では乱数で 2〜4秒見る。ここでは太郎が自分で目を動かす）、親は
    - 見始めて 0.5秒：音のある物なら、P_SOUND（半分）の割合で動かして鳴らす（env.object_sound）
    - 見始めて 1.5秒：P_NAME（半分）の割合で名前を言う（env.say。言い方は段階5の話しかけと同じ：1割は語だけ、残りは文の型）
    - 見ている間、1秒ごとに物を回す（向きが隣へ。見た目の素の表の向きの番号と同じ並び）
  「見ている」＝視線と物の中心の角が LOOK_DEG（10度）か物の見かけの半分の大きい方より小さく、いちばん近い物
* 物の向き：見た目の素の表（tools/make_appearance_bank.VIEWS、24通り）と同じ左右・上下の回し方。物が太郎から見て横にずれても
  同じ面が太郎へ向くように、ずれた分だけ上の軸のまわりに回す
* 置き方：目から下へ EL0（20度、見た目の素の表と同じ）の向きを真ん中に、2つなら横に、3つなら三角、4つなら 2×2。
  物どうしの角度の間は、いちばん大きい物が重ならないだけ（最小 MIN_SEP_DEG＝17度＝noun_bed の8マス）。距離は自作 35cm・本物 50cm
  （本物は 8〜25cm と大きいので、近いと3つが視野に入らない）［Tier3・モデル上の選択］

この形は、目の入口が本物（DINOv2、PC）でも代わりの目（eye_backend.TableEye、クラウド）でも同じ。
"""

from __future__ import annotations

import math
import random

import numpy as np

from . import instances as ins
from . import scene as scene_mod

ROOMS = ("食事", "お風呂", "遊び")
EL0 = -20.0
DIST = {"made": 0.35, "real": 0.5}
MIN_SEP_DEG = 17.0
GAP_M = 0.03
LOOK_DEG = 10.0
SCENE_S = 12.0
GAP_S = 3.0
N_SHOW = 3
NAME_AT_S = 1.5
SOUND_AT_S = 0.5
P_NAME = 0.5
REPEAT_LABELS = 2       # 名前を言ったあと、太郎が見続けていれば言い直す回数（旧太郎 parent.py の連呼 repeat_labels。親は1回の提示で同じ語を繰り返す：Gogate 2000）
REPEAT_GAP_S = 2.0      # 言い直しの間隔（旧太郎の repeat_gap_sec と同じ）
P_SOUND = 0.5
TURN_EVERY_S = 1.0
TICK_S = 0.1
N_VIEWS = 24


def views():
    from ..tools.make_appearance_bank import VIEWS
    return VIEWS


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def view_quat(v: int, az_deg: float = 0.0) -> np.ndarray:
    """向き v（表の左右・上下）で置き、太郎から見て横に az 度ずれた所でも同じ面が太郎へ向くように回す。"""
    from ..tools.make_appearance_bank import _quat
    V = views()[v % N_VIEWS]
    a = math.radians(-az_deg) / 2
    return _qmul(np.array([math.cos(a), 0.0, 0.0, math.sin(a)]), _quat(V["yaw"], V["pitch"]))


def direction(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return (math.cos(el) * (math.cos(az) * scene_mod.FORWARD + math.sin(az) * scene_mod.RIGHT)
            + math.sin(el) * scene_mod.UP)


def layout(n: int, smax: float, dist: float) -> list[tuple[float, float]]:
    """物 n 個の (左右の角, 上下の角)（度、右・上が正）。真ん中は (0, EL0)。"""
    sep = max(MIN_SEP_DEG, math.degrees(2.0 * math.atan((smax + GAP_M) / (2.0 * dist))))
    h = sep * math.sqrt(3.0) / 4.0
    if n == 1:
        pts = [(0.0, 0.0)]
    elif n == 2:
        pts = [(-sep / 2, 0.0), (sep / 2, 0.0)]
    elif n == 3:
        pts = [(0.0, h), (-sep / 2, -h), (sep / 2, -h)]
    elif n == 4:
        pts = [(-sep / 2, sep / 2), (sep / 2, sep / 2), (-sep / 2, -sep / 2), (sep / 2, -sep / 2)]
    else:
        raise ValueError(f"置ける物は1〜4つ（{n}）")
    return [(a, EL0 + e) for a, e in pts]


def scene_dict(room: str, objects: str = "made", words=None, insts=None, px: int = 224) -> dict:
    """EyeWorldEnv に渡す場面の辞書（run/scenes の JSON は使わない）。"""
    s9 = {"room": room, "objects": objects}
    if words is not None:
        s9["words"] = list(words)
    if insts is not None:
        s9["insts"] = list(insts)
    return {"name": f"段階9_{room}", "body": {"vision_px": px},
            "world": {"word_test": {"schedule": []}, "eye_world": {"stage9": s9}}}


def make_env(room: str, objects: str = "made", log_dir: str | None = None, save_images: bool = True,
             words=None, insts=None, seed: int = 0):
    from .env import EyeWorldEnv
    env = EyeWorldEnv(scene_dict(room, objects, words, insts), seed=seed, log_dir=log_dir, save_images=save_images)
    env.n9_objects = objects
    return env


def body_of(env, word: str, k: int) -> str:
    for nm, m in env.n9.items():
        if m["word"] == word and m["k"] == k:
            return nm
    raise KeyError(f"この部屋に {word} の個体{k} が無い")


class Placed:
    """場面に出した物1つ：体の名前・語・個体・向き・置いた角。"""

    def __init__(self, name, word, k, v, az, el, dist):
        self.name, self.word, self.k, self.v = name, word, k, v
        self.az, self.el, self.dist = az, el, dist


def place(env, items: list[tuple[str, int, int]], cause: str = "scene") -> list[Placed]:
    """items＝[(語, 個体, 向き)]。並べて見せる（env.show、前後の画像と events.csv が残る）。"""
    objs = getattr(env, "n9_objects", "made")
    dist = DIST[objs]
    names = [body_of(env, w, k) for w, k, _ in items]
    smax = max(env.n9[n]["size"] for n in names)
    out = []
    for (w, k, v), nm, (az, el) in zip(items, names, layout(len(items), smax, dist)):
        pos = scene_mod.EYE_LEFT + dist * direction(az, el)
        env.show(nm, pos, view_quat(v, az), cause=cause)
        env.n9_view[nm] = v
        out.append(Placed(nm, w, k, v, az, el, dist))
    return out


def turn(env, p: Placed, step: int, cause: str = "parent_turn") -> None:
    """親が物を回す（向きの番号が隣へ）。"""
    p.v = (p.v + step) % N_VIEWS
    pos, _ = env.pose(p.name)
    env.move(p.name, pos, view_quat(p.v, p.az), cause=cause)
    env.n9_view[p.name] = p.v


def clear(env, placed: list[Placed], cause: str = "scene_end") -> None:
    for p in placed:
        env.hide(p.name, cause=cause)
        env.n9_view.pop(p.name, None)


def looking_at(env, placed: list[Placed]) -> Placed | None:
    best = None
    for p in placed:
        a = env._gaze_angle_to(p.name)
        if a is None:
            continue
        lim = max(LOOK_DEG, math.degrees(math.atan(0.5 * env.n9[p.name]["size"] / p.dist)))
        if a <= lim and (best is None or a < best[0]):
            best = (a, p)
    return best[1] if best else None


def naming_utterance(word: str, rng: random.Random) -> str:
    """親の名づけの言い方：段階5の話しかけと同じ（P_ALONE で語だけ、残りは文の型から1つ）。"""
    from ..tools.parent_speech import FRAMES, P_ALONE
    if rng.random() < P_ALONE:
        return word
    return rng.choice(FRAMES).format(w=word)


class NounParent:
    """1回（0.1秒）ごとに呼ぶ親。太郎が見ている物を見て、鳴らす・名前を言う・回す。"""

    def __init__(self, rng: random.Random, p_name: float = P_NAME, p_sound: float = P_SOUND,
                 repeat: int = REPEAT_LABELS) -> None:
        self.rng = rng
        self.p_name, self.p_sound = p_name, p_sound
        self.repeat = repeat
        self.stats = {"names": 0, "sounds": 0, "turns": 0, "looks": 0, "look_ticks": 0}
        self.named: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.cur = None
        self.d = 0
        self.say = self.ring = False

    def tick(self, env, placed: list[Placed]) -> dict:
        """戻り値：その回に親がしたこと {"said": 文, "rang": 語, "looking": 語}。"""
        from . import sounds as snd
        p = looking_at(env, placed)
        did = {"looking": p.word if p else ""}
        if p is not self.cur:
            self.cur, self.d = p, 0
            if p is not None:
                self.stats["looks"] += 1
                self.say = self.rng.random() < self.p_name
                self.ring = ins.KIND_OF[p.word] in snd.BASE_LEN and self.rng.random() < self.p_sound
            return did
        if p is None:
            return did
        self.d += 1
        self.stats["look_ticks"] += 1
        if self.d % int(round(TURN_EVERY_S / TICK_S)) == 0:
            turn(env, p, self.rng.choice((-1, 1)))
            self.stats["turns"] += 1
        if self.ring and self.d == int(round(SOUND_AT_S / TICK_S)):
            wave = snd.synth(ins.KIND_OF[p.word], p.k, seed=self.rng.randrange(10 ** 6))
            env.object_sound(p.name, wave)
            self.stats["sounds"] += 1
            did["rang"] = p.word
        at, gap = int(round(NAME_AT_S / TICK_S)), int(round(REPEAT_GAP_S / TICK_S))
        if self.say and self.d >= at and (self.d - at) % gap == 0 and (self.d - at) // gap <= self.repeat:
            text = naming_utterance(p.word, self.rng)
            env.say(text, target=p.name, cause="parent_name")
            self.stats["names"] += 1
            self.named[p.word] = self.named.get(p.word, 0) + 1
            did["said"] = text
        return did
