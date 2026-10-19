"""育ちの段3（名前）：目だけの場面で、20語・部屋3つ・1語4個体（自作か本物）の名づけで太郎を育て、段3の写しを残す。

体なしの試験場（tools/noun_bed.py）の親の場面を、そのまま目だけの場面へ移した（world_eye/noun_scene.py の説明）：
場面1つ＝部屋を1つ選び、その部屋の語から3つ、覚える個体（個体0〜2）を並べて12秒。太郎は自分で目を動かす（41係の脳そのまま）。
太郎が物を見始めて 0.5秒で音のある物を半分の割合で鳴らし、1.5秒で半分の割合で名前を言う。見ている間は1秒ごとに物を回す。
場面と場面のあいだは3秒、何も無い。ほうびは無い。
場面の頭ごとに、親が太郎を机の方へ向けて座らせる（目を最初の向き (0, EL0) に置く。EyeTaro.face_table）。

* 始める太郎：育ち帳の、段2（無ければ段1）の終わりの写し（growth.start_point）。どちらも無ければ生まれたまま（そう表示する）
* 育つ係：意味の辞書（まばら版、既定）・耳（名づけの中でも聞き続ける。段1の耳は grow_early の既定で20語を知っている）・
  種類の層（--kind on のときだけ。既定は止めたまま＝段階9で合格した形）
* 目：--eye dinov2（本物の目の入口、PC）／--eye table（代わりの目：DINOv2 の見た目の素の表を引く。画像から切り出さない。クラウドの確かめ用）
* 物：--objects made（自作の形）／real（本物の物、G/assets が要る＝PC）
* 終わりに growth.save で段3の写し（world=eye、目、物、種、場面の数、元の写し。git の版は growth が書く）

使い方：
  python -m core_v2.tools.grow_names --check [--objects real] [--eye dinov2]   … 1場面だけ回し、前後の画像を残す（写しは作らない）
  python -m core_v2.tools.grow_names [--seed 0] [--scenes 300] [--objects made|real] [--eye dinov2|table]
                                     [--kind off|on] [--from 写しの名前|auto|birth] [--image-scenes 3] [--log-dir 段階9_段3_目]
出力：G/logs/<log-dir>_<物>_<目>_種<seed>/ の 部屋ごとの前後の画像と events.csv（--image-scenes の場面まで画像、そのあとは events.csv だけ）・
      出来事.csv（名づけ・音・覚えた回）・まとめ.json。育ち帳に段3の写し
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch

from .. import run_v2
from ..world_eye import instances as ins
from ..world_eye import noun_scene as ns
from . import growth

K = run_v2.K
TICK_S = ns.TICK_S


class EyeTaro:
    """目だけの場面の部屋（部屋ごとに1つの環境）と、41係の脳と、目。段3の育ちと段階9の試しで共通。"""

    def __init__(self, seed: int, objects: str = "made", eye: str = "table", table: str | None = None,
                 log_root: str | None = None, start: str = "auto", stage: int = 3, all_insts: bool = False) -> None:
        from .. import world_eye  # noqa: F401  （描き方を mujoco より先に決める）
        from ..brain import Brain
        from ..world_eye.eye_backend import install_eye
        ins.use_objects(objects)
        self.seed, self.objects, self.log_root = seed, objects, log_root
        self.all_insts = all_insts
        torch.manual_seed(seed)
        self.brain = Brain()
        self.start_row = None
        if start == "auto":
            self.start_row = growth.start_point(stage, seed=seed, world="eye")
        elif start not in ("birth", ""):
            self.start_row = growth.find(start)
        if self.start_row is not None:
            growth.load(self.brain.boxes, self.start_row)
        self.envs: dict[str, object] = {}
        self.env = None
        self.eye_info = install_eye(self.brain, eye, env_ref=lambda: self.env, table=table, objects=objects)
        self.eye_info["backend"] = eye
        self.n_tick = 0

    def start_name(self) -> str:
        return self.start_row["name"] if self.start_row else "生まれたまま"

    def use_room(self, room: str):
        prev = self.env
        if room not in self.envs:
            log = os.path.join(self.log_root, room) if self.log_root else None
            words = None if self.all_insts else [w for w in ins.WORDS if ins.ROOM_OF[w] == room]
            insts = None if self.all_insts else ins.TRAIN_INST
            env = ns.make_env(room, self.objects, log_dir=log, words=words, insts=insts, seed=self.seed)
            env.reset(seed=self.seed)            # 記録のフォルダを開く
            env.set_eye(0.0, ns.EL0, cause="start")
            self.envs[room] = env
        env = self.envs[room]
        if prev is not None and prev is not env:   # 部屋が替わっても目の向きはそのまま（場面の頭で face_table が戻す）
            env.set_eye(prev._orienting.pan, prev._orienting.tilt, cause="room_change")
        self.env = env
        return env

    def face_table(self, cause: str = "scene_start") -> None:
        """親が太郎を机の方へ向けて座らせる（環境の側。太郎の跳びではない）：目を最初の向き (0, EL0) に置く。
        場面・試しの頭ごとに呼ぶ。呼ばないと、目は前の場面の終わりの向き（可動域の端のことがある）のまま始まり、
        物が視野の外だと何も見えず、目を動かすきっかけも無いので、そのまま動かない（2026-10-18、PC の段3）。
        控えていた跳びの指令も消す（run_v2.run_trial の試行の頭と同じ）。"""
        self.env.set_eye(0.0, ns.EL0, cause=cause)
        self.env._orienting._map_target = None

    def save_images(self, on: bool) -> None:
        for e in self.envs.values():
            e.save_images = on

    def tick(self) -> dict:
        """環境10歩→脳1回→目の指令。戻り値：{"heard", "sound"}。"""
        env = self.env
        heard, sound = [], None
        for _ in range(K):
            obs, _r, _t, _tr, info = env.step()
            if info.get("object_sound") is not None:
                sound = info["object_sound"]
            utt = info.get("parent_utterance")
            if utt and utt.get("text"):
                heard.append(str(utt["text"]))
        # 段階9：太郎の声も次の回に自分の耳へ戻す（人間も自分の声を聞く。run_v2・say_names と同じ）。耳は自分の声の写しに
        # 合う音だけを自分の声とし、同じ回の親の声はふつうに聞く
        ext = {"eyein": obs["eye_left"], "earin": " ".join(x for x in [self._echo] + heard if x)}
        if sound is not None:
            ext["soundin"] = sound
        to_env = self.brain.tick(ext)
        self._echo = to_env.get("voiceout", "") or ""
        cmd = to_env.get("eyeout")
        if cmd is not None:
            env._orienting.set_map_target(float(cmd[0]), float(cmd[1]))
        self.n_tick += 1
        return {"heard": " ".join(heard), "sound": sound is not None}

    _echo = ""        # 前の回に太郎が出した声（この回に耳へ戻る）

    def close(self) -> None:
        for e in self.envs.values():
            e.close()


class Grower(EyeTaro):
    def __init__(self, *a, kind: str = "off", repeat: int = ns.REPEAT_LABELS, **kw) -> None:
        super().__init__(*a, **kw)
        self.repeat = repeat
        b = self.brain.boxes
        b["semantic"].learning = True
        b["listen"].learning = True
        b["kind"].learning = kind == "on"
        self.kind = kind
        self.rng = random.Random(self.seed)
        self.parent = ns.NounParent(self.rng, repeat=self.repeat)
        self.events: list[dict] = []
        self.n_scenes = 0
        self.n_learn = 0
        self.last_said = ""
        self.last_named = ""                       # 親が最後に名前を言った物の語
        self.learn_by_word: dict[str, int] = {}    # 覚えた回を、その前に親が最後に名前を言った物の語で数える（目安）

    def scene(self) -> None:
        rng = self.rng
        room = rng.choice(ns.ROOMS)
        env = self.use_room(room)
        self.face_table()
        words = rng.sample([w for w in ins.WORDS if ins.ROOM_OF[w] == room], ns.N_SHOW)
        items = [(w, rng.choice(ins.TRAIN_INST), rng.randrange(ns.N_VIEWS)) for w in words]
        placed = ns.place(env, items)
        self.parent.reset()
        sem = self.brain.boxes["semantic"]
        for t in range(int(round(ns.SCENE_S / TICK_S))):
            did = self.parent.tick(env, placed)
            got = self.tick()
            if "said" in did:
                self.last_said = did["said"]
                self.last_named = did["looking"]
            for key in ("said", "rang"):
                if key in did:
                    self.events.append({"scene": self.n_scenes, "room": room, "t": round(t * TICK_S, 1), "event": key,
                                        "word": did.get("rang", ""), "text": did.get("said", ""),
                                        "looking": did["looking"]})
            if sem.last_event in ("learn", "learn_skip"):
                # 覚え方B は名残りが消えてから（聞いて数秒あと）覚える。結び付くのは聞く前3秒に見ていた物なので、
                # 「looking」はその回に見ている物（参考）、「text」は親が最後に言った文
                self.n_learn += sem.last_event == "learn"
                if sem.last_event == "learn" and self.last_named:
                    self.learn_by_word[self.last_named] = self.learn_by_word.get(self.last_named, 0) + 1
                cur = self.parent.cur.word if self.parent.cur else ""
                self.events.append({"scene": self.n_scenes, "room": room, "t": round(t * TICK_S, 1),
                                    "event": sem.last_event, "word": "", "text": self.last_said, "looking": cur})
        ns.clear(env, placed)
        self.parent.reset()
        for _ in range(int(round(ns.GAP_S / TICK_S))):
            self.tick()
        self.n_scenes += 1


def summary(g: Grower, wall_s: float) -> dict:
    ear = g.brain.boxes["listen"]
    known = ear.known()
    return {"scenes": g.n_scenes, "ticks": g.n_tick, "wall_s": round(wall_s, 1),
            "s_per_scene": round(wall_s / max(1, g.n_scenes), 2), "parent": g.parent.stats,
            "named_by_word": {w: g.parent.named.get(w, 0) for w in ins.WORDS},
            "semantic_learn": g.n_learn, "learn_by_word": {w: g.learn_by_word.get(w, 0) for w in ins.WORDS},
            "learn_min": min(g.learn_by_word.get(w, 0) for w in ins.WORDS), "ear_knows": {w: w in known for w in ins.WORDS},
            "ear_knows_n": sum(w in known for w in ins.WORDS), "start": g.start_name(), "eye": g.eye_info,
            "objects": g.objects, "kind": g.kind}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--objects", default="made", choices=["made", "real"])
    ap.add_argument("--eye", default="dinov2", choices=["dinov2", "table"])
    ap.add_argument("--table", default=None, help="代わりの目の見た目の素の表（既定：物に合わせて core_v2/data/段階9_見た目の素*.pt）")
    ap.add_argument("--kind", default="off", choices=["off", "on"])
    ap.add_argument("--from", dest="start", default="auto", help="auto（段2→段1→生まれたまま）／birth／写しの名前")
    ap.add_argument("--image-scenes", type=int, default=3, help="前後の画像を残す場面の数（そのあとは events.csv だけ）")
    ap.add_argument("--log-dir", default="段階9_段3_目")
    ap.add_argument("--name", default=None, help="写しの名前（既定 段3_目_<物>_<目>_種<seed>）")
    ap.add_argument("--repeat", type=int, default=ns.REPEAT_LABELS, help="親が見続けている物の名前を言い直す回数（0＝1回だけ＝2026-10-19より前）")
    ap.add_argument("--check", action="store_true", help="1場面だけ回して画像を残す（写しは作らない）")
    a = ap.parse_args(argv)
    if a.check:
        a.scenes, a.image_scenes = 1, 1
    tag = f"{a.objects}_{a.eye}_種{a.seed}" + ("_確かめ" if a.check else "")
    log_root = os.path.join(growth.REPO, "G", "logs", f"{a.log_dir}_{tag}")
    os.makedirs(log_root, exist_ok=True)
    g = Grower(a.seed, a.objects, a.eye, a.table, log_root=log_root, start=a.start, kind=a.kind, repeat=a.repeat)
    print(f"[段3] 始める太郎：{g.start_name()}　目：{g.eye_info}　物：{a.objects}　場面 {a.scenes}", flush=True)
    t0 = time.time()
    try:
        for s in range(a.scenes):
            if s == a.image_scenes:
                g.save_images(False)
            g.scene()
            if (s + 1) % 10 == 0 or s + 1 == a.scenes:
                el = time.time() - t0
                print(f"[段3] 場面 {s + 1}/{a.scenes}　{el:.0f}秒（残り約 {el / (s + 1) * (a.scenes - s - 1) / 60:.0f}分）　"
                      f"名づけ {g.parent.stats['names']}　音 {g.parent.stats['sounds']}　覚えた {g.n_learn}"
                      f"（いちばん少ない語 {min(g.learn_by_word.get(w, 0) for w in ins.WORDS)}回）　"
                      f"耳が知っている語 {sum(w in g.brain.boxes['listen'].known() for w in ins.WORDS)}/20", flush=True)
        res = summary(g, time.time() - t0)
    finally:
        g.close()
    with open(os.path.join(log_root, "出来事.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "room", "t", "event", "word", "text", "looking"])
        w.writeheader()
        w.writerows(g.events)
    if not a.check:
        name = a.name or growth.free_name(f"段3_目_{a.objects}_{a.eye}_種{a.seed}")
        row = growth.save(g.brain.boxes, 3, name, end=True, parent=g.start_row["name"] if g.start_row else None,
                          conditions={"world": "eye", "eye": a.eye, "eye_info": g.eye_info, "objects": a.objects,
                                      "seed": a.seed, "scenes": a.scenes, "kind": a.kind, "repeat": a.repeat, "words": 20, "rooms": 3,
                                      "semantic": g.brain.boxes["semantic"].memory},
                          note=f"名づけ {res['parent']['names']}・音 {res['parent']['sounds']}・覚えた {res['semantic_learn']}・"
                               f"耳が知っている語 {res['ear_knows_n']}/20")
        res["snapshot"] = row["name"]
        print(f"[段3] 写し：{row['name']}（{row['file']}）", flush=True)
    res["log_root"] = log_root
    with open(os.path.join(log_root, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("ear_knows", "named_by_word", "learn_by_word")}, ensure_ascii=False,
                     default=str), flush=True)
    return res


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
