"""家の場面の太郎：41係の脳と目を、AI2-THOR の家（ThorEnv）につなぐ。

1回（0.1秒）の回し方は目だけの場面と同じ（grow_names.EyeTaro.tick をそのまま使う：環境10歩 → 脳1回 → 目の指令を
env._orienting.set_map_target へ。太郎の声は次の回に自分の耳へ戻す）。違うのは環境（部屋ごとの MuJoCo ではなく、
家1つ）と目（eye.install_thor_eye）と親（parent.FollowInParent）だけ。

* ThorTaro：脳＋目＋家。face_home() で目を最初の向き（真正面・下へ10度）に置く（親が座らせる。EyeTaro.face_table と同じ役）
* ThorGrower：意味の辞書と耳を育てながら（grow_names.Grower と同じ係を learning=True）、見えない親の手と回す。
  run(秒) で回し、見た時間（LookLog）と親のしたこと（名づけの回数など）を数える。写しはまだ作らない（計画②で）
"""

from __future__ import annotations

import random

import torch

from ..tools import growth
from ..tools.grow_names import EyeTaro
from .parent import TICK_S, FollowInParent, LookLog

FROM_PHASE = 6      # start="auto" のとき：段5以下の、目だけの場面で育った終わりの写しのうち、いちばん上の段の最新


def make_env(scene: str = "FloorPlan1", fake: bool = False, px: int = 224, render_px: int | None = None,
             platform: str | None = None, log_dir: str | None = None, seed: int = 0, **kw):
    if fake:
        from .fake import fake_env
        return fake_env(scene, px=px, render_px=render_px, log_dir=log_dir, seed=seed, **kw)
    from .env import ThorEnv
    return ThorEnv(scene, px=px, render_px=render_px, platform=platform, log_dir=log_dir, seed=seed, **kw)


class ThorTaro(EyeTaro):
    """env を渡せばそれを使う（テストは偽物の家）。eye：dinov2｜seg。"""

    def __init__(self, seed: int = 0, eye: str = "dinov2", env=None, start: str = "auto", **env_kw) -> None:
        from ..brain import Brain
        from .eye import install_thor_eye
        self.seed = seed
        self.objects = "thor"
        torch.manual_seed(seed)
        self.brain = Brain()
        self.start_row = None
        if start == "auto":
            self.start_row = growth.start_point(FROM_PHASE, seed=seed, world="eye")
        elif start not in ("birth", ""):
            self.start_row = growth.find(start)
        if self.start_row is not None:
            growth.load(self.brain.boxes, self.start_row)
        self.env = env if env is not None else make_env(seed=seed, **env_kw)
        self.envs = {"home": self.env}
        self.eye_info = install_thor_eye(self.brain, eye, env_ref=lambda: self.env)
        self.eye_info["backend"] = eye
        self.n_tick = 0
        self._echo = ""
        self.face_home("start")

    def face_home(self, cause: str = "start") -> None:
        """目を最初の向き（真正面・下へ BASE_TILT）に置き、控えていた跳びの指令を消す。"""
        self.env.set_eye(0.0, self.env.base_tilt, cause=cause)
        self.env._orienting._map_target = None

    face_table = face_home

    def save_images(self, on: bool) -> None:
        pass

    def close(self) -> None:
        self.env.close()


class ThorGrower(ThorTaro):
    def __init__(self, *a, repeat: int | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        b = self.brain.boxes
        b["semantic"].learning = True
        b["listen"].learning = True
        self.rng = random.Random(self.seed)
        self.parent = FollowInParent(self.rng, **({} if repeat is None else {"repeat": repeat}))
        self.looks = LookLog()
        self.events: list[dict] = []
        self.n_learn = 0
        self.last_said = ""
        self.last_named = ""
        self.learn_by_word: dict[str, int] = {}
        self.n_scenes = 0

    def step_once(self) -> dict:
        """親1回 → 太郎1回。戻り値は親のしたこと（parent.tick）に "got"（太郎の1回）を足したもの。"""
        did = self.parent.tick(self.env)
        self.looks.add(self.parent.looking, did["looking"])
        got = self.tick()
        t = round(self.n_tick * TICK_S, 1)
        for key in ("brought", "said", "put"):
            if key in did:
                self.events.append({"t": t, "event": key, "word": did.get("holding") or did.get("brought") or did.get("put", ""),
                                    "text": did.get("said", ""), "looking": did["looking"]})
        if "said" in did:
            self.last_said, self.last_named = did["said"], did.get("holding", "")
        sem = self.brain.boxes["semantic"]
        if sem.last_event in ("learn", "learn_skip"):
            self.n_learn += sem.last_event == "learn"
            if sem.last_event == "learn" and self.last_named:
                self.learn_by_word[self.last_named] = self.learn_by_word.get(self.last_named, 0) + 1
            self.events.append({"t": t, "event": sem.last_event, "word": "", "text": self.last_said,
                                "looking": did["looking"]})
        did["got"] = got
        return did

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / TICK_S))):
            self.step_once()
        self.looks.flush()

    def summary(self) -> dict:
        return {"ticks": self.n_tick, "seconds": round(self.n_tick * TICK_S, 1), "parent": dict(self.parent.stats),
                "named_by_word": dict(self.parent.named), "looks": self.looks.summary(),
                "semantic_learn": self.n_learn, "learn_by_word": dict(self.learn_by_word),
                "eye": self.eye_info, "start": self.start_name(), "scene": self.env.scene_name,
                "thor_calls": self.env.n_calls, "thor_fail": self.env.n_fail,
                "saccades": self.env._orienting.n_saccades}
