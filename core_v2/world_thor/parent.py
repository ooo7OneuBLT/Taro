"""家の場面の親：「見えない親の手」（follow-in。旧太郎のよい親の決まりを引き継ぐ）。

親は太郎の見ている物に合わせる（follow-in labeling：Tomasello・Farrar 1986。子が見ている物に親が名前を言うと、
子の注意を移させて言うより語がよく残る。旧太郎 parent.py の「見ている物に合わせて言う」と同じ考え）：
* 太郎がある物を BRING_AFTER_S（1秒）続けて見たら（目の真ん中に写っている物＝環境の正解 centre_object）、
  親がその物を手に取り、目の前 HOLD_DIST_M（0.3m）、そのときの視線の先へ運んでくる（TeleportObject、forceKinematic）
* 運ぶ（2026-10-19、動画を見て「物がワープして浮いている」と言われて）：前は置き場所から目の前へ1回で飛ばしていた。
  いまは 0.1秒ごとに少しずつ Teleport して、なめらかな道を運ぶ。道は2次のベジェ曲線で、真ん中の点（control）を
  置き場所の真上寄り（目の前へ向かって MOVE_CTRL_FRAC＝1/4、長くても MOVE_CTRL_MAX_M＝0.15m のところ）・両端の高い方より
  MOVE_LIFT_M（0.1m）上に置く（0.2m では床のおもちゃが太郎の目の高さより上へ浮き上がって見えた）
  ＝まず持ち上げて、弧を描いて太郎の前へ来る。速さは最小躍度（minimum-jerk：10s³−15s⁴+6s⁵。人の手の届く動きの形、
  Flash & Hogan 1985）で、ゆっくり動き出してゆっくり止まる。かかる時間は BRING_MOVE_S（1.0秒）か、道の長さ÷MOVE_SPEED_MPS
  （平均 0.6m/秒）の長い方（遠い物ほどゆっくり。1回の動きが 0.15m を超えないように）、長くても MOVE_MAX_S（5秒）。
  運んでくる間、着く所は太郎の今の視線の先（目の前 0.3m）へ 1回に MOVE_FOLLOW_M（4cm）まで寄せていく（親は子の目を見ながら
  持ってくる。寄せないと、運んでいる 1〜1.5秒の間に生まれたての太郎の目がよそへ移り、着いたときには見ていなかった：
  偽の家・脳つき25秒で名づけ 10→1回）。運んでいる間も物は親の手の中（env.in_hand）だが、名前は言わない。
  持つ時間・名づけ・見たかどうかは、着いた回から数える
* 持っている間は、太郎から見て左右に ±SHAKE_M（2.5cm）、SHAKE_HZ（1.5回/秒）でやさしく揺らし、上下にも少し
  （SHAKE_UP_M＝0.8cm、左右の倍の速さ：横に揺らすと手首で少し上下する）。揺れの大きさは着いてから SHAKE_RAMP_S（0.3秒）で
  0 から大きくなる（着いたとたんに振りはじめない）
* 持ってきて（着いて）NAME_AT_S（0.5秒）で、太郎が見ていれば名前を言う（P_NAME＝1.0：この世界では毎回言う）。言い方は段階5の話しかけと同じ
  （noun_scene.naming_utterance：1割は語だけ、残りは文の型）。そのあと太郎が見続けていれば REPEAT_GAP_S（2秒）ごとに
  最大 REPEAT（2）回言い直す（旧太郎の連呼 repeat_labels。noun_scene と同じ）
* HOLD_S（5秒）たったら、来た道を逆にたどって（同じ形の弧：持ち上げて、置き場所の上から下ろす）元の場所へ運び、
  着いたら元の場所・向きに置いて手を離す（put_back）。戻した物は COOLDOWN_S（3秒）は持ってこない（戻したとたんにまた見て、
  同じ物ばかり行き来しないため）［仮置き］
* ひとりで見せる（bring-in。旧太郎の親の solo_presentation：見せたい物を1つだけ子の視線の先へ持ってきて振る）：
  太郎が BRING_IN_AFTER_S（5秒）のあいだ名前を言える物を1つも見ず、親の手が空いていて、戻したばかり（COOLDOWN_S）でなければ、
  親が名前を言える拾える物を1つ選んで持ってくる（ここ BRING_IN_RECENT 回に見せた物は選ばない。目から BRING_IN_NEAR_M（3m）
  以内の物を先に、見せた回数の少ない順、近い順。物を順にめぐる）。持ってくる所と振り方は follow-in と同じ（目の前 0.3m、視線の先）。
  名前は太郎がその物を見てから言う（はじめて見た回から NAME_AT_S 後に1回、そのあと見続けていれば REPEAT_GAP_S ごとに言い直す。
  follow-in で持ってきた時を「見はじめた時」とするのと同じ時間の決め方）。太郎が見たら HOLD_S（5秒）で戻す
  （見はじめが遅くても、はじめの名前を言うまでは持っている）。一度も見なければ BRING_IN_GIVE_UP_S（8秒）で、名前を言わずに戻す。
  持ってきたこと・言ったことは、どちらの決まりか（mode：follow｜bringin）を events と数に残す［Tier3・数は仮置き］
* 名づけは「親がその物を持っていて、太郎がその物を見ている（centre_object＝持っている物）回」だけ（依頼17 の PC の結果を受けて、
  2026-10-19）。前は、はじめの1回（持ってきて 0.5秒）は太郎が見ていなくても言っていた（k==0 は見ていなくてよい、という決まり）。
  持ってきた物がいきなり目の前に来て太郎が目をそらすと、親は「りんご」と言い、太郎は戸棚やトースターを見ていた。
  意味の辞書は名づけの前後に見ていた物と結び付けるので、名前が違う物に付いていた。いまは、言う時が来ても太郎が見ていなければ待ち、
  見たときに言う（言い直しも同じ。言い直しの間は前に言った回から REPEAT_GAP_S）。最後に言ってから NAME_TAIL_S（1秒）は戻さない
  （前は 4.5秒目の言い直しの 0.5秒あとに戻していた）。環境の側でも、持っていない物の名づけは届けない（env.in_hand・say_drop）
* 親がおもちゃを床に並べた場面（env.toys）では、ひとりで見せる物は床のおもちゃから先に選ぶ。戻すのは床の置いた所（env.home）
* 親の手そのものは描かない（AI2-THOR に手が無い。物だけが動く。手で持って運ぶような道と速さにして、浮いて見えるのを減らす）。物の音はまだ無い（音の素が無い。object_sound の口だけある）

語の表（TYPE_WORD）：AI2-THOR の物の種類（objectType。unity/Assets/Scripts/SimObjType.cs で確かめた名前）→ 子どもの語。
台所（FloorPlan1〜30）と居間（FloorPlan201〜230）にある、拾える小さな物から10語。目だけの場面の20語と重なる語
（りんご・ぱん・こっぷ・おさら・すぷうん・ぼおる・くま）は同じかなで書く（耳がもう知っている語）。ほん・りもこん・まくらは新しい語。
くま（TeddyBear）・ぼおる（BasketBall）は寝室に多い（台所・居間に無ければ、その場面では出てこないだけ）。
Mug と Cup はどちらも「こっぷ」（子どもに向けては同じ語で呼ぶ）［Tier3・モデル上の選択］
"""

from __future__ import annotations

import math
import random

import numpy as np

TYPE_WORD = {
    "Apple": "りんご", "Bread": "ぱん", "Cup": "こっぷ", "Mug": "こっぷ", "Plate": "おさら", "Spoon": "すぷうん",
    "Book": "ほん", "RemoteControl": "りもこん", "Pillow": "まくら", "TeddyBear": "くま", "BasketBall": "ぼおる",
}
WORDS = sorted(set(TYPE_WORD.values()), key=list(TYPE_WORD.values()).index)

TICK_S = 0.1
BRING_AFTER_S = 1.0
HOLD_DIST_M = 0.3
SHAKE_M = 0.025         # 左右の揺れ（片側）
SHAKE_UP_M = 0.008      # 上下の揺れ（片側。左右の倍の速さ）
SHAKE_HZ = 1.5
SHAKE_RAMP_S = 0.3      # 着いてから揺れが大きくなるまで
BRING_MOVE_S = 1.0      # 運ぶ時間（短くてもこれだけ。持ってくるのも戻すのも）
MOVE_SPEED_MPS = 0.6    # 運ぶ平均の速さ（遠い物は BRING_MOVE_S より長くかける）
MOVE_MAX_S = 5.0
MOVE_LIFT_M = 0.1       # 道の真ん中の点を、両端の高い方よりこれだけ上に置く（まず持ち上げる。0.2 では目の高さより上へ浮いて見えた）
MOVE_CTRL_FRAC = 0.25   # 道の真ん中の点は、置き場所から目の前へ向かってこれだけの所の上
MOVE_CTRL_MAX_M = 0.15  # ただし横には長くてもこれだけ（遠い物でも、まず上へ持ち上げる）
MOVE_FOLLOW_M = 0.04    # 運んでくる間、着く所を太郎の今の視線の先へ寄せる（1回にこれだけまで）
NAME_AT_S = 0.5
P_NAME = 1.0
REPEAT = 2
REPEAT_GAP_S = 2.0
HOLD_S = 5.0
COOLDOWN_S = 3.0
BRING_IN_AFTER_S = 5.0
BRING_IN_GIVE_UP_S = 8.0
BRING_IN_NEAR_M = 3.0
BRING_IN_RECENT = 3
NAME_TAIL_S = 1.0       # 名づけのあと、少なくともこれだけは持っている（聞いた語の名残り WORD_TRACE_S＝1秒の間、物を見せておく）
MIN_Y_M = 0.05          # 持ってくる所が床より下にならないように


def naming_utterance(word: str, rng: random.Random) -> str:
    from ..world_eye.noun_scene import naming_utterance as nu
    return nu(word, rng)


def min_jerk(s: float) -> float:
    """最小躍度の進み具合（0→1。はじめと終わりで速さも加速度も 0）。"""
    s = min(1.0, max(0.0, float(s)))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def carry_path(start, end, lift_end: bool = False) -> tuple:
    """start→end を運ぶ2次ベジェの3点。真ん中の点は「持ち上げる側」（lift_end=False なら start、True なら end）の
    真上寄り（相手へ横に MOVE_CTRL_FRAC、長くても MOVE_CTRL_MAX_M）で、両端の高い方より MOVE_LIFT_M 上。戻すときは lift_end=True（置き場所の上から下ろす）。"""
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    near, far = (b, a) if lift_end else (a, b)
    flat = (far - near) * np.array([1.0, 0.0, 1.0])
    dist = float(np.linalg.norm(flat))
    ctrl = near + (min(MOVE_CTRL_FRAC * dist, MOVE_CTRL_MAX_M) / dist) * flat if dist > 1e-9 else near.copy()
    ctrl[1] = max(a[1], b[1]) + MOVE_LIFT_M
    return a, ctrl, b


def bezier(p, u: float) -> np.ndarray:
    a, c, b = p
    return (1 - u) ** 2 * a + 2 * u * (1 - u) * c + u * u * b


def path_len(p, n: int = 40) -> float:
    pts = [bezier(p, i / n) for i in range(n + 1)]
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(n)))


def move_ticks(p) -> int:
    t = min(MOVE_MAX_S, max(BRING_MOVE_S, path_len(p) / MOVE_SPEED_MPS))
    return max(1, int(math.ceil(t / TICK_S - 1e-9)))


class FollowInParent:
    """0.1秒ごとに tick(env) を呼ぶ親（上の説明）。"""

    def __init__(self, rng: random.Random, p_name: float = P_NAME, repeat: int = REPEAT,
                 bring_in_after: float | None = BRING_IN_AFTER_S) -> None:
        self.rng = rng
        self.p_name, self.repeat = p_name, repeat
        self.bring_in_after = bring_in_after        # None＝ひとりで見せない（follow-in だけ）
        self.stats = {"brings": 0, "names": 0, "repeats": 0, "puts": 0,
                      "brings_follow": 0, "brings_bringin": 0, "names_follow": 0, "names_bringin": 0,
                      "bringin_looked": 0, "bringin_unlooked": 0}
        self.named: dict[str, int] = {}
        self.presented: dict[str, int] = {}         # ひとりで見せた回数（物の id ごと）
        self.recent: list[str] = []                 # ここ何回かに見せた物
        self.reset()

    def reset(self) -> None:
        self.looking = None        # 太郎が今見ている物（親から見た正解）
        self.look_ticks = 0
        self.held = None           # 持っている物の id（運んでいる間も）
        self.d = 0                 # 着いてからの回数（運んでいる間は 0）
        self.phase = ""            # in＝運んでくる｜hold＝持って見せる｜out＝戻しに運ぶ
        self.path = None           # 運ぶ道（ベジェの3点）
        self.move_n = 0            # 運ぶ回数
        self.move_k = 0            # 運んだ回数
        self.say = False
        self.base = None           # 振る真ん中
        self.axis = None           # 振る向き（太郎から見て左右）
        self.cool: dict[str, int] = {}
        self.word = ""             # 持っている物の語
        self.mode = ""             # follow｜bringin
        self.first_look = None     # 持ってから太郎がはじめて見た回（follow は 0）
        self.no_look = 0           # 名前を言える物を見ていない回の続き
        self.hand_cool = 0         # 戻してからの、ひとりで見せない回
        self.n_said = 0            # 今持っている物の名前を言った回数
        self.next_name = 0         # 次に名前を言ってよい回（持ってからの回）
        self.last_name = None      # 最後に名前を言った回（持ってからの回）

    def tick(self, env) -> dict:
        """戻り値：その回に親がしたこと {"looking": 語, "said": 文, "brought": 語, "put": 語, "holding": 語}。"""
        oid = env.centre_object()
        if oid != self.looking:
            self.looking, self.look_ticks = oid, 0
        if oid is not None:
            self.look_ticks += 1
            self.no_look = 0
        else:
            self.no_look += 1
        did = {"looking": env.word_of(oid)}
        for k in list(self.cool):
            self.cool[k] -= 1
            if self.cool[k] <= 0:
                del self.cool[k]
        if self.hand_cool > 0:
            self.hand_cool -= 1
        if self.held is None:
            if (oid is not None and oid not in self.cool
                    and self.look_ticks >= int(round(BRING_AFTER_S / TICK_S))):
                self._bring(env, oid, "follow")
                did["brought"] = env.word_of(oid)
                did["mode"] = "follow"
            elif (self.bring_in_after is not None and self.hand_cool == 0
                    and self.no_look >= int(round(self.bring_in_after / TICK_S))):
                pick = self._choose(env)
                if pick is not None:
                    self._bring(env, pick, "bringin")
                    did["brought"] = env.word_of(pick)
                    did["mode"] = "bringin"
            return did
        word = env.word_of(self.held)
        did["holding"] = word
        did["mode"] = self.mode
        if self.phase in ("in", "out"):
            return self._carry(env, did, word)
        # 持っている間：揺らす・言う・戻しに運ぶ
        self.d += 1
        if self.first_look is None and oid == self.held:
            self.first_look = self.d
            env._log("look_held", self.held, f"{self.d * TICK_S:.1f}s", cause=f"parent_{self.mode}")
        at, gap = int(round(NAME_AT_S / TICK_S)), int(round(REPEAT_GAP_S / TICK_S))
        tail = int(round(NAME_TAIL_S / TICK_S))
        if self.first_look is None:
            end = int(round(BRING_IN_GIVE_UP_S / TICK_S))
        else:
            end = max(int(round(HOLD_S / TICK_S)), self.first_look + at + 1)
            if self.n_said == 0:
                self.next_name = max(self.next_name, self.first_look + at)
        if self.last_name is not None:
            end = max(end, self.last_name + tail)
        if self.d >= end:
            from .env import vec
            o = env.objects[self.held]
            home = env.home[self.held][0] + (env._centre(o) - vec(o["position"]))     # 置き場所での物の真ん中
            self._start_move(self.shake_pos(self.d - 1), home, "out", lift_end=True)
            env._log("carry_out", self.held, f"{self.move_n * TICK_S:.1f}s", cause="parent_put")
            return self._carry(env, did, word)
        env.hold(self.held, self.shake_pos(self.d), cause="parent_shake")
        if self.first_look is None:
            return did
        if self.say and self.n_said <= self.repeat and self.d >= self.next_name and oid == self.held:
            text = naming_utterance(word, self.rng)
            env.say(text, target=self.held, cause=f"parent_name_{self.mode}")
            self.stats["names"] += 1
            self.stats[f"names_{self.mode}"] += 1
            self.stats["repeats"] += self.n_said > 0
            self.n_said += 1
            self.last_name = self.d
            self.next_name = self.d + gap
            self.named[word] = self.named.get(word, 0) + 1
            did["said"] = text
        return did

    def _choose(self, env):
        """ひとりで見せる物を選ぶ（上の説明）。選べなければ None。"""
        toys = [i for i in getattr(env, "toys", []) if i in env.objects and i not in self.cool]
        targets = toys or [i for i in env.targets if i in env.objects and i not in self.cool]
        if not targets:
            return None
        eye = np.asarray(env.eye_pos, dtype=float)
        dist = {i: float(np.linalg.norm(env._centre(env.objects[i]) - eye)) for i in targets}
        near = [i for i in targets if dist[i] <= BRING_IN_NEAR_M] or targets
        n_recent = max(0, min(BRING_IN_RECENT, len(near) - 1))
        recent = set(self.recent[-n_recent:]) if n_recent else set()
        pool = [i for i in near if i not in recent] or near
        return min(pool, key=lambda i: (self.presented.get(i, 0), dist[i], i))

    def _bring(self, env, oid: str, mode: str = "follow") -> None:
        g = env.gaze_dir()
        base = np.asarray(env.eye_pos, dtype=float) + HOLD_DIST_M * g
        base[1] = max(float(base[1]), MIN_Y_M)
        from .env import cam_basis
        self.base = base
        self.axis = cam_basis(env.cam_yaw, env.cam_pitch)[1]
        self.held, self.d = oid, 0
        self.word = env.word_of(oid)
        self.say = self.rng.random() < self.p_name
        self.mode = mode
        self.first_look = 0 if mode == "follow" else None
        self.n_said, self.next_name, self.last_name = 0, 0, None
        if mode == "bringin":
            self.presented[oid] = self.presented.get(oid, 0) + 1
            self.recent.append(oid)
        self._start_move(env._centre(env.objects[oid]), base, "in")
        env._log("bring", oid, f"{self.word} {mode} {self.move_n * TICK_S:.1f}s", cause=f"parent_{mode}")
        self.stats["brings"] += 1
        self.stats[f"brings_{mode}"] += 1
        self._carry(env, {}, self.word)          # 持ってきた回から動き出す

    def _start_move(self, start, end, phase: str, lift_end: bool = False) -> None:
        self.path = carry_path(start, end, lift_end=lift_end)
        self.move_n = move_ticks(self.path)
        self.move_k = 0
        self.phase = phase

    def _follow_gaze(self, env) -> None:
        """運んでくる道の着く所を、太郎の今の視線の先へ少し寄せる（1回に MOVE_FOLLOW_M まで。道はつながったまま）。"""
        from .env import cam_basis
        want = np.asarray(env.eye_pos, dtype=float) + HOLD_DIST_M * env.gaze_dir()
        want[1] = max(float(want[1]), MIN_Y_M)
        a, c, b = self.path
        step = want - b
        n = float(np.linalg.norm(step))
        if n > MOVE_FOLLOW_M:
            step *= MOVE_FOLLOW_M / n
        if n > 1e-9:
            self.path = (a, c, b + step)
            self.base = b + step
            self.axis = cam_basis(env.cam_yaw, env.cam_pitch)[1]

    def carry_pos(self, k: int | None = None) -> np.ndarray:
        """運んでいる道の k 回目（0＝出発、move_n＝着いた所）の物の真ん中。"""
        k = self.move_k if k is None else k
        return bezier(self.path, min_jerk(k / self.move_n))

    def shake_pos(self, d: int) -> np.ndarray:
        """着いて d 回目の、揺らしている物の真ん中（左右に SHAKE_M、上下に SHAKE_UP_M、はじめ SHAKE_RAMP_S で大きくなる）。"""
        t = max(0, d) * TICK_S
        ramp = min_jerk(t / SHAKE_RAMP_S) if SHAKE_RAMP_S > 0 else 1.0
        w = 2.0 * math.pi * SHAKE_HZ * t
        return self.base + ramp * (SHAKE_M * math.sin(w) * self.axis + np.array([0.0, SHAKE_UP_M * math.sin(2.0 * w), 0.0]))

    def _carry(self, env, did: dict, word: str) -> dict:
        """運ぶ1回分（持ってくるのも戻すのも）。名前は言わない。着いたら持つ（in）か、置いて手を離す（out）。"""
        self.move_k += 1
        did["moving"] = self.phase
        if self.phase == "in":
            self._follow_gaze(env)
            if self.move_k >= self.move_n:
                env.hold(self.held, self.base, cause="parent_arrive")
                self.phase, self.d = "hold", 0
                env._log("arrive", self.held, word, cause=f"parent_{self.mode}")
                did["arrived"] = word
                did.pop("moving")
            else:
                env.hold(self.held, self.carry_pos(), cause="parent_carry")
            return did
        # out：戻しに運ぶ
        if self.move_k < self.move_n:
            env.hold(self.held, self.carry_pos(), cause="parent_carry")
            return did
        env.put_back(self.held, cause="parent_put")
        self.cool[self.held] = int(round(COOLDOWN_S / TICK_S))
        self.hand_cool = int(round(COOLDOWN_S / TICK_S))
        self.stats["puts"] += 1
        if self.mode == "bringin":
            self.stats["bringin_looked" if self.first_look is not None else "bringin_unlooked"] += 1
        did["put"] = word
        did.pop("moving")
        self.held = None
        self.word = ""
        self.mode = ""
        self.phase = ""
        self.path = None
        return did


class LookLog:
    """太郎が物を見続けた時間（目の真ん中に同じ物が写っていた回の続き。環境の正解）。0.1秒ごとに add(id, 語)。"""

    def __init__(self) -> None:
        self.runs: list[tuple[str, float]] = []    # (語, 秒)
        self.cur = None
        self.n = 0
        self.word = ""

    def add(self, oid, word: str) -> None:
        if oid != self.cur:
            self.flush()
            self.cur, self.word, self.n = oid, word, 0
        if oid is not None:
            self.n += 1

    def flush(self) -> None:
        if self.cur is not None and self.n > 0:
            self.runs.append((self.word, round(self.n * TICK_S, 1)))
        self.cur, self.n = None, 0

    def summary(self) -> dict:
        d = [s for _w, s in self.runs]
        by: dict[str, list] = {}
        for w, s in self.runs:
            by.setdefault(w, []).append(s)
        return {"looks": len(d), "mean_s": round(float(np.mean(d)), 2) if d else 0.0,
                "median_s": round(float(np.median(d)), 2) if d else 0.0, "max_s": max(d) if d else 0.0,
                "n_ge_1s": sum(s >= 1.0 for s in d), "total_s": round(sum(d), 1),
                "by_word": {w: {"looks": len(v), "total_s": round(sum(v), 1), "max_s": max(v)} for w, v in by.items()}}
