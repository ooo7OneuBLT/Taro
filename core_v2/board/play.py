"""6段目「親役で話す」：人（上野さん）がママの役をして、育った太郎とやり取りする（2026-10-20）。

盤面の世界の大人（world.Board）の代わりに、人の書いた親の行動で世界を進める。太郎の脳は core_run.grow_core と
同じ形で1分ずつ回し、そのまま覚え続ける（ミルク・名前・やめてくれた・喜んでくれた、のほうびは同じ）。
親の行動（1回の書き込みに何個でも）：
* ミルクをあげる：おなかが FEED_IF_BELOW より空いていれば飲ませる。満ちていれば「すすめ」になる（太郎には、いやなこと）
* 飲ませる：すすめている最中なら、満腹のまま飲ませてしまう。やめる：すすめをやめる（太郎の「いや」が通じた）
* 物の名前＋見せる・置く・あげる：目の前に物を置く。片付ける：物をどける
* 出ていく・戻る：ママが部屋を出る・戻る。パパが来る・パパが出る：パパ（何もしない人）が部屋にいるか
* 笑う・よろこぶ：太郎に喜んで応える
* 「…」：話しかける（かぎかっこの中。かぎかっこが無く、ほかの行動も無ければ全部を話した言葉とする）。
  耳の下駄の表にある一言はその単語、無い一言は知っている語を長い順に拾う。目の前の物の名前を言えば「一緒に見ている」
* 待つ N分：何もせずに N 分
そのあと、太郎が何かするまで（最長 wait 分）世界を進め、太郎のしたことと、その時の頭の中（欲求の強さ・思い浮かんだ
選択肢と見込み・どの欲求のために選んだか）を返す。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from core_v2.board.core_run import (ATTEND_MIN, BOARD_IDLE, EXTRA_OBJECTS, NEW_OBJECTS, OBJECTS, _who,
                                    speech_data, use_language)
from core_v2.board.world import EMPTY_AFTER_MIN, FEED_AMOUNT, FEED_IF_BELOW, CRY_COST
from core_v2.boxes.earin import normalize_kana
from core_v2.boxes.want import HEADS
from core_v2.boxes.worth import ACT_TAGS, TELL_TAGS
from core_v2.rhythm import TICKS_PER_MIN
from core_v2.tagging import word_tag

ALL_OBJECTS = OBJECTS + NEW_OBJECTS + EXTRA_OBJECTS
HEAD_JA = {"food": "おなか（ミルクがほしい）", "know": "知りたい（目の前の物の名前）", "stop": "いや（すすめをやめてほしい）",
           "social": "相手（戻ってきた人に向かいたい）"}
ACT_JA = {"cry": "泣く", "reach": "手を伸ばす", "tell_new": "前のことを話す（相手がいなかった間の物）",
          "tell_old": "前のことを話す（相手も見ていた物）"}


@dataclass
class Play:
    brain: object
    fullness: float = 0.5
    people: dict = field(default_factory=lambda: {"ママ": True, "パパ": False})
    object: str | None = None
    offered: bool = False
    attend_left: int = 0
    t: int = 0
    rng_state: object = None

    @classmethod
    def new(cls, brain) -> "Play":
        brain.skip = set(BOARD_IDLE)
        brain.reuse_pure = True
        use_language(brain)
        brain.boxes["semantic"].use_board()
        brain.reset()
        torch.manual_seed(60_000)
        return cls(brain=brain, rng_state=torch.get_rng_state())

    # ---- 親の言葉を単語にする（耳の下駄） ----
    def _vocab(self) -> list[str]:
        d = speech_data()
        words = {normalize_kana(w) for ws in d["ear"].values() for w in ws}
        words |= {normalize_kana(o) for o in ALL_OBJECTS}
        return sorted((w for w in words if w), key=len, reverse=True)

    def hear(self, text: str) -> list[str]:
        """一言を耳の下駄の表に入れ、太郎に届く単語を返す。"""
        ear = self.brain.boxes["listen"].__dict__["_board_ear"]
        key = normalize_kana(text)
        if key in ear:
            return ear[key]
        # 知っている語で区切る：拾えない字が一番少なく、その中で語の数が一番少ない区切り方（左から長い順に拾うと
        # 「これはなあに」が これ／はな になった）
        vocab = set(self._vocab())
        best = [(0, 0, [])] + [None] * len(key)
        for i in range(len(key)):
            if best[i] is None:
                continue
            miss, n, ws = best[i]
            cands = [(i + 1, miss + 1, n, ws)] + [(i + len(v), miss, n + 1, ws + [v]) for v in vocab
                                                  if key.startswith(v, i)]
            for j, m, k, w in cands:
                if best[j] is None or (m, k) < best[j][:2]:
                    best[j] = (m, k, w)
        words = best[len(key)][2]
        ear[key] = words
        return words

    # ---- 親の行動を読む ----
    def parse(self, text: str) -> dict:
        ev = {"fed": False, "overfed": False, "came": None, "spoke": None, "smiled": None, "heard": None,
              "attend": False, "wait": None, "notes": []}
        quoted = re.findall(r"[「『\"](.+?)[」』\"]", text)
        rest = re.sub(r"[「『\"](.+?)[」』\"]", " ", text)
        did = False
        if "パパ" in rest and re.search(r"来|くる|入", rest):
            self.people["パパ"] = True; did = True; ev["notes"].append("パパが部屋に来た")
        elif "パパ" in rest and re.search(r"出|いなく|行", rest):
            self.people["パパ"] = False; did = True; ev["notes"].append("パパが部屋を出た")
        elif re.search(r"出ていく|出て行く|部屋を出|でかけ|出かけ|いなくなる", rest):
            self.people["ママ"] = False; did = True; ev["notes"].append("ママが部屋を出た")
            self.offered = False                       # すすめていたら、やめて出ていく
        elif re.search(r"戻る|もどる|帰る|かえる|ただいま", rest):
            self.people["ママ"] = True; did = True; ev["notes"].append("ママが部屋に戻った")
        if re.search(r"飲ませ", rest):
            did = True
            if self.offered:
                self.fullness, self.offered, ev["overfed"] = 1.0, False, True
                ev["notes"].append("満腹なのに飲ませた")
            else:
                self._feed(ev)
        elif re.search(r"ミルク|まんま", rest) and re.search(r"あげ|やる|あたえ|飲む|のむ", rest):
            did = True
            self._feed(ev)
        if re.search(r"やめ|止め", rest) and self.offered:
            self.offered, ev["spoke"] = False, "ママ"; did = True
            ev["notes"].append("すすめをやめた")
        if re.search(r"片付け|かたづけ|しまう|どけ", rest):
            self.object = None; did = True; ev["notes"].append("物を片付けた")
        else:
            obj = next((o for o in sorted(ALL_OBJECTS, key=len, reverse=True)
                        if o in rest or normalize_kana(o) in normalize_kana(rest)), None)
            if obj and re.search(r"見せ|みせ|置|おく|あげ|わた|渡|出す", rest):
                self.object = obj; did = True; ev["notes"].append(f"{obj}を目の前に置いた")
        if re.search(r"笑|わら|にっこり|よろこ|喜|ほめ|褒", rest):
            ev["smiled"] = "ママ"; did = True; ev["notes"].append("喜んで応えた")
        m = re.search(r"待つ\s*(\d+)?|まつ\s*(\d+)?|(\d+)\s*分", rest)
        if m:
            ev["wait"] = int(next((g for g in m.groups() if g), 10)); did = True
        if not quoted and not did:
            quoted = [text]
        if quoted:
            said = "、".join(quoted)
            ev["heard"], ev["spoke"] = said, "ママ"
            words = self.hear(said)
            ev["notes"].append(f"話しかけた「{said}」→ 太郎の耳に届いた語：{' / '.join(words) or '（知っている語なし）'}")
            if self.object and normalize_kana(self.object) in words:
                ev["attend"] = True
        return ev

    def _feed(self, ev):
        if self.offered:
            ev["notes"].append("（すすめている最中）")
        elif self.fullness < FEED_IF_BELOW:
            self.fullness = min(1.0, self.fullness + FEED_AMOUNT)
            ev["fed"], ev["came"] = True, "ママ"
            ev["notes"].append("ミルクを飲ませた")
        else:
            self.offered = True
            ev["notes"].append("おなかがいっぱいなので「もう少し飲む？」とすすめた（太郎にはいやなこと）")

    # ---- 1分すすめる ----
    def _present(self) -> list[str]:
        return [n for n, here in self.people.items() if here]

    def minute(self, ev: dict | None) -> dict:
        ev = ev or {}
        present = self._present()
        came, spoke, smiled = ev.get("came"), ev.get("spoke"), ev.get("smiled")
        if ev.get("attend"):
            self.attend_left = ATTEND_MIN
        ext = {"gutin": {"fullness": self.fullness, "fed": bool(ev.get("fed")), "offered": self.offered,
                         "overfed": bool(ev.get("overfed"))},
               "eyein": {"people": [(n, n in (came, spoke), n == smiled) for n in present], "object": self.object,
                         "attend": self.attend_left > 0}}
        brain = self.brain
        for k in range(TICKS_PER_MIN):
            if k == 0 and ev.get("heard"):
                ext["earin"] = ev["heard"]
            elif "earin" in ext:
                del ext["earin"]
            to_env = brain.tick(ext)
        want = brain.boxes["want"]
        voice, body = to_env.get("voiceout"), to_env.get("bodyout")
        act = {"action": "nothing", "strength": 0.0, "to": None, "word": None, "say": None}
        if isinstance(voice, dict) and "cry" in voice:
            act.update(action="cry", strength=voice["cry"], to=_who(voice["to"], present))
        elif isinstance(voice, dict) and "say" in voice:
            act.update(action="say", strength=voice["strength"], to=_who(voice["to"], present), word=voice["word"],
                       say=voice["say"])
        elif isinstance(body, dict) and "reach" in body:
            act.update(action="reach", strength=body["reach"], to=_who(body["to"], present))
        if act["to"] is None:
            act["action"] = "nothing"
        if act["action"] == "say":
            act["about"] = self._about(act["word"])
        act["why"] = self._why(want.last, present)
        self.fullness -= 1.0 / EMPTY_AFTER_MIN
        if act["action"] == "cry":
            self.fullness -= CRY_COST * act["strength"]
        self.fullness = max(0.0, self.fullness)
        self.attend_left = max(0, self.attend_left - 1)
        self.t += 1
        act["t"] = self.t
        return act

    def turn(self, text: str, wait: int = 5) -> dict:
        """親の行動を1つ受け、太郎が何かするまで（最長 wait 分。「待つ N分」ならその分）すすめる。"""
        torch.set_rng_state(self.rng_state)
        ev = self.parse(text) if text.strip() else {"notes": [], "wait": None}
        steps = []
        n = ev.get("wait") or wait
        for i in range(n):
            a = self.minute(ev if i == 0 else None)
            steps.append(a)
            if a["action"] != "nothing" and not ev.get("wait"):
                break
        self.rng_state = torch.get_rng_state()
        return {"notes": ev["notes"], "steps": steps, "state": self.state()}

    def state(self) -> dict:
        sem = self.brain.boxes["semantic"]
        known = None
        if self.object:
            known = sem.knowing(word_tag("obj:" + self.object)) >= 1.0
        return {"t": self.t, "fullness": round(self.fullness, 2), "object": self.object, "knows_object": known,
                "present": self._present(), "offered": self.offered}

    # ---- 頭の中を言葉にする ----
    def _about(self, word: str) -> str | None:
        """言った語が思い出の帳面の名前なら、その思い出の物（見せるため。太郎はその物をその語で呼んでいる）。"""
        tag = word_tag(word)
        diary = [e for e in self.brain.boxes["episodic"].diary if e["word"] is not None]
        e = max(diary, key=lambda e: float(e["word"] @ tag), default=None)
        if e is None or float(e["word"] @ tag) < 0.7:    # 帳面の名前の語は意味の辞書から来るので少し揺れる（約0.9）
            return None
        return max(ALL_OBJECTS, key=lambda o: float(e["obj"] @ word_tag("obj:" + o)))

    def _label(self, tag) -> str:
        for k, t in {**ACT_TAGS, **TELL_TAGS}.items():
            if float(t @ tag) > 0.99:
                return ACT_JA[k]
        mouth = self.brain.boxes["voiceout"].__dict__.get("_mouth") or []
        w, say, _ = max(mouth, key=lambda x: float(x[2] @ tag))
        return f"「{say}」と言う"

    def _why(self, last: dict, present: list) -> dict:
        drives = {h: round(float(v), 2) for h, v in (last.get("drives") or {}).items()}
        head = last.get("head") if last.get("action") != "nothing" else None
        opts = []
        for tag, j, vs in last.get("options") or []:
            who = _who(j, present)
            if who is None:
                continue
            for r, h in enumerate(HEADS):
                if vs[r] > 0:
                    opts.append({"head": h, "what": self._label(tag), "to": who, "value": round(vs[r], 3)})
        return {"drives": drives, "head": head, "options": opts}


def save(play: Play, path: str) -> None:
    torch.save(play, path)


def load(path: str) -> Play:
    return torch.load(path, weights_only=False)
