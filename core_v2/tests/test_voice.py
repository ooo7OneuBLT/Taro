"""段階7（言える）：口の表・口の帳面・つながり M・言う条件・自分の声の写し・逆向きに引く・試験場の飛ばし方。

移し方の文書 付録 D3 の「数秒で終わる4つの確かめ」（口の表が月齢どおり／帳面が育つ／M は親の語のあとの似た声だけ数える／
M が強いときだけ言う）と、それを支える部品の確かめ。
"""

import random

import torch

from core_v2 import mouth
from core_v2.boxes.listen import Listen
from core_v2.boxes.semantic import Semantic
from core_v2.boxes.voice import ECHO_TICKS, M_MIN, M_WINDOW_TICKS, Voice
from core_v2.boxes.voiceout import Voiceout
from core_v2.boxes.base import ENV
from core_v2.tagging import word_tag
from core_v2.tools import voice_bed as vb


def say(text, month):
    return "".join(mouth.speak(mouth.CHAR_TO_PARAMS[c], month) for c in text)


# ---------------------------------------------------------------- 1. 口の表が月齢どおり
def test_mouth_table_by_month():
    assert say("まんま", 5.9) != "まんま"           # 6か月までは母音だけ
    assert say("まんま", 6.0) == "まんま"
    assert say("ぼおる", 9.0) != "ぼおる"           # 連動：唇は鼻音（も）、舌は破裂音（づ）
    assert say("ぼおる", 11.0) == "ぼおる"          # 11か月で調音点と調音法が別々に
    assert say("こ", 11.9) != "こ"                  # 軟口蓋・無声は12か月から
    assert say("こ", 12.0) == "こ"
    assert say("だっこ", 9.0) != "だっこ"
    assert say("こっぷ", 24.0) == "こたぷ"          # 「っ」は口の形では「た」（時間の現象なので表に無い）
    assert "ゃ" not in mouth.CHAR_TO_PARAMS and "ゃ" not in mouth.TABLE.values()   # 小さい字は口の表に無い


def test_voiceout_reads_motor_and_month():
    box = Voiceout()
    m = mouth.to_motor(mouth.CHAR_TO_PARAMS["ぼ"])
    assert box.step({"voice": m, ENV: 11.0})[ENV] == "ぼ"
    assert box.step({"voice": m, ENV: 7.0})[ENV] == "も"
    assert box.step({"voice": torch.zeros(mouth.N_MOTOR), ENV: 11.0})[ENV] == ""


# ---------------------------------------------------------------- 声の係だけを回す道具
class Ear:
    """声の係の出した口の指令を、ECHO_TICKS 回あとに音として戻す（声の出口→耳→聞き分けの代わり）。"""

    def __init__(self, voice, month):
        self.voice, self.month, self.t, self.back, self.heard = voice, month, 0, {}, []

    def tick(self, parent="", plan=None):
        heard = " ".join(x for x in (self.back.pop(self.t, ""), parent) if x)
        inbox = {"listen": heard}
        if plan is not None:
            inbox["wordplan"] = plan
        out = self.voice.step(inbox)
        p = mouth.from_motor(out["voiceout"])
        if p is not None:
            c = mouth.speak(p, self.month)
            self.back[self.t + ECHO_TICKS] = c
            self.heard.append(c)
        self.t += 1
        return out

    def run(self, n, **kw):
        for _ in range(n):
            self.tick(**kw)


def taught_voice(chars="まん", month=12.0):
    v = Voice(learning=True)
    for c in chars:
        v.note(mouth.CHAR_TO_PARAMS[c], c)
    return v


# ---------------------------------------------------------------- 2. 帳面が育つ
def test_notebook_grows_from_babble():
    v = Voice(learning=True, babbling=True, seed=3)
    ear = Ear(v, month=12.0)
    n = 0
    while sum(1 for e in v.log if e["event"] == "babble") < 100:
        ear.tick()
        n += 1
        assert n < 200000
    ear.run(20)
    assert v.known_sounds() == set(ear.heard)          # 出して耳に戻った音が、ちょうど帳面に載る
    assert len(v.known_sounds()) > 20
    for c in v.known_sounds():                          # 逆引きの口の指令で、その音が出る
        assert mouth.speak(v.plan(c)[0], 12.0) == c


def test_notebook_follows_the_growing_mouth():
    """同じ口の指令でも、口が育つと別の音が出る。古い音の逆引きは、その指令から外れる。"""
    v = Voice(learning=True)
    cmd = (1, 2, 0, 4)                                  # ぽ の指令。12か月までは声帯が有声なので「ぼ」
    for _ in range(50):
        v.note(cmd, mouth.speak(cmd, 11.5))
    v.note((1, 2, 1, 4), "ぼ")
    assert v.plan("ぼ") == [cmd]
    v.note(cmd, mouth.speak(cmd, 12.0))                 # 12か月：同じ指令で「ぽ」が出た
    assert v.plan("ぽ") == [cmd] and v.plan("ぼ") == [(1, 2, 1, 4)]
    v2 = Voice()
    v2.set_extra_state(v.get_extra_state())
    assert v2.plan("ぼ") == [(1, 2, 1, 4)] and v2.notebook == v.notebook


def test_newborn_babble_is_vowels_only():
    v = Voice(learning=True, babbling=True, seed=4)
    ear = Ear(v, month=3.0)
    ear.run(30000)
    assert v.known_sounds() and v.known_sounds() <= set("あいうえお")


# ---------------------------------------------------------------- 3. M は親の語のあとの似た声だけ数える
def _say(v, ear, text, parent_before=None, gap=10):
    if parent_before:
        ear.tick(parent=parent_before)
        ear.run(gap)
    v.queue = v.plan(text)
    ear.run(30)


def test_m_counts_only_similar_voice_after_parent_word():
    v = taught_voice("まんば")
    ear = Ear(v, 12.0)
    _say(v, ear, "まんま", parent_before="まんま")
    assert v.M["まんま"]["まんま"] == 1.0
    _say(v, ear, "ば", parent_before="まんま")                      # 似ていない声は数えない
    assert "ば" not in v.M["まんま"]
    _say(v, ear, "まま", parent_before="まんま")                     # 似ている声は似かたの分だけ
    assert abs(v.M["まんま"]["まま"] - mouth.similarity("まんま", "まま")) < 1e-9
    _say(v, ear, "まんま", parent_before="まんま", gap=M_WINDOW_TICKS + 5)   # 窓の外
    assert v.M["まんま"]["まんま"] == 1.0
    v2 = taught_voice("まん")
    ear2 = Ear(v2, 12.0)
    _say(v2, ear2, "まんま")                                       # 親の語が来る前の声は数えない
    ear2.tick(parent="まんま")
    ear2.run(10)
    assert not v2.M
    assert v2.lexicon["まんま"] == 1 and not v2.lexicon.get("ま")  # 自分の声は聞いた語に入らない


def test_m_is_not_counted_when_not_learning():
    v = taught_voice()
    v.learning = False
    ear = Ear(v, 12.0)
    _say(v, ear, "まんま", parent_before="まんま")
    assert not v.M and not v.lexicon


# ---------------------------------------------------------------- 4. M が強いときだけ言う
def test_speaks_only_when_m_is_strong():
    v = taught_voice()
    ear = Ear(v, 12.0)
    tag = word_tag("まんま")
    v.M["まんま"] = __import__("collections").Counter({"まんま": M_MIN - 0.5})
    ear.run(20, plan=tag)
    assert ear.heard == [] and v.last_decision["say"] is False and v.last_decision["word"] == "まんま"
    ear.run(5)                                          # 見るのをやめて、また見る
    v.M["まんま"]["まんま"] = M_MIN
    ear.run(20, plan=tag)
    assert "".join(ear.heard) == "まんま"
    ear.run(20, plan=tag)                               # 同じ語を持ち続けている間はくり返さない
    assert "".join(ear.heard) == "まんま"
    ear.run(5)
    ear.run(20, plan=word_tag("ぼおる"))                # M に無い語は黙る
    assert "".join(ear.heard) == "まんま"


def test_does_not_say_sounds_missing_from_notebook():
    v = taught_voice("ま")                              # 「ん」をまだ出したことがない
    v.M["まんま"] = __import__("collections").Counter({"まんま": 10.0})
    ear = Ear(v, 12.0)
    ear.run(20, plan=word_tag("まんま"))
    assert ear.heard == []


def test_parent_response_brings_next_babble_sooner():
    """親が応えると、次の喃語までの待ちが短く引き直される（やる気の側の生まれつきのほうび）。応えなければそのまま。"""
    waits = {False: [], True: []}
    for responded in (False, True):
        for s in range(40):
            v = taught_voice()
            v.babbling, v.rng = True, random.Random(s)
            ear = Ear(v, 12.0)
            v.countdown = 10**6
            _say(v, ear, "まんま")
            ear.tick(parent="まんま" if responded else "")
            waits[responded].append(v.countdown)
    assert sorted(waits[True])[20] < 100
    assert min(waits[False]) > 10**6 - 100


# ---------------------------------------------------------------- 自分の声の写し・逆向きに引く
def test_listen_does_not_learn_own_voice():
    ear = Listen(learning=True)
    out = ear.step({"earin": "ま", "voice": torch.tensor(1.0)})
    assert out["voice"] == "ま" and not ear.alone
    assert float(torch.linalg.vector_norm(out["semantic"])) == 0.0
    out = ear.step({"earin": "ま", "voice": torch.tensor(0.0)})
    assert ear.alone["ま"] == 1


def test_semantic_reverse_lookup():
    objs = vb.object_tags()
    box = Semantic(learning=True)
    q = word_tag("ぼおる")

    def name_once():
        for t in range(90):                     # 名残りが消えて覚えるまで（語のあと5秒）
            box.step({"listen": q if t == 30 else torch.zeros(64), "objfile": vb.roster(objs["ぼおる"])})

    for _ in range(2):
        name_once()
    assert box.recall_word(vb.roster(objs["ぼおる"])) is None      # 2回ではまだ出ない
    name_once()
    w = box.recall_word(vb.roster(objs["ぼおる"]))
    assert w is not None and float(w @ q) > 0.95
    out = box.step({"objfile": vb.roster(objs["ぼおる"])})
    assert float(out["wordplan"] @ q) > 0.9
    assert box.recall_word(vb.roster(objs["こっぷ"])) is None


# ---------------------------------------------------------------- 試験場
def test_bed_formats_and_echo_loop():
    """6つの係を書式の確かめ付きで回す。自分の声は帳面に、親の語だけが聞いた語に入る。"""
    bed = vb.Bed(seed=1, start_day=200)
    bed.brain.check_formats = True
    bed.run_play(600)
    v, ear = bed.boxes["voice"], bed.boxes["listen"]
    assert bed.stats["own"] > 3 and bed.stats["talk"] > 10
    assert v.known_sounds()
    assert not any(len(u) == 1 for u in ear.alone)               # 自分の1音の喃語を親の発話として覚えていない
    assert set(v.lexicon) <= ear.known()


def test_bed_skipping_idle_time_gives_same_result():
    res = []
    for skip in (True, False):
        bed = vb.Bed(seed=2, start_day=340)
        bed.skip_idle = skip
        bed.run_play(900)
        b = bed.boxes
        res.append((b["voice"].get_extra_state(), dict(b["listen"].alone), b["semantic"].W.clone(),
                    {k: v for k, v in bed.stats.items() if k != "ticks"}))
    assert res[0][0] == res[1][0]
    assert res[0][1] == res[1][1]
    assert torch.allclose(res[0][2], res[1][2], atol=1e-4)
    assert res[0][3] == res[1][3]
