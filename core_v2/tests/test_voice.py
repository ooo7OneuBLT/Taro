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
    """声の係の出した口の指令を、ECHO_TICKS 回あとに音として戻す（声の出口→耳→聞き分けの代わり。聞き分けが自分の声と分けた形で渡す）。"""

    def __init__(self, voice, month):
        self.voice, self.month, self.t, self.back, self.heard = voice, month, 0, {}, []

    def tick(self, parent="", plan=None):
        own = self.back.pop(self.t, "")
        heard = (own + "|" if own else "") + parent      # 聞き分ける係の形：「聞いた自分の声|親の語」（段階9）
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


def test_word_choice_is_share_of_heard_and_own_babble():
    """語を言う確率＝その語を聞いた回数÷（聞いた語の合計＋自分の自由な喃語の回数）。言えない語の分は喃語になる。"""
    v = taught_voice("まん")
    v.lexicon.update({"まんま": 300, "ぼおる": 300})   # ぼおる はまだ言えない
    v.n_free = 400
    v.rng = random.Random(0)
    picks = [v.pick_word() for _ in range(20000)]
    share = picks.count("まんま") / len(picks)
    assert abs(share - 0.3) < 0.02 and "ぼおる" not in picks
    v.learning, v.babbling = True, True
    n0 = v.n_free
    for _ in range(200):
        v.babble()
    assert 0 < v.n_free - n0 < 200                      # 自由な喃語を出した回だけ数える


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
    out = ear.step({"earin": "ま", "voice": "ま"})
    assert out["voice"] == "ま|" and not ear.alone and ear.last_self == "ま"
    assert float(torch.linalg.vector_norm(out["semantic"])) == 0.0
    out = ear.step({"earin": "ま", "voice": ""})
    assert ear.alone["ま"] == 1


def test_listen_hears_parent_while_speaking():
    """段階9：自分が話している間も、親の声は聞く。写しに合う音だけが自分の声。"""
    ear = Listen(learning=True)
    for _ in range(2):
        ear.step({"earin": "ぼおる"})
        ear.step({"earin": "ぼおるだね"})
        ear.step({"earin": "ほらぼおる"})
    assert "ぼおる" in ear.known()
    n = ear.alone["ぼおるだね"]
    out = ear.step({"earin": "ば ぼおるだね", "voice": "ば"})
    assert ear.last_self == "ば" and ear.alone["ぼおるだね"] == n + 1 and "ば" not in ear.alone
    assert out["voice"] == "ば|ぼおる" and float(out["semantic"] @ word_tag("ぼおる")) > 0.99
    # 予想と違う音が出ても（体が育った・帳面に無い指令）、1音だけの発話を自分の声とみる
    out = ear.step({"earin": "ぼおる ぶ", "voice": "？"})
    assert ear.last_self == "ぶ" and out["voice"] == "ぶ|ぼおる" and "ぶ" not in ear.alone


def test_voice_takes_own_and_parent_from_same_tick():
    v = Voice(learning=True)
    p = (0, 0, 0, 0)
    v.fifo, v.own_start = [(p, -ECHO_TICKS)], -ECHO_TICKS
    v.step({"listen": "ば|ぼおる"})
    assert [e["voice"] for e in v.log if e["event"] == "own"] == ["ば"] and v.lexicon["ぼおる"] == 1
    assert v.known_sounds() == {"ば"}
    v2 = Voice(learning=True)
    v2.fifo = [(p, -ECHO_TICKS)]
    v2.step({"listen": "ぼおる"})                 # 親の語だけ（自分の声と取りちがえない）
    assert v2.own == "" and v2.lexicon["ぼおる"] == 1
    v2.note(p, "ま")
    v2.fifo = [(p, v2.t - 2)]
    assert v2.step({})["listen"] == "ま"         # 写しは帳面から予想した音


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
        res.append((b["voice"].get_extra_state(), dict(b["listen"].alone),
                    {k: v.clone() for k, v in b["semantic"].state_dict().items()},
                    {k: v for k, v in bed.stats.items() if k != "ticks"}))
    assert res[0][0] == res[1][0]
    assert res[0][1] == res[1][1]
    assert res[0][2].keys() == res[1][2].keys()
    assert all(torch.allclose(res[0][2][k].double(), res[1][2][k].double(), atol=1e-4) for k in res[0][2])
    assert res[0][3] == res[1][3]


def test_export_and_import_voice_into_full_brain(tmp_path):
    """試験場で育てた声（帳面・M・聞いた語・言う語の一覧）を、41係の脳へ読める。意味の辞書は持っていかない。"""
    from core_v2.brain import Brain
    bed = vb.Bed(seed=1, start_day=340)
    bed.run_play(600)
    p = tmp_path / "v.json"
    vb.export_voice(bed, str(p), {"seed": 1, "month": round(bed.month, 3)})
    brain = Brain()
    meta = vb.import_voice(brain.boxes, str(p))
    assert meta["seed"] == 1
    assert brain.boxes["voice"].notebook == bed.boxes["voice"].notebook
    assert brain.boxes["voice"].M == bed.boxes["voice"].M
    assert len(brain.boxes["wordplan"].words) == len(bed.boxes["wordplan"].words)
    sem = brain.boxes["semantic"]
    assert int(sem.n_learn) == 0 and sem.recall(word_tag("ぼおる")) is None
    assert not brain.boxes["voice"].learning and not brain.boxes["voice"].babbling
    brain.reset()                                       # 試行の切り替わりで、育てた中身は消えない
    assert brain.boxes["voice"].M == bed.boxes["voice"].M


def test_full_brain_voice_is_silent_by_default():
    """41係の脳：声の係は既定で黙る（段階2〜6の体ありの試行は声を出さない）。"""
    from core_v2.brain import Brain
    brain = Brain()
    for _ in range(300):
        out = brain.tick({"voiceout": 14.0, "earin": ""})
        assert out.get("voiceout", "") == ""


def test_say_names_scoring():
    """体ありの確かめ（say_names）の数え方：声を出し始めた回に見ていた物と、1秒以上続けて見た回。"""
    from core_v2.tools.say_names import score

    def row(t, ball, voice=""):
        return {"tick": t, "voice": voice, "ang_ball": ball, "ang_cup": 40.0, "ang_distractor": 40.0}
    rows = [row(t, 3.0) for t in range(12)] + [row(12, 3.0, "ぼ"), row(13, 3.0), row(14, 3.0), row(15, 3.0, "お"),
                                                 row(18, 3.0, "る")] + [row(t, 30.0) for t in range(19, 40)]
    s = score(rows)
    assert s["voices"] == [("ball", "ぼおる")] and s["looks"]["ball"] == 1
