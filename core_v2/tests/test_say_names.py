"""say_names の数え方（段階9の部屋で見ていた物の語を渡す形）。

予想：声を出し始めた回に見ていた物の語と声が組になる。一番近い名前は20語の中から選ぶ（「ぼおる」→ぼおる）。
"""

from core_v2.tools import say_names


def test_score_with_look_fn_pairs_voice_and_looked_word():
    rows = [{"tick": i, "voice": "", "look": "ぼおる"} for i in range(12)]
    rows[11]["voice"] = "ぼ"
    rows += [{"tick": 12, "voice": "お", "look": "ぼおる"}, {"tick": 13, "voice": "る", "look": "たおる"}]
    rows += [{"tick": 14 + i, "voice": "", "look": ""} for i in range(10)]
    s = say_names.score(rows, look_fn=lambda r: r["look"], objects=["ぼおる", "たおる", "さかな"])
    assert s["voices"] == [("ぼおる", "ぼおる")]
    assert s["looks"]["ぼおる"] == 1 and s["looks"]["さかな"] == 0


def test_closest_name_among_stage9_words():
    from core_v2.world_eye import instances as ins
    assert say_names.closest_name("ぼおる", ins.WORDS)[0] == "ぼおる"
