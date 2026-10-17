"""スミスとユー（Smith & Yu 2008）の再現（段階6）の予定表を作る。

物の出し入れ（world.pairs.schedule、run/world/toy_env.py の _hold_pairs）と、
親が言う語（world.word_test.schedule、run/world/parent.py の WordSchedule）の予定表を、
種から作って返す。記録用に、各画面の左右の物と、言った語の正解の物も返す。

元の実験（Cognition 106, 1558-1568、PMC2271000。本文の数字は段階6の仕様書と
doc/文献調査/2026-08-18_言語発達の月齢マイルストーン.md）：
* 学ぶ：30画面。1画面に物2つを4秒。出てから0.5秒後に語を2つ、0.5秒あけて1回ずつ
  （どちらの語がどちらの物かは分からない）。正しい語と物の組は各10回
* テスト：12試行。1試行8秒、正解の物と別の物（学んだ物）の2つ。語を4回。1語につき2回

ここでの決め方（論文に書いていない所。仕様書の決めに合わせる）：
* 画面の組：6つの物から作れる15組を、ちょうど2回ずつ使う（30画面。各物が10回ずつ出る）。
  並びは種で混ぜる。続く2画面で同じ物が出ないように並べる（出来なければ諦めて混ぜたまま）。
  左右は種で決める。2つの語の順番も種で決める（物の左右とは無関係）
* 語と物の対応も種で混ぜる（どの語がどの物の名前か）
* テストの語の時刻：出てから0.5秒後から2秒おきに4回（0.5, 2.5, 4.5, 6.5 秒）
* テストの別の物：12試行で、各物が正解2回・別の物2回になるように選ぶ。正解の左右は1語につき左右1回ずつ

時計：env.data.time（語の予定表 WordSchedule と同じ）。リセットの落ち着かせのあと **0.5秒から**
始まる（run_v2 の記録の t も 0.6 から）。t_start（既定1.0）はこの時計での最初の画面の時刻。

使い方（G/scripts を sys.path に入れてから）:
  from smith_yu_schedule import make, OBJECTS, WORDS
  sy = make(seed=0)
  env = run_v2.build_env("段階6_スミスとユー_6物体_2026-10-17", None, world_extra=world_extra)
  world_extra = {"pairs": {"schedule": sy["pairs_schedule"]},
                 "word_test": {"schedule": sy["word_schedule"]}}
  python G/scripts/smith_yu_schedule.py 0        （予定表を表で出す）
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

# かなの新しい語6つ（元の論文の bosa, gasser, manu, colat, kaki, regli を、かなにしたもの）
WORDS = ["ぼさ", "がっさ", "まぬ", "これっと", "かき", "れぐり"]

# 見分けやすい6つの物（形×色がすべて違う）。番号＝場面の body sy_obj0〜5
OBJECTS = [
    {"name": "赤い球", "shape": "sphere", "radius": 0.03, "rgba": [0.90, 0.15, 0.15, 1.0]},
    {"name": "青い円柱", "shape": "cylinder", "radius": 0.03, "rgba": [0.15, 0.35, 0.95, 1.0]},
    {"name": "緑の箱", "shape": "box", "radius": 0.025, "rgba": [0.15, 0.75, 0.25, 1.0]},
    {"name": "黄色い円柱", "shape": "cylinder", "radius": 0.03, "rgba": [0.95, 0.85, 0.15, 1.0]},
    {"name": "紫の箱", "shape": "box", "radius": 0.025, "rgba": [0.60, 0.25, 0.80, 1.0]},
    {"name": "だいだい色の球", "shape": "sphere", "radius": 0.03, "rgba": [0.98, 0.55, 0.10, 1.0]},
]

TRAIN_SLIDE_S = 4.0
TRAIN_WORD_T = (0.5, 1.0)          # 画面が出てからの時刻（語1、語2）
TEST_TRIAL_S = 8.0
TEST_WORD_T = (0.5, 2.5, 4.5, 6.5)


def _order_no_repeat(pairs, rng, max_steps=200000):
    """隣り合う画面で同じ物が出ない並びを、種で混ぜながら後戻りつきで探す。
    見つからなければ（手数の上限）混ぜただけの並びを返す。"""
    remaining = list(pairs)
    out = []
    steps = [0]

    def dfs():
        if not remaining:
            return True
        steps[0] += 1
        if steps[0] > max_steps:
            return False
        idx = list(range(len(remaining)))
        rng.shuffle(idx)
        tried = set()
        for i in idx:
            p = remaining[i]
            if p in tried or (out and set(p) & set(out[-1])):
                continue
            tried.add(p)
            out.append(remaining.pop(i))
            if dfs():
                return True
            remaining.insert(i, out.pop())
        return False

    if dfs():
        return out, True
    order = list(pairs)
    rng.shuffle(order)
    return order, False


def _test_distractors(n, rng, tries=2000):
    """12試行（各物が正解2回）の別の物：各物がちょうど2回、正解と違う物になるように選ぶ。"""
    targets = [k for k in range(n) for _ in range(2)]
    for _ in range(tries):
        d = list(targets)
        rng.shuffle(d)
        if all(a != b for a, b in zip(targets, d)):
            return targets, d
    raise RuntimeError("テストの別の物が選べなかった")


def make(seed: int = 0, *, t_start: float = 1.0, gap_s: float = 0.0, test_gap_s: float = 2.0,
         n_objects: int = 6, train_slide_s: float = TRAIN_SLIDE_S, train_word_t=TRAIN_WORD_T,
         test_trial_s: float = TEST_TRIAL_S, test_word_t=TEST_WORD_T, words=None):
    """予定表を作る。

    t_start    : 最初の画面を出す時刻（秒）。それまでは何も出さない
    gap_s      : 学ぶ画面と画面のあいだに、何も出さない時間（秒）。元の実験は画面が続いて替わる
                 （注意を引く画面19枚は挟む）。既定0＝続けて替える
    test_gap_s : 学ぶ画面の最後からテストの最初までの、何も出さない時間（秒）
    戻り値（辞書）:
      pairs_schedule : world.pairs.schedule にそのまま入れる [{"t0","t1","left","right"}]
      word_schedule  : world.word_test.schedule にそのまま入れる [{"t","word"}]
      word_of        : {物の番号: 語}（この種での語と物の対応）
      slides         : 記録用。[{"phase":"train"|"test","i","t0","t1","left","right",
                        "words":[{"t","word","object"}], ("target","distractor")}]
      events         : 記録用。言った語ごとに [{"t","word","object","phase","left","right"}]
                       （object＝その語の正解の物の番号）
      t_end          : 最後の画面が消える時刻
      no_repeat      : 学ぶ画面の並びで、隣り合う画面に同じ物が出ないようにできたか
    """
    rng = np.random.default_rng(int(seed))
    words = list(WORDS if words is None else words)[:n_objects]
    if len(words) < n_objects:
        raise ValueError("語が物の数より少ない")
    perm = rng.permutation(n_objects)
    word_of = {int(k): words[int(perm[k])] for k in range(n_objects)}
    pairs = list(itertools.combinations(range(n_objects), 2)) * 2
    order, no_repeat = _order_no_repeat(pairs, rng)

    slides, events, psched, wsched = [], [], [], []
    t = float(t_start)
    for i, (a, b) in enumerate(order):
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        said = [a, b] if rng.random() < 0.5 else [b, a]
        t0, t1 = t, t + train_slide_s
        ws = []
        for dt, k in zip(train_word_t, said):
            ws.append({"t": round(t0 + dt, 3), "word": word_of[k], "object": int(k)})
        slides.append({"phase": "train", "i": i, "t0": round(t0, 3), "t1": round(t1, 3),
                       "left": int(left), "right": int(right), "words": ws})
        t = t1 + gap_s
    t = slides[-1]["t1"] + test_gap_s
    targets, distractors = _test_distractors(n_objects, rng)
    side_of = {}
    trials = list(zip(targets, distractors))
    rng.shuffle(trials)
    for i, (tg, ds) in enumerate(trials):
        seen = side_of.setdefault(tg, [])
        if not seen:
            target_left = bool(rng.random() < 0.5)
        else:
            target_left = not seen[0]
        seen.append(target_left)
        left, right = (tg, ds) if target_left else (ds, tg)
        t0, t1 = t, t + test_trial_s
        ws = [{"t": round(t0 + dt, 3), "word": word_of[tg], "object": int(tg)} for dt in test_word_t]
        slides.append({"phase": "test", "i": i, "t0": round(t0, 3), "t1": round(t1, 3),
                       "left": int(left), "right": int(right), "words": ws,
                       "target": int(tg), "distractor": int(ds)})
        t = t1
    for s in slides:
        psched.append({"t0": s["t0"], "t1": s["t1"], "left": s["left"], "right": s["right"]})
        for w in s["words"]:
            wsched.append({"t": w["t"], "word": w["word"]})
            events.append({"t": w["t"], "word": w["word"], "object": w["object"], "phase": s["phase"],
                           "left": s["left"], "right": s["right"]})
    wsched.sort(key=lambda e: e["t"])
    return {"pairs_schedule": psched, "word_schedule": wsched, "word_of": word_of,
            "slides": slides, "events": events, "t_end": slides[-1]["t1"], "no_repeat": no_repeat,
            "objects": OBJECTS[:n_objects]}


def pairs_world(n_objects: int = 6) -> dict:
    """場面の world.pairs に入れる物の定義（予定表は空）。"""
    return {"objects": [{k: o[k] for k in ("shape", "radius", "rgba")} for o in OBJECTS[:n_objects]],
            "bodies": [f"sy_obj{i}" for i in range(n_objects)], "schedule": []}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sy = make(seed)
    names = [o["name"] for o in sy["objects"]]
    print("語と物:", {sy["word_of"][k]: names[k] for k in sorted(sy["word_of"])})
    print("隣り合う画面に同じ物なし:", sy["no_repeat"], " 終わり:", sy["t_end"], "秒")
    for s in sy["slides"]:
        ws = " ".join(f'{w["t"]:.1f}:{w["word"]}' for w in s["words"])
        extra = f' 正解={names[s["target"]]}' if s["phase"] == "test" else ""
        print(f'{s["phase"]:5s} {s["i"]:2d} {s["t0"]:6.1f}-{s["t1"]:6.1f} 左={names[s["left"]]:8s} '
              f'右={names[s["right"]]:8s} {ws}{extra}')
