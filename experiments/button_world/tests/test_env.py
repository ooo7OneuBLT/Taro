"""ボタンのゲームのテスト。pytest でも、`python tests/test_env.py` でも回る。"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np

from button_world import ButtonWorld, Config, TOUCH_BAD, TOUCH_DECO, TOUCH_GOOD
from button_world import sprites as S
from button_world.env import features_for_similarity, similarity_table

ENV = ButtonWorld(Config(P=5))
STEP = jax.jit(ENV.step)
KEY = jax.random.PRNGKey(0)
NOOP, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

# 横からのテスト用の地図: 行5の段は列1〜7（はしご列4）、行3の段は列2〜6（はしご列3）
SIDE_MAP = dict(l5=1, r5=7, l3=2, r3=6, a=4, b=3)


def make_state(view, agent, btn_pos, deco_pos, door, good=0, door_open=False, side_map=SIDE_MAP):
    _, s = ENV.reset(KEY)
    if view == 0:
        walls, ladders = ENV.top_walls, jnp.zeros((9, 9), bool)
    else:
        walls, ladders, _ = ENV.side_map(*[jnp.array(side_map[k]) for k in ("l5", "r5", "l3", "r3", "a", "b")])
    game = s.game._replace(view=jnp.array(view, jnp.int32), good=jnp.array(good, jnp.int32))
    return s._replace(
        game=game,
        walls=walls,
        ladders=ladders,
        agent=jnp.array(agent, jnp.int32),
        start=jnp.array(agent, jnp.int32),
        btn_pos=jnp.array(btn_pos, jnp.int32),
        deco_pos=jnp.array(deco_pos, jnp.int32),
        door=jnp.array(door, jnp.int32),
        door_open=jnp.array(door_open),
    )


TOP = dict(view=0, btn_pos=[[4, 5], [6, 6], [2, 2]], deco_pos=[[4, 3], [6, 2], [2, 6]], door=[0, 4])
SIDE = dict(view=1, btn_pos=[[6, 2], [6, 6], [4, 6]], deco_pos=[[6, 1], [4, 2], [2, 5]], door=[6, 8])


def pos(s):
    return list(np.asarray(s.agent))


def test_good_button_opens_door():
    for base, agent, action in [(TOP, [4, 4], RIGHT), (SIDE, [6, 3], LEFT)]:
        s = make_state(agent=agent, good=0, **base)
        _, s2, r, done, info = STEP(KEY, s, action)
        assert bool(info["door_open"]), base["view"]
        assert int(info["first_touch"]) == TOUCH_GOOD
        assert bool(s2.pressed[0])


def test_bad_button_and_deco_do_not_open():
    s = make_state(agent=[6, 5], good=0, **TOP)
    _, s2, _, _, info = STEP(KEY, s, RIGHT)
    assert not bool(info["door_open"]) and int(info["first_touch"]) == TOUCH_BAD and bool(s2.pressed[1])
    s = make_state(agent=[4, 4], good=0, **TOP)
    _, _, _, _, info = STEP(KEY, s, LEFT)
    assert not bool(info["door_open"]) and int(info["first_touch"]) == TOUCH_DECO
    s = make_state(agent=[6, 7], good=0, **SIDE)
    _, _, _, _, info = STEP(KEY, s, LEFT)
    assert not bool(info["door_open"]) and int(info["first_touch"]) == TOUCH_BAD
    s = make_state(agent=[6, 0], good=0, **SIDE)
    _, _, _, _, info = STEP(KEY, s, RIGHT)
    assert not bool(info["door_open"]) and int(info["first_touch"]) == TOUCH_DECO


def test_closed_door_blocks_and_open_door_succeeds():
    s = make_state(agent=[1, 4], **TOP)
    _, s2, r, done, _ = STEP(KEY, s, UP)
    assert pos(s2) == [1, 4] and not bool(done) and float(r) == 0.0
    s = make_state(agent=[1, 4], door_open=True, **TOP)._replace(t=jnp.array(9, jnp.int32))
    _, _, r, done, info = STEP(KEY, s, UP)
    assert bool(done) and bool(info["success"]) and abs(float(r) - (1 - 0.5 * 10 / 60)) < 1e-6
    s = make_state(agent=[6, 7], **SIDE)
    _, s2, _, done, _ = STEP(KEY, s, RIGHT)
    assert pos(s2) == [6, 7] and not bool(done)
    _, _, r, done, info = STEP(KEY, s._replace(door_open=jnp.array(True)), RIGHT)
    assert bool(done) and bool(info["success"]) and float(r) > 0


def test_top_walls_block():
    s = make_state(agent=[1, 1], **TOP)
    _, s2, _, _, _ = STEP(KEY, s, LEFT)
    assert pos(s2) == [1, 1]


def test_ladder_up_and_down():
    s = make_state(agent=[6, 4], **SIDE)
    seen = []
    for a in (UP, UP, UP, NOOP, DOWN, DOWN, DOWN):
        _, s, _, _, _ = STEP(KEY, s, a)
        seen.append(pos(s))
    # 6→5→4 と登り、4 の上（行3は段の床）には行けず、はしごの上で落ちない。下りて 6 で止まる
    assert seen == [[5, 4], [4, 4], [4, 4], [4, 4], [5, 4], [6, 4], [6, 4]], seen


def test_up_down_only_on_ladder_in_side_view():
    s = make_state(agent=[6, 5], **SIDE)
    for a in (UP, DOWN):
        _, s2, _, _, _ = STEP(KEY, s, a)
        assert pos(s2) == [6, 5]
    # 上からの版では上・下はいつも効く
    s = make_state(agent=[4, 4], **TOP)
    _, s2, _, _, _ = STEP(KEY, s, UP)
    assert pos(s2) == [3, 4]


def test_fall_from_platform_edge():
    side = dict(SIDE, door=[6, 0], deco_pos=[[6, 1], [4, 2], [2, 5]])
    side["deco_pos"] = [[6, 3], [4, 2], [2, 5]]
    s = make_state(agent=[4, 7], **side)
    rows = []
    for a in (RIGHT, NOOP, NOOP):
        _, s, _, _, _ = STEP(KEY, s, a)
        rows.append(pos(s))
    assert rows == [[5, 8], [6, 8], [6, 8]], rows


def test_cannot_enter_door_from_above():
    for opened in (False, True):
        s = make_state(agent=[4, 7], door_open=opened, **SIDE)  # 扉は (6,8)
        for a in (RIGHT, NOOP, NOOP):
            _, s, _, _, info = STEP(KEY, s, a)
            assert not bool(info["success"])
        assert pos(s) == [5, 8]  # 扉の上に立つ
        _, s2, _, _, _ = STEP(KEY, s, DOWN)
        assert pos(s2) == [5, 8]


def test_enter_counts_and_open_to_enter_steps():
    s = make_state(agent=[4, 4], good=0, **TOP)
    seq = [RIGHT, NOOP, LEFT, RIGHT, LEFT, LEFT]  # 効くボタンに2回入る、飾りに1回
    for a in seq:
        _, s, _, _, info = STEP(KEY, s, a)
    assert int(info["n_good"]) == 2 and int(info["n_bad"]) == 0 and int(info["n_deco"]) == 1
    # 扉 (0,4) へ: いまは (4,3)。右へ1つ、上へ3つで (1,4)、もう1つ上で扉
    for a in (RIGHT, UP, UP, UP, UP):
        _, s, _, done, info = STEP(KEY, s, a)
    assert bool(info["success"])
    assert int(info["steps_open_to_enter"]) == 11 - 1, int(info["steps_open_to_enter"])  # 1歩目で開き、11歩目で入った


def _with_rule(env, view, agent, good=0, **kw):
    """別の Config の env で、決めた配置の状態を作る。"""
    _, s = env.reset(KEY)
    base = make_state(view=view, agent=agent, good=good, **kw)
    return base._replace(game=base.game._replace(btn_colors=s.game.btn_colors[:3], deco_colors=s.game.deco_colors[:3]))


def test_reset_to_start_rule():
    env = ButtonWorld(Config(P=3, wrong_button="reset_to_start", layout_pool=256))
    step = jax.jit(env.step)
    s = _with_rule(env, 0, [6, 4], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, s, _, _, _ = step(KEY, s, LEFT)  # (6,3)
    _, s, _, _, _ = step(KEY, s, RIGHT)  # (6,4)
    _, s2, r, done, info = step(KEY, s, RIGHT)
    _, s2, r, done, info = step(KEY, s2, RIGHT)  # (6,6) は効かないボタン → 始めの (6,4) へ
    assert pos(s2) == [6, 4] and int(s2.t) == 4 and not bool(done) and not bool(info["door_open"])
    assert not bool(s2.pressed[1]) and int(info["entered_button"]) == 1 and int(info["n_bad"]) == 1
    # 効くボタンなら戻されず扉が開く
    s = _with_rule(env, 0, [4, 4], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, s2, _, _, info = step(KEY, s, RIGHT)
    assert pos(s2) == [4, 5] and bool(info["door_open"])
    # 横からも
    s = _with_rule(env, 1, [6, 7], good=0, **{k: v for k, v in SIDE.items() if k != "view"})
    _, s2, _, _, info = step(KEY, s, LEFT)
    assert pos(s2) == [6, 7] and int(s2.t) == 1


def test_end_episode_rule():
    env = ButtonWorld(Config(P=3, wrong_button="end_episode", show_wrong_frame=False, layout_pool=256))
    step = jax.jit(env.step)
    s = _with_rule(env, 0, [6, 5], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, _, r, done, info = step(KEY, s, RIGHT)  # 効かないボタン
    assert bool(done) and bool(info["wrong_end"]) and not bool(info["success"]) and float(r) == 0.0
    s = _with_rule(env, 0, [4, 4], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, s2, _, done, info = step(KEY, s, RIGHT)  # 効くボタン
    assert not bool(done) and bool(info["door_open"]) and not bool(info["wrong_end"])
    # 扉が開いた後は、外れを踏んでも終わらない
    s3 = s2._replace(agent=jnp.array([6, 5], jnp.int32))
    _, _, _, done, info = step(KEY, s3, RIGHT)
    assert not bool(done) and not bool(info["wrong_end"])


def test_show_wrong_frame():
    """押したら終わり（初期値）: 外れを押した瞬間の画面を1枚見せ、次の歩でどの行動でも終わる。"""
    env = ButtonWorld(Config(P=5, wrong_button="end_episode", layout_pool=256))
    step = jax.jit(env.step)
    s = _with_rule(env, 0, [6, 5], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    obs, s2, r, done, info = step(KEY, s, RIGHT)  # (6,6) は外れ
    assert not bool(done) and bool(info["ending"]) and not bool(info["wrong_end"]) and float(r) == 0.0
    assert pos(s2) == [6, 6] and bool(s2.pressed[1]) and int(info["entered_button"]) == 1
    # 見せる1枚: 押したボタンのマスに自分（白い矢印）が描かれている
    cell = np.asarray(obs)[6 * 5:7 * 5, 6 * 5:7 * 5].astype(int)
    away = np.asarray(env.render(s2._replace(agent=jnp.array([4, 4], jnp.int32))))[6 * 5:7 * 5, 6 * 5:7 * 5].astype(int)
    assert np.abs(cell - away).mean() > 10, "押したボタンの上に自分が見えない"
    assert np.array_equal(np.asarray(obs), np.asarray(env.render(s2)))
    for a in (UP, NOOP):
        _, s3, r3, done3, info3 = step(KEY, s2, a)
        assert bool(done3) and bool(info3["wrong_end"]) and not bool(info3["success"]) and float(r3) == 0.0
        assert list(np.asarray(info3["agent"])) == [6, 6] and int(info3["t"]) == 2


def test_small_agent_on_items_shows_item_color():
    """物の上にいる時は自分を小さく描き、下のボタンの色が周りに見える（上から・横から）。"""
    L = S.layout()
    for view, base, agent in ((0, TOP, [4, 5]), (1, SIDE, [6, 2])):
        seen = {}
        for small in (False, True):
            env = ButtonWorld(Config(P=10, small_agent_on_items=small))
            s = make_state(agent=agent, good=0, **base)
            img = np.asarray(env.render(s)).astype(int)
            r, c = agent
            cell = img[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10]
            vt = view * S.N_THEMES + int(s.game.theme)
            shape = int(env.shape_of_vt[vt])
            tile = np.asarray(env.table)[L["btn0"] + ((vt * S.N_SHAPES + shape) * S.N_COLORS + int(s.game.btn_colors[0])) * 2 + 0].astype(int)
            seen[small] = (np.abs(cell - tile).sum(-1) == 0).mean()  # ボタンの絵がそのまま見えている画素の割合
        assert seen[True] > 0.7 and seen[True] > seen[False] + 0.1, (view, seen)
    # 物の上にいない時は、いつもの大きさ
    env = ButtonWorld(Config(P=10, small_agent_on_items=True))
    s = make_state(agent=[5, 5], **TOP)
    env0 = ButtonWorld(Config(P=10))
    assert np.array_equal(np.asarray(env.render(s)), np.asarray(env0.render(s)))


def test_wrong_penalty():
    env = ButtonWorld(Config(P=3, wrong_button="end_episode", wrong_penalty=0.3, layout_pool=256))
    step = jax.jit(env.step)
    s = _with_rule(env, 0, [6, 5], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, s2, r, done, info = step(KEY, s, RIGHT)  # 外れ
    assert abs(float(r) + 0.3) < 1e-6 and not bool(done)
    _, _, r2, done2, _ = step(KEY, s2, NOOP)  # 見せた後の歩: 罰は1回だけ
    assert float(r2) == 0.0 and bool(done2)
    s = _with_rule(env, 0, [4, 4], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    _, s3, r3, _, info3 = step(KEY, s, RIGHT)  # 当たり: 罰なし
    assert float(r3) == 0.0 and bool(info3["door_open"])
    # 扉が開いた後に外れを踏んでも罰なし
    _, _, r4, _, _ = step(KEY, s3._replace(agent=jnp.array([6, 5], jnp.int32)), RIGHT)
    assert float(r4) == 0.0
    assert Config.preset("gate_v1").wrong_penalty == 0.3 and Config.preset("gate_v1").small_agent_on_items


def test_six_buttons_layout():
    env = ButtonWorld(Config(P=3, n_buttons=6))
    for seed in range(200):
        _, s = env.reset(jax.random.PRNGKey(seed))
        cells = np.concatenate([np.asarray(s.agent)[None], np.asarray(s.btn_pos), np.asarray(s.deco_pos), np.asarray(s.door)[None]])
        assert len({tuple(c) for c in cells}) == len(cells) == 11
        bc = np.asarray(s.game.btn_colors).tolist()
        assert sorted(bc) == list(range(6))
        assert set(np.asarray(s.game.deco_colors).tolist()) <= set(bc) and len(set(np.asarray(s.game.deco_colors).tolist())) == 3
        if int(s.game.view) == 1:
            walls = np.asarray(s.walls)
            for r, c in cells:
                assert r in (2, 4, 6) and walls[r + 1, c]


def test_preset_train_v1():
    cfg = Config.preset("train_v1", P=3)
    assert cfg.wrong_button == "end_episode" and cfg.n_buttons == 3 and cfg.P == 3
    assert Config().wrong_button == "none"


def test_solvable_check_on_hand_made_layouts():
    """解けるかの判定を、手で作った配置で確かめる。"""
    env = ButtonWorld(Config(P=3, wrong_button="end_episode", layout_pool=256))
    # 横から: 効くボタン (6,2)、外れ (6,6)・(4,6)、扉 (6,8)、始め (6,3)。扉の前の (6,6)〜(6,7) を通るには外れ (6,6) を踏む → 解けない
    s = _with_rule(env, 1, [6, 3], good=0, **{k: v for k, v in SIDE.items() if k != "view"})
    assert not bool(env.solvable(s))
    # 扉を左端 (6,0) にして、間の飾りだけなら解ける
    side = {k: v for k, v in SIDE.items() if k != "view"}
    side["door"] = [6, 0]
    side["deco_pos"] = [[6, 1], [4, 2], [2, 5]]
    s = _with_rule(env, 1, [6, 3], good=0, **side)
    assert bool(env.solvable(s))
    # 上から: 開けた部屋なので解ける
    s = _with_rule(env, 0, [4, 4], good=0, **{k: v for k, v in TOP.items() if k != "view"})
    assert bool(env.solvable(s))


def test_strict_rules_give_solvable_layouts():
    for rule in ("end_episode", "reset_to_start"):
        for nb in (3, 6):
            for view in (0, 1):
                env = ButtonWorld(Config(P=3, wrong_button=rule, n_buttons=nb, views=(view,), layout_pool=512))
                reset = jax.jit(jax.vmap(env.reset))
                _, st = reset(jax.random.split(jax.random.PRNGKey(nb * 10 + view), 256))
                ok = np.asarray(jax.vmap(env.solvable)(st))
                assert ok.mean() > 0.99, (rule, nb, view, ok.mean())
                assert np.array_equal(ok, np.asarray(st.layout_ok))


def test_good_rank_is_uniform():
    """厳しいルールの配置では、効くボタンが始めの場所から近い順で何番目かが均等になる（近さの手がかりが無い）。"""
    for nb in (3, 6):
        for view in (0, 1):
            env = ButtonWorld(Config(P=3, wrong_button="end_episode", n_buttons=nb, views=(view,), layout_pool=nb * 200))
            _, st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(view + 7 * nb), 1200))
            ranks = np.asarray(jax.vmap(env.good_rank)(st, jax.random.split(jax.random.PRNGKey(3), 1200)))
            freq = np.bincount(ranks, minlength=nb) / len(ranks)
            assert np.all(np.abs(freq - 1 / nb) < 0.06), (nb, view, freq)
            assert np.asarray(jax.vmap(env.solvable)(st)).all()


def test_deco_colors_match_buttons():
    for seed in range(50):
        _, s = ENV.reset(jax.random.PRNGKey(seed))
        assert sorted(np.asarray(s.game.deco_colors).tolist()) == sorted(np.asarray(s.game.btn_colors).tolist())
    env = ButtonWorld(Config(P=3, deco_colors="random_with_good"))
    for seed in range(30):
        _, s = env.reset(jax.random.PRNGKey(seed))
        assert int(s.game.deco_colors[0]) == int(s.game.btn_colors[s.game.good])


def test_good_color_fixed_within_game_and_changes_across_games():
    env = ButtonWorld(Config(P=3, episode_len=8))
    _, s = env.reset(jax.random.PRNGKey(1))

    def body(s, key):
        ka, ks = jax.random.split(key)
        _, s2, _, done, info = env.step(ks, s, jax.random.randint(ka, (), 0, 5))
        return s2, (info["game_id"], info["good_color"], info["view"], info["theme"], info["episode_in_game"], done, info["new_game"])

    _, ys = jax.lax.scan(jax.jit(body), s, jax.random.split(jax.random.PRNGKey(2), 2000))
    gid, color, view, theme, ep, done, newg = map(np.asarray, ys)
    per_game = {}
    for g, c, v, th in zip(gid, color, view, theme):
        per_game.setdefault(g, set()).add((int(c), int(v), int(th)))
    assert all(len(x) == 1 for x in per_game.values()), "ゲームの中で効く色・視点・絵柄が変わった"
    assert len({next(iter(x))[0] for x in per_game.values()}) > 1
    assert ep.max() == 3 and np.all(newg <= done) and newg.sum() == done.sum() // 4
    assert set(theme.tolist()) <= {0, 1}, "試し用の絵柄2が練習に出た"


def test_resolution_keeps_shapes():
    """P を変えても、48通りの形がどの色・視点・絵柄・押す前後でも見分けられる。"""
    L = S.layout()
    for P in (3, 5, 7, 10):
        env = ButtonWorld(Config(P=P))
        obs, _ = env.reset(KEY)
        assert obs.shape == (9 * P, 9 * P, 3) and obs.dtype == jnp.uint8
        table = np.asarray(env.table).astype(np.float32)
        for vt, c, pr in itertools.product(range(S.N_VT), range(S.N_COLORS), (0, 1)):
            imgs = [table[L["btn0"] + ((vt * S.N_SHAPES + s) * S.N_COLORS + c) * 2 + pr] for s in range(S.N_SHAPES)]
            assert len({im.astype(np.uint8).tobytes() for im in imgs}) == S.N_SHAPES, (P, vt, c, pr)


def test_top_agent_differs_from_buttons_and_deco():
    """P=5 で、上からの「自分」の絵（床の上）と、すべてのボタン・飾りとの画素の差が、ボタンどうしの差の最小より大きい。"""
    env = ButtonWorld(Config(P=5))
    L = S.layout()
    table = np.asarray(env.table).astype(np.float32)
    a = np.asarray(env.agent_alpha)[0][..., None]
    for th in range(S.N_THEMES):
        vt = 0 * S.N_THEMES + th
        floor = table[L["bg0"] + vt * S.N_BG + 0]
        agent = np.round(floor * (1 - a) + np.asarray(env.agent_img)[0] * a)
        btns = [table[L["btn0"] + ((vt * S.N_SHAPES + s) * S.N_COLORS + c) * 2 + pr] for s in range(S.N_SHAPES) for c in range(S.N_COLORS) for pr in (0, 1)]
        decos = [table[L["deco0"] + vt * S.N_COLORS + c] for c in range(S.N_COLORS)]
        b = np.stack(btns).reshape(len(btns), -1)
        dmin_btn = min(np.abs(b[i] - b[j]).mean() for i in range(len(b)) for j in range(i + 1, len(b)))
        d_agent = min(np.abs(agent - x).mean() for x in btns + decos)
        assert d_agent > dmin_btn, (th, d_agent, dmin_btn)


def test_similarity_knob_by_pixels():
    tr, far = (3, 0, 1, 2), (0, 1, 0, 1)
    assert features_for_similarity(tr, far, 1.0) == tr
    assert features_for_similarity(tr, far, 0.0) == far
    rows, d0 = similarity_table(tr, far)
    mid = features_for_similarity(tr, far, 0.5)
    dmid = [r["diff"] for r in rows if r["features"] == mid][0]
    assert abs(dmid - d0 / 2) == min(abs(r["diff"] - d0 / 2) for r in rows)
    env = ButtonWorld(Config(P=3))
    assert env.features[(1, 0)] == tr and env.features[(0, 0)] == far


def test_start_layout_is_valid():
    """始めの配置: 物が重ならず、自分は物の上にいない。横からは全部が足場のある歩く行にある。"""
    for seed in range(300):
        _, s = ENV.reset(jax.random.PRNGKey(seed))
        cells = np.concatenate([np.asarray(s.agent)[None], np.asarray(s.btn_pos), np.asarray(s.deco_pos), np.asarray(s.door)[None]])
        assert len({tuple(c) for c in cells}) == len(cells)
        if int(s.game.view) == 1:
            walls, ladders = np.asarray(s.walls), np.asarray(s.ladders)
            for r, c in cells:
                assert r in (2, 4, 6) and walls[r + 1, c] and not ladders[r, c] and not walls[r, c]


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n, flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", n, repr(e), flush=True)
    print(f"{len(names) - failed}/{len(names)} passed")
    sys.exit(1 if failed else 0)
