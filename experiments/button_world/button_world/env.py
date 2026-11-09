"""ボタンのゲーム（上から見る版と横から見る版）。JAX で書き、jit・vmap・lax.scan で並列に回せる。

1ゲーム＝同じ視点・同じ絵柄・同じ効く色で、配置だけ変えた数回（episodes_per_game）。
部屋は 9×9 マス。観察は RGB 画像（9P×9P×3、uint8）だけ。

行動（共通の5つ）: 0 何もしない、1 上、2 下、3 左、4 右。
- 上から: 4方向に動く。外周は壁。扉は外周の壁の1か所にある。重力なし。
- 横から: 床が3段（下から2・4・6段目＝行7・5・3）。立って歩く行は 6・4・2。
  上の2段は幅がランダムで、端から落ちられる。段の間ははしごでつながり、はしごのマスでだけ上・下が効く。
  足元が空なら1歩ごとに1マス落ちる。扉は段の左右の端にあり、横からだけ入れる（上から落ちて入れない）。

物: ボタン（効くのは1色だけ）、色付きの「ボタンでない物」（上から＝宝石、横から＝花）、扉。
効く色のボタンに乗ると扉が開く。開いた扉に入ると成功で、点は 1 − 0.5×(歩数/1回の長さ)。
"""
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import sprites as S

H = W = 9
WALK_ROWS = (6, 4, 2)  # 横から: 立って歩く行（下の段から）
FLOOR_ROWS = (7, 5, 3)  # 横から: 床の行
DELTAS = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
FEATURE_NAMES = ("外形", "輪", "高さ", "中の印")

# 最初に踏んだ物の番号（info["first_touch"]）
TOUCH_NONE, TOUCH_GOOD, TOUCH_BAD, TOUCH_DECO = -1, 0, 1, 2
DECO_COLOR_MODES = ("match", "random", "random_with_good")
WRONG_BUTTON_RULES = ("none", "reset_to_start", "end_episode")


@dataclass(frozen=True)
class Config:
    P: int = 5  # 1マスの画素数（観察は 9P×9P×3）
    n_buttons: int = 3
    n_deco: int = 3
    episode_len: int = 60
    episodes_per_game: int = 4
    views: tuple = (0, 1)  # ゲームごとに選ぶ視点（0 上から、1 横から）
    themes: tuple = ((0, 1), (0, 1))  # 視点ごとに選んでよい絵柄。0・1 が練習用、2 が試し用
    # 飾りの色: "match"＝ボタンと同じ色を1つずつ（色だけでは半分しか当たらない、初期値）
    #           "random"＝6色から独立、"random_with_good"＝独立だが1つは必ず効く色
    deco_colors: str = "match"
    # 効かないボタンを踏んだ時: "none"＝何も起きない（初期値）、
    # "reset_to_start"＝その回の始めの場所に戻される（配置・扉はそのまま、時計は止まらない、押された見た目は付かない）、
    # "end_episode"＝扉が開く前にどれかのボタンを押した時点で選択は終わり。外れならその回は失敗で即終わり（点0）
    wrong_button: str = "none"
    # 厳しいルール（end_episode / reset_to_start）の時、外れを踏まずに 始め→効くボタン→扉 と行ける配置だけを出す。
    # 起動時に解ける配置を視点ごとに layout_pool 個ためておき、回の始めにそこから引く（毎歩の計算を増やさないため）
    solvable_layout: bool = True
    layout_pool: int = 8192
    # 形: 練習側の視点（train_view）は train_features。もう一方は、形の似方 similarity で決める。
    # similarity=0 で far_features（4特徴とも違う）、1 で train_features と同じ。
    # 間は、train_features との画素の差（P=5、押す前、灰色の背景）が (1−similarity)×差0% に一番近い組。
    train_view: int = 1
    train_features: tuple = (3, 0, 1, 2)  # 横から: 板・輪なし・盛り上がり・線
    far_features: tuple = (0, 1, 0, 1)  # 上から: 円・輪あり・平ら・点
    similarity: float = 0.0
    # 視点・絵柄ごとに特徴を直接決めたい時: {(view, theme): (外形, 輪, 高さ, 中の印)}
    shape_override: Optional[dict] = field(default=None, hash=False, compare=False)

    @staticmethod
    def preset(name, **overrides):
        """名前で設定の組を選ぶ。例: Config.preset("train_v1", P=5)"""
        return Config(**{**PRESETS[name], **overrides})

    def __post_init__(self):
        assert 1 + self.n_buttons + self.n_deco <= 7 * 7, "上からの部屋に物が入りきらない"
        assert 1 + self.n_buttons + self.n_deco <= 13, "横からの段に物が入りきらない"
        assert self.n_buttons <= S.N_COLORS
        assert 0.0 <= self.similarity <= 1.0
        assert self.deco_colors in DECO_COLOR_MODES
        if self.deco_colors == "match":
            assert self.n_deco <= self.n_buttons, "match は飾りがボタンより多くない時だけ"
        assert self.wrong_button in WRONG_BUTTON_RULES
        assert all(x != y for x, y in zip(self.train_features, self.far_features)), "far_features は4特徴とも train_features と違うこと"


def similarity_table(train_features, far_features, P=5):
    """48通りの形それぞれの、train_features との画素の差と、差0%（far）に対する似方。"""
    base = S.shape_index(*train_features)
    d0 = S.shape_pixel_diff(base, S.shape_index(*far_features), P)
    rows = []
    for s in range(S.N_SHAPES):
        d = S.shape_pixel_diff(base, s, P)
        rows.append(dict(shape=s, features=S.shape_features(s), diff=d, similarity=1 - d / d0))
    return rows, d0


def features_for_similarity(train_features, far_features, similarity, P=5):
    """形の似方つまみ: 画素の差が (1−similarity)×差0% に一番近い特徴の組。0 と 1 は端の組そのもの。"""
    if similarity >= 1.0:
        return tuple(train_features)
    if similarity <= 0.0:
        return tuple(far_features)
    rows, d0 = similarity_table(train_features, far_features, P)
    target = (1 - similarity) * d0
    best = min(rows, key=lambda r: (abs(r["diff"] - target), sum(a != b for a, b in zip(r["features"], train_features))))
    return tuple(best["features"])


def features_per_vt(cfg):
    """視点・絵柄（vt）ごとのボタンの特徴の組。"""
    test = features_for_similarity(cfg.train_features, cfg.far_features, cfg.similarity)
    feats = {}
    for view in range(S.N_VIEWS):
        for theme in range(S.N_THEMES):
            feats[(view, theme)] = tuple(cfg.train_features) if view == cfg.train_view else test
    if cfg.shape_override:
        feats.update(cfg.shape_override)
    return feats


class Game(NamedTuple):
    view: jnp.ndarray  # 0 上から、1 横から
    theme: jnp.ndarray
    btn_colors: jnp.ndarray  # (n_buttons,) 色の番号。重ならない
    good: jnp.ndarray  # 効くボタンの番号（btn_colors の中の位置）
    deco_colors: jnp.ndarray  # (n_deco,)


class State(NamedTuple):
    game: Game
    walls: jnp.ndarray  # (9,9) 通れないマス（扉は別）
    ladders: jnp.ndarray  # (9,9) はしご（横からだけ）
    agent: jnp.ndarray  # (2,) 行・列
    start: jnp.ndarray  # (2,) その回の始めの場所（reset_to_start で戻る先）
    btn_pos: jnp.ndarray  # (n_buttons, 2)
    deco_pos: jnp.ndarray  # (n_deco, 2)
    door: jnp.ndarray  # (2,)
    door_open: jnp.ndarray
    pressed: jnp.ndarray  # (n_buttons,)
    t: jnp.ndarray
    t_open: jnp.ndarray  # 扉が開いた歩（開いていなければ -1）
    n_good: jnp.ndarray  # この回で効くボタンのマスに入った回数
    n_bad: jnp.ndarray  # 効かないボタン
    n_deco: jnp.ndarray  # 飾り
    episode_in_game: jnp.ndarray
    game_id: jnp.ndarray
    first_touch: jnp.ndarray
    success: jnp.ndarray
    layout_tries: jnp.ndarray  # この回の配置を何個目の候補で選んだか（1 なら引き直しなし）
    layout_ok: jnp.ndarray  # 外れを踏まずに解ける配置か（厳しいルールの時だけ調べる。それ以外は True）


# 学習で使う設定の組。Config.preset("train_v1") で選ぶ（そのほかの値は上書きできる）
PRESETS = {
    "train_v1": dict(wrong_button="end_episode", n_buttons=3),
}


def _ring_doors():
    cells = []
    for i in range(1, W - 1):
        cells += [(0, i), (H - 1, i), (i, 0), (i, W - 1)]
    return np.array(cells, np.int32)


def _pick_cells(key, mask, k):
    """mask（9,9）が真のマスから、重ならずに k 個をランダムに選ぶ。(k,2)。"""
    scores = jnp.where(mask.reshape(-1), jax.random.uniform(key, (H * W,)), -1.0)
    _, idx = jax.lax.top_k(scores, k)
    return jnp.stack([idx // W, idx % W], axis=-1).astype(jnp.int32)


def _at(grid, cell):
    r = jnp.clip(cell[0], 0, H - 1)
    c = jnp.clip(cell[1], 0, W - 1)
    return grid[r, c]


class ButtonWorld:
    n_actions = 5

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        table, agents, alphas = S.build_tables(cfg.P)
        self.table = jnp.asarray(table)
        self.agent_img = jnp.asarray(agents, jnp.float32)
        self.agent_alpha = jnp.asarray(alphas)
        self.L = S.layout()
        feats = features_per_vt(cfg)
        self.features = feats
        self.shape_of_vt = jnp.array(
            [S.shape_index(*feats[(v, t)]) for v in range(S.N_VIEWS) for t in range(S.N_THEMES)], jnp.int32
        )
        self.ring_doors = jnp.asarray(_ring_doors())
        views = np.zeros(S.N_VIEWS, bool)
        views[list(cfg.views)] = True
        themes = np.zeros((S.N_VIEWS, S.N_THEMES), bool)
        for v in range(S.N_VIEWS):
            themes[v, list(cfg.themes[v])] = True
        assert views.any() and all(themes[v].any() for v in cfg.views)
        self.allowed_views = jnp.asarray(views)
        self.allowed_themes = jnp.asarray(themes)
        top_walls = np.zeros((H, W), bool)
        top_walls[0, :] = top_walls[-1, :] = top_walls[:, 0] = top_walls[:, -1] = True
        self.top_walls = jnp.asarray(top_walls)
        self.top_interior = jnp.asarray(~top_walls)
        if cfg.wrong_button in ("end_episode", "reset_to_start") and cfg.solvable_layout:
            self._build_pool()

    @property
    def obs_shape(self):
        return (H * self.cfg.P, W * self.cfg.P, 3)

    # ---------- ゲームと配置を選ぶ ----------
    def sample_game(self, key):
        cfg = self.cfg
        kv, kt, kc, kg, kd = jax.random.split(key, 5)
        view = jax.random.choice(kv, S.N_VIEWS, p=self.allowed_views / self.allowed_views.sum())
        tp = self.allowed_themes[view]
        theme = jax.random.choice(kt, S.N_THEMES, p=tp / tp.sum())
        btn_colors = jax.random.permutation(kc, S.N_COLORS)[: cfg.n_buttons].astype(jnp.int32)
        good = jax.random.randint(kg, (), 0, cfg.n_buttons)
        if cfg.deco_colors == "match":
            deco_colors = jax.random.permutation(kd, btn_colors)[: cfg.n_deco]
        else:
            deco_colors = jax.random.randint(kd, (cfg.n_deco,), 0, S.N_COLORS)
            if cfg.deco_colors == "random_with_good":
                deco_colors = deco_colors.at[0].set(btn_colors[good])
        return Game(view.astype(jnp.int32), theme.astype(jnp.int32), btn_colors, good, deco_colors.astype(jnp.int32))

    def side_map(self, l5, r5, l3, r3, a, b):
        """横からの地図。段の幅（行5は列 l5〜r5、行3は列 l3〜r3）と、はしごの列 a（行6↔4）・b（行4↔2）。"""
        cols = jnp.arange(W)[None, :]
        rows = jnp.arange(H)[:, None]
        in5 = (cols >= l5) & (cols <= r5)
        in3 = (cols >= l3) & (cols <= r3)
        walls = (rows >= 7) | ((rows == 5) & in5 & (cols != a)) | ((rows == 3) & in3 & (cols != b))
        ladders = (((rows == 6) | (rows == 5)) & (cols == a)) | (((rows == 4) | (rows == 3)) & (cols == b))
        # 物と自分を置けるマス: 足元が床の歩く行。はしごのマスと、はしごの上のマスは外す
        stand = ((rows == 6) | ((rows == 4) & in5) | ((rows == 2) & in3)) & ~ladders
        stand = stand & ~((rows == 4) & (cols == a)) & ~((rows == 2) & (cols == b))
        return walls, ladders, stand

    def sample_layout(self, key, view):
        cfg = self.cfg
        k = 1 + cfg.n_buttons + cfg.n_deco
        k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(key, 9)
        # 上から: 外周が壁。扉は外周の角以外。物と自分は内側 7×7 から
        top_door = self.ring_doors[jax.random.randint(k1, (), 0, len(self.ring_doors))]
        top_cells = _pick_cells(k2, self.top_interior, k)
        top = (self.top_walls, jnp.zeros((H, W), bool), top_door, top_cells)
        # 横から: 段の幅とはしごの列を選ぶ
        l5 = jax.random.randint(k3, (), 0, 3)
        r5 = jax.random.randint(k4, (), 6, 9)
        l3 = jax.random.randint(k5, (), 0, 3)
        r3 = jax.random.randint(k6, (), 6, 9)
        a = jax.random.randint(k7, (), l5 + 1, r5)
        b = jax.random.randint(k8, (), l3 + 1, r3)
        walls, ladders, stand = self.side_map(l5, r5, l3, r3, a, b)
        # 扉: 段の左右の端（置けるマスのうち）
        cand = jnp.stack(
            [
                jnp.array([6, 0]),
                jnp.array([6, W - 1]),
                jnp.stack([4, l5]),
                jnp.stack([4, r5]),
                jnp.stack([2, l3]),
                jnp.stack([2, r3]),
            ]
        ).astype(jnp.int32)
        ok = stand[cand[:, 0], cand[:, 1]]
        kd, kc = jax.random.split(k9)
        pick = jnp.argmax(jnp.where(ok, jax.random.uniform(kd, (6,)), -1.0))
        side_door = cand[pick]
        side_cells = _pick_cells(kc, stand.at[side_door[0], side_door[1]].set(False), k)
        side = (walls, ladders, side_door, side_cells)
        return jax.tree.map(lambda x, y: jnp.where(view == 0, x, y), top, side)

    def _layout_state(self, key, game, episode_in_game, game_id):
        cfg = self.cfg
        walls, ladders, door, cells = self.sample_layout(key, game.view)
        nb, nd = cfg.n_buttons, cfg.n_deco
        z = jnp.array(0, jnp.int32)
        return State(
            game=game,
            walls=walls,
            ladders=ladders,
            agent=cells[0],
            start=cells[0],
            btn_pos=cells[1 : 1 + nb],
            deco_pos=cells[1 + nb : 1 + nb + nd],
            door=door.astype(jnp.int32),
            door_open=jnp.array(False),
            pressed=jnp.zeros((nb,), bool),
            t=z,
            t_open=jnp.array(-1, jnp.int32),
            n_good=z,
            n_bad=z,
            n_deco=z,
            episode_in_game=jnp.asarray(episode_in_game, jnp.int32),
            game_id=jnp.asarray(game_id, jnp.int32),
            first_touch=jnp.array(TOUCH_NONE, jnp.int32),
            success=jnp.array(False),
            layout_tries=jnp.array(1, jnp.int32),
            layout_ok=jnp.array(True),
        )

    def transitions(self, state):
        """T[c, a] = マス c（番号 行×9＋列）で行動 a をした後のマスの番号。ゲームの本当の動き（はしご・落下も含む）。"""
        cells = jnp.stack([jnp.arange(H * W) // W, jnp.arange(H * W) % W], axis=-1).astype(jnp.int32)

        def one(cell):
            st = state._replace(agent=cell)
            return jax.vmap(lambda a: self._move(st, a))(jnp.arange(5))

        nxt = jax.vmap(one)(cells)
        return nxt[..., 0] * W + nxt[..., 1]

    def _reach(self, state, src, avoid):
        """src から行けるマス（avoid のマスには入らない）。"""
        T = self.transitions(state).reshape(-1)
        r = jnp.zeros((H * W,), bool).at[src].set(True)

        def body(_, r):
            add = jnp.zeros((H * W,), bool).at[T].max(jnp.repeat(r, 5))
            return r | (add & ~avoid)

        return jax.lax.fori_loop(0, 48, body, r)

    def solvable(self, state):
        """外れのボタンを踏まずに、始めの場所→効くボタン（扉は閉じたまま）→扉（開いた状態）と行けるか。"""
        cfg = self.cfg
        bidx = state.btn_pos[:, 0] * W + state.btn_pos[:, 1]
        good_mask = jnp.arange(cfg.n_buttons) == state.game.good
        wrong = jnp.zeros((H * W,), bool).at[bidx].max(~good_mask)
        good = bidx[state.game.good]
        start = state.start[0] * W + state.start[1]
        door = state.door[0] * W + state.door[1]
        r1 = self._reach(state._replace(door_open=jnp.array(False)), start, wrong)
        r2 = self._reach(state._replace(door_open=jnp.array(True)), good, wrong)
        return r1[good] & r2[door]

    def _build_pool(self):
        """解ける配置を視点ごとに layout_pool 個つくる。効くボタンは番号0に置いておく（引く時に並べ替える）。"""
        cfg = self.cfg
        pools, stats = [], {}
        for view in range(S.N_VIEWS):
            game = Game(jnp.array(view, jnp.int32), jnp.array(0, jnp.int32), jnp.arange(cfg.n_buttons, dtype=jnp.int32),
                        jnp.array(0, jnp.int32), jnp.zeros((cfg.n_deco,), jnp.int32))
            make = jax.jit(jax.vmap(lambda k: self._layout_state(k, game, 0, 0)))
            check = jax.jit(jax.vmap(self.solvable))
            kept, n_drawn, n_ok, seed = [], 0, 0, 0
            while n_ok < cfg.layout_pool:
                cands = make(jax.random.split(jax.random.PRNGKey(10_000 + 97 * view + seed), 4096))
                ok = np.asarray(check(cands))
                kept.append(jax.tree.map(lambda x: x[np.flatnonzero(ok)], cands))
                n_drawn += len(ok)
                n_ok += int(ok.sum())
                seed += 1
            pool = jax.tree.map(lambda *xs: jnp.concatenate(xs)[: cfg.layout_pool], *kept)
            pools.append(pool)
            stats[view] = dict(drawn=n_drawn, solvable=n_ok, solvable_rate=n_ok / n_drawn)
        self.pool = jax.tree.map(lambda a, b: jnp.stack([a, b]), *pools)
        self.pool_stats = stats

    def _new_episode(self, key, game, episode_in_game, game_id):
        cfg = self.cfg
        strict = cfg.wrong_button in ("end_episode", "reset_to_start")
        if not (strict and cfg.solvable_layout):
            return self._layout_state(key, game, episode_in_game, game_id)
        if not hasattr(self, "pool"):
            self._build_pool()
        i = jax.random.randint(key, (), 0, cfg.layout_pool)
        e = jax.tree.map(lambda x: x[game.view, i], self.pool)
        # ためた配置は効くボタンが番号0。このゲームの効くボタンの番号 good へ並べ替える
        slots = (jnp.arange(cfg.n_buttons) - game.good) % cfg.n_buttons
        return e._replace(
            game=game,
            btn_pos=e.btn_pos[slots],
            episode_in_game=jnp.asarray(episode_in_game, jnp.int32),
            game_id=jnp.asarray(game_id, jnp.int32),
        )

    def reset(self, key):
        kg, kl = jax.random.split(key)
        state = self._new_episode(kl, self.sample_game(kg), 0, 0)
        return self.render(state), state

    # ---------- 動き ----------
    @staticmethod
    def _out(cell):
        return (cell[0] < 0) | (cell[0] >= H) | (cell[1] < 0) | (cell[1] >= W)

    def _blocked(self, state, cell):
        is_door = jnp.all(cell == state.door)
        # 上からの版は外周の壁の地図に扉のマスも入っているので、扉のマスは壁として扱わない
        return self._out(cell) | (_at(state.walls, cell) & ~is_door) | (is_door & ~state.door_open)

    def _support(self, state, cell):
        """横から: 下にあると落ちない物（床・はしご・扉）。"""
        return self._out(cell) | _at(state.walls, cell) | _at(state.ladders, cell) | jnp.all(cell == state.door)

    def _move(self, state, action):
        side = state.game.view == 1
        cur = state.agent
        d = DELTAS[action]
        target = cur + d
        vertical = d[0] != 0
        # 横から: 上・下は、今いるマスか行き先のマスがはしごの時だけ効く。扉には上・下からは入れない
        ladder_ok = _at(state.ladders, cur) | _at(state.ladders, target)
        side_vertical_bad = vertical & (~ladder_ok | jnp.all(target == state.door))
        stay = self._blocked(state, target) | (side & side_vertical_bad)
        pos = jnp.where(stay, cur, target)
        below = pos + jnp.array([1, 0])
        falls = side & ~_at(state.ladders, pos) & ~self._support(state, below)
        return jnp.where(falls, below, pos)

    def step(self, key, state, action):
        cfg = self.cfg
        pos = self._move(state, action)
        entered = jnp.any(pos != state.agent)
        on_btn = jnp.all(state.btn_pos == pos, axis=-1) & entered
        on_deco = jnp.any(jnp.all(state.deco_pos == pos, axis=-1))
        good_mask = jnp.arange(cfg.n_buttons) == state.game.good
        hit_good = jnp.any(on_btn & good_mask)
        hit_bad = jnp.any(on_btn & ~good_mask)
        hit_deco = entered & on_deco
        entered_button = jnp.where(jnp.any(on_btn), jnp.argmax(on_btn), -1)
        door_open = state.door_open | hit_good
        success = state.door_open & jnp.all(pos == state.door)
        if cfg.wrong_button == "reset_to_start":
            # 外れ: 始めの場所に戻す。押された見た目は付けない
            pressed = state.pressed | (on_btn & good_mask)
            pos = jnp.where(hit_bad, state.start, pos)
            wrong_end = jnp.array(False)
        elif cfg.wrong_button == "end_episode":
            # 扉が開く前に外れを押したら、その回は失敗で終わり
            pressed = state.pressed | on_btn
            wrong_end = hit_bad & ~state.door_open
        else:
            pressed = state.pressed | on_btn
            wrong_end = jnp.array(False)
        touch = jnp.where(hit_good, TOUCH_GOOD, jnp.where(hit_bad, TOUCH_BAD, jnp.where(hit_deco, TOUCH_DECO, TOUCH_NONE)))
        first_touch = jnp.where(state.first_touch == TOUCH_NONE, touch, state.first_touch)
        t = state.t + 1
        t_open = jnp.where((state.t_open < 0) & door_open, t, state.t_open)
        done = success | wrong_end | (t >= cfg.episode_len)
        reward = jnp.where(success, 1.0 - 0.5 * t / cfg.episode_len, 0.0)
        end = state._replace(
            agent=pos,
            pressed=pressed,
            door_open=door_open,
            t=t,
            t_open=t_open,
            n_good=state.n_good + hit_good,
            n_bad=state.n_bad + hit_bad,
            n_deco=state.n_deco + hit_deco,
            first_touch=first_touch,
            success=success,
        )

        # 回が終わったら自動で次の回へ。episodes_per_game 回で新しいゲーム
        kg, kl = jax.random.split(key)
        next_ep = state.episode_in_game + 1
        new_game = done & (next_ep >= cfg.episodes_per_game)
        game = jax.tree.map(lambda a, b: jnp.where(new_game, a, b), self.sample_game(kg), state.game)
        restart = self._new_episode(kl, game, jnp.where(new_game, 0, next_ep), jnp.where(new_game, state.game_id + 1, state.game_id))
        nxt = jax.tree.map(lambda a, b: jnp.where(done, a, b), restart, end)

        info = {
            # 今の歩の結果（回が終わった時は、終わった回の値）
            "agent": end.agent,
            "btn_pos": end.btn_pos,
            "btn_colors": end.game.btn_colors,
            "good_color": end.game.btn_colors[end.game.good],
            "deco_pos": end.deco_pos,
            "deco_colors": end.game.deco_colors,
            "door": end.door,
            "door_open": end.door_open,
            "pressed": end.pressed,
            "view": end.game.view,
            "theme": end.game.theme,
            "first_touch": end.first_touch,
            "n_good": end.n_good,
            "n_bad": end.n_bad,
            "n_deco": end.n_deco,
            # 扉が開いてから（扉はいつも見えている）入るまでの歩数。成功しなければ -1
            "steps_open_to_enter": jnp.where(success, t - t_open, -1),
            "entered_button": entered_button,  # この歩で入ったボタンの番号（-1 は無し）
            "wrong_end": wrong_end,  # end_episode で外れを押して終わった
            "layout_tries": end.layout_tries,  # この回の配置を何個目の候補で選んだか
            "layout_ok": end.layout_ok,  # 外れを踏まずに解ける配置か
            "success": success,
            "episode_done": done,
            "episode_in_game": end.episode_in_game,
            "game_id": end.game_id,
            "t": t,
            # 返す状態が新しいゲームの始まりか（モデルの記憶はここで消す）
            "new_game": new_game,
        }
        return self.render(nxt), nxt, reward, done, info

    # ---------- 描画 ----------
    def tile_ids(self, state):
        L = self.L
        g = state.game
        vt = g.view * S.N_THEMES + g.theme
        rows = jnp.arange(H)[:, None]
        side_kind = jnp.where(state.ladders, 3, jnp.where(state.walls, jnp.where(rows == H - 1, 1, 2), 0))
        top_kind = jnp.where(state.walls, 1, 0)
        kind = jnp.where(g.view == 0, top_kind, side_kind)
        ids = L["bg0"] + vt * S.N_BG + kind
        ids = ids.at[state.door[0], state.door[1]].set(L["door0"] + vt * 2 + state.door_open.astype(jnp.int32))
        ids = ids.at[state.deco_pos[:, 0], state.deco_pos[:, 1]].set(L["deco0"] + vt * S.N_COLORS + g.deco_colors)
        shape = self.shape_of_vt[vt]
        btn = L["btn0"] + ((vt * S.N_SHAPES + shape) * S.N_COLORS + g.btn_colors) * 2 + state.pressed.astype(jnp.int32)
        ids = ids.at[state.btn_pos[:, 0], state.btn_pos[:, 1]].set(btn)
        return ids

    def render(self, state):
        P = self.cfg.P
        tiles = self.table[self.tile_ids(state)]  # (9,9,P,P,3)
        img = tiles.transpose(0, 2, 1, 3, 4).reshape(H * P, W * P, 3)
        r0, c0 = state.agent[0] * P, state.agent[1] * P
        patch = jax.lax.dynamic_slice(img, (r0, c0, 0), (P, P, 3))
        v = state.game.view
        a = self.agent_alpha[v][..., None]
        patch = jnp.round(patch * (1 - a) + self.agent_img[v] * a).astype(jnp.uint8)
        return jax.lax.dynamic_update_slice(img, patch, (r0, c0, 0))
