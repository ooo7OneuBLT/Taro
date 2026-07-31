"""学習ループ本体。太郎を動かし、学習させ、プラグインに測らせる。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
1175〜1304行（学習ループ）＋1069〜1098行（checkpoint）＋1128〜1148行（睡眠リプレイ）
＋1150〜1183行（体を育てる）を**そのまま写した**もの。

元は930行の1つの関数の中にあり、測定（自己モデル・接触・手が視野に入る割合）も
同じ場所に書かれていたため、「測り方を変える」と「学習を変える」が区別できなかった。
→ 測るのは**プラグイン**に渡す（`run/plugins/`）。ここは学習だけを持つ。

【役割の分担】
    run/config.py       設定（実験ファイルから）
    run/taro_setup.py   太郎の中身（脳・学習器・神経調節・小脳）
    run/trainer.py      ここ。太郎と環境を噛み合わせて回す
    run/plugins/        外から測る道具（太郎を変えない）

注意：同じ設定で同じ結果が出るかを必ず確かめる（`run/tools/check_divergence.py`）。
  乱数を消費する順序が1つ違うだけで同じシードでも別の学習になる（落とし穴 項3）。
  注意：測定は体を進めるので、測定の前後で状態を控えて戻している（項79・項80）。
    その仕組みは `_snapshot` / `_restore`。ここを崩すと再現性が失われる。
"""
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from run.taro_setup import Taro, rescale_action, mse            # noqa: E402
from run.context import Ctx                                     # noqa: E402

torch.set_num_threads(1)
DT = 0.01                    # MuJoCo の1物理ステップの秒数


def _mj_arrays(d):
    """MuJoCo の `data` から**書き換えられる配列を全部**控える。

    【なぜ機械的に取るか、2026-07-30】測定は体を進めるので、測定の前後で状態を
    戻す必要がある（落とし穴 項79・項80）。ところが `qpos` と `qvel` だけ戻しても
    足りず、実測では「環境を進めた結果（obs_out）」だけが食い違い続けた。
    加速度・ソルバの前回解・センサの値も次の計算に影響するため。
    注意：手で並べると必ず取りこぼす。`mujoco.mj_copyData` は Python に露出していない
      （mujoco 3.3.0 で確認）ので、属性を走査して配列を集める。
    注意：読み取り専用の配列（サイズ情報など）は書き戻せないので飛ばす。
    """
    out = {}
    for nm in dir(d):
        if nm.startswith("_"):
            continue
        try:
            v = getattr(d, nm)
        except Exception:
            continue          # 一部の属性は参照するだけで例外を出す（触らない）
        if isinstance(v, np.ndarray) and v.size and v.flags.writeable:
            out[nm] = v.copy()
    return out


def _release_renderers(env):
    """眼球の描画用メモリ（OpenGL）を明示的に返す。体を作り直す前に呼ぶ。

    【なぜ要るか、2026-07-30】体を育てる実験（500回ごとに体を作り直す）で、
    GPU のメモリが尽きて学習が落ちた（OpenGL error 0x505 → exit 139）。
    `e_toy_env` は眼球カメラの `mujoco.Renderer` を辞書に持ち、
    **モデルが変わったときにしか閉じない**。作り直しのたびに古いものが残る。
    注意：`env.close()` はこの辞書を触らないので、ここで自分で閉じる。
    """
    u = getattr(env, "unwrapped", env)
    cache = getattr(u, "_eye_renderers", None)
    if not cache:
        return 0
    n = 0
    for r in list(cache.values()):
        try:
            r.close()
            n += 1
        except Exception as e:      # noqa: BLE001
            # 注意：握りつぶすが黙らない（閉じ損ねはメモリが残るだけ）
            print(f"注意[renderer] 閉じるのに失敗: {type(e).__name__}: {e}", flush=True)
    cache.clear()
    u._eye_renderers = {}
    u._eye_renderers_model = None
    # 使い回しのキャッシュも捨てる（次の体で作り直させる）
    for nm in ("_vision_cache", "_vision_t"):
        if hasattr(u, nm):
            setattr(u, nm, None)
    return n


class Trainer:
    """太郎に「生きて学ぶ」をさせる。

    注意：このクラスは**測らない**。測るのはプラグイン。
      ここが測り始めると、また「学習ループの中に測定が埋まる」状態に戻る。
    """

    def __init__(self, cfg, *, plugins=(), verbose=True, log_row=None):
        self.cfg = cfg
        self.plugins = list(plugins)
        self.verbose = verbose
        self._log_row = log_row or (lambda row: None)
        self.env = None
        self.taro = None
        # 測定の前に体を控えるための入れ物（重いので1つ作って使い回す）
        self._snap_data = None
        self._snap_model = None

    # ------------------------------------------------------------ 組み立て
    def build(self):
        """環境と太郎を作る。乱数の順序を元の実装に合わせる。"""
        cfg = self.cfg
        # 太郎の乱数は**4系統**ある。1つでも撒き忘れると同じシードで再現できない。
        #   【2026-07-30 に判明】4つ目（Python標準の random）を誰も撒いていなかった。
        #     `taro_core/src/body/internal_state.py` が「泣き始めるか」「寝入るか」
        #     「うとうとするか」を `random.random()` で決めている（同 168/121/146行）。
        #     泣く・眠るは**内受容感覚として脳の入力に入る**ので、1回でもタイミングが
        #     ずれると以後の学習が丸ごと分岐する。
        #   注意：元の経路（E/scripts/e_growth_train.py）にも同じ撒き忘れがある
        #     ＝過去の実験は同一シードでも再現していなかった（落とし穴 項79）。
        torch.manual_seed(cfg.seed)     # ①脳の初期値・探索のゆらぎ・睡眠リプレイの抜き取り
        np.random.seed(cfg.seed)        # ②numpy を使う処理（agency の並べ替えなど）
        random.seed(cfg.seed)           # ③内臓（泣く・寝る・うとうと）
        # ④環境の初期姿勢（gym の np_random）は env.reset(seed=) で撒く（taro_setup.py）

        from run.plugins.common import scene as scene_mod
        # 月齢は「学習ループ側の月齢」で上書きする（体を育てると学習中に変わるので、
        #   シーンの固定値では合わない）。＝シーンは月齢以外の環境を担う。
        self._cur_age = cfg.age_at(0)
        taro_spec = dict(cfg._taro)
        if self._cur_age is not None:
            taro_spec["age_months"] = float(self._cur_age)
        self._taro_spec = taro_spec
        self.env, self.scene, self.hands = scene_mod.build(
            cfg.scene, taro=taro_spec, seed=cfg.seed, verbose=self.verbose, hybrid=True)
        if self._cur_age is None:
            # シーンが決めた月齢を控える（体を育てない実験ではこのまま）
            self._cur_age = float(self.scene["body"]["age_months"])

        self.taro = Taro(cfg, self.env, seed=cfg.seed, verbose=self.verbose)
        self.state = self.taro.init_state(self.taro.first_obs)
        # 目標指向の探索が使う（過去に経験した感覚）
        # 注意：【逸脱/工学近似・2026-07-30 判明】これは Self-Prior ではない。
        #   先行研究（Kim, Kanazawa, Yoshida, Kuniyoshi 2025, arXiv:2504.11075）の
        #   self-prior は「経験した観測の**頻度分布**」を明示的に学習する
        #   （離散版＝カウント→Categorical／連続版＝正規化フロー NSF を最尤推定）。
        #   太郎は生ベクトルをFIFOに貯めて**一様ランダムに1つ選ぶだけ**で、
        #   頻度の重みが完全に消えている。⇒ doc/人間模倣からの逸脱リスト.md
        self.goal_buf = []
        self._t0 = time.time()
        self._build_probe_ctx()
        self._build_ctx()
        for p in self.plugins:
            p.setup(self.ctx)
        return self

    # -------------------------------------------------------- 環境の操作
    def step_k(self, a):
        """1回の判断で K 物理ステップ進める（＝同じ命令を K tick 保持する）。

        注意：【逸脱・2026-07-30】**感覚運動遅延（神経伝導の遅れ）が無い**。
          人間は運動指令が筋に届くまで、感覚が脳に届くまでに 0.1〜0.2秒かかる。
          太郎は命令がその tick で力になり、感覚もその tick で返る。
          ⇒ K=10（0.1秒）は「人間のフィードバック遅延に合わせた**間隔**」であって、
            遅れそのものは入っていない。
          MIMo grows!（López et al. 2025）は FIFO バッファとして実装している。
          → 人間模倣からの逸脱リスト「2026-07-30 感覚運動遅延が無い」
        """
        o, term = self.state["obs"], False
        for _ in range(self.cfg.K):
            o, r, te, tr, info = self.env.step(a)
            if te or tr:
                term = True
                break
        return o, term

    def reset_state(self):
        self.state["obs"], _ = self.env.reset()
        self.state["hidden"] = self.taro.brain.init_motor_hidden()
        self.state["prev_a"] = torch.zeros(self.taro.n_act)

    # -------------------------------------------------- 測定器へ渡す入れ物
    def _build_probe_ctx(self):
        """`E/scripts/e_probes.py` に渡す入れ物。判定の実体は e_probes 側にある。"""
        import e_probes
        t = self.taro
        self.probe_ctx = e_probes.ProbeContext(
            brain=t.brain, fusion=t.fusion, target_fusion=t.target_fusion,
            nat_head=t.nat_head, env=self.env, rescale_action=rescale_action,
            zc=t.infer_latent, act_mean=t.act_mean, step_k=self.step_k,
            ln_prop=t.encode_target, reset_state=self.reset_state,
            infer_goal_action=t.infer_goal_action,
            state=self.state, n_act=t.n_act, n_eval=self.cfg.n_eval,
            K=self.cfg.K, DT=DT, seed=self.cfg.seed,
            # 注意：リポジトリのルート基準にする。cwd 相対だと、どこから実行したかで
            #   逆モデルの診断ログ（inv_probe_*.txt 等）の出力先が変わる。
            log_dir=os.path.join(_ROOT, os.path.dirname(self.cfg.log or "")
                                 or os.path.join("E", "logs", "run")))

    def _build_ctx(self):
        """プラグインに渡す入れ物。プラグインは読むだけ。"""
        u = self.env.unwrapped
        dt = float(u.model.opt.timestep) * int(u.frame_skip) * self.cfg.K
        # 注意：name を入れ忘れると、ダッシュボードの見出しがフォルダ名になる
        #   （2026-07-31 に実際にそうなっていた）。プラグインは spec しか見られない。
        self.ctx = Ctx(env=self.env, spec={"name": self.cfg.name,
                                           "scene": self.cfg.scene, "taro": self.cfg._taro,
                                           "run": self.cfg._run},
                       scene=self.scene, n_steps=self.cfg.steps, dt=dt,
                       brain=self.taro.brain, log=self._log_row)
        # 自己モデルのプラグインが測るのに使う（e_probes への入れ物）
        self.ctx.probe_ctx = self.probe_ctx
        self.ctx.taro = self.taro

    # ---------------------------------------------------------- 体を育てる
    def _regrow(self, new_age):
        """保存を挟まずに env だけ作り直す（脳・経験・オプティマイザは残る）。

        【なぜ、2026-07-29】以前は「0ヶ月で学ぶ → 保存 → 別プロセスで4ヶ月として
        読み込み」で体を切り替えていた。ところが保存されるのは**脳の重みだけ**で、
        オプティマイザの状態・経験バッファ・馴化は消える（落とし穴 項75）。
        段階的成長を9回の保存で作ると、その条件だけ内部状態が8回消える＝交絡。
        """
        cfg = self.cfg
        old = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        # 視覚のレンダラ（OpenGLの描画用メモリ）を**明示的に閉じる**。
        #   【なぜ、2026-07-30】体を育てる実験で学習が落ちた：
        #     WARNING: OpenGL error 0x505 in or before mjr_makeContext
        #     ⇒ 0x505 は GPU のメモリ不足。体を作り直すたびに眼球用の
        #       `mujoco.Renderer` が新しく作られ、**古いものが解放されずに積み上がる**。
        #     条件Cは500回ごとに作り直すので36回ぶん溜まり、実測で
        #     3000回目（6回目の作り直し）で落ちた（exit=139＝メモリアクセス違反）。
        #   注意：`env.close()` では解放されない（レンダラは環境が持つ Python の辞書）。
        _release_renderers(self.env)
        self.env.close()
        from run.plugins.common import scene as scene_mod
        spec = dict(self._taro_spec)
        spec["age_months"] = float(new_age)
        self.env, self.scene, self.hands = scene_mod.build(
            cfg.scene, taro=spec, seed=cfg.seed, verbose=False, hybrid=True)
        new = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        # 注意：触覚ONではセンサ点が月齢で変わる（0ヶ月1734→4ヶ月4274）ので次元が合わなくなる。
        #   黙って壊れると「体を育てたのに学習が進まない」と読み違えるのでここで止める。
        if old != new:
            # 注意：ここで止めるとき、新しく作った env を閉じてから投げる
            #   （閉じないと描画コンテキストが残る）
            close_env(self.env)
            raise AssertionError(
                f"体を作り直したら観測/行動の次元が変わった {old} -> {new}。"
                "触覚ONではこの方式は使えない（落とし穴 項75）")
        self.probe_ctx.env = self.env       # 測定器も新しい体を見る
        self.ctx.env = self.env
        u = self.env.unwrapped
        self.ctx.model, self.ctx.data = u.model, u.data
        # プラグインに知らせる（geom の id を引き直させる）。
        #   注意：元の実装はここが無く、さらに `env.unwrapped` を作り直し前のまま
        #     参照していた＝体を育てる実験では接触を古い体で見ていた。
        for p in self.plugins:
            p.on_body_change(self.ctx)
        self.reset_state()                  # 新しい体で立て直す。脳と経験は保持されたまま

    # ------------------------------------------------------------ 睡眠
    def consolidate(self, n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチで再生し、予測経路を復習で固める。"""
        t = self.taro
        eps = t.hippo.replay()
        N = len(eps)
        if N < bs:
            return
        SV = torch.stack([e[0] for e in eps]); PA = torch.stack([e[1] for e in eps])
        AA = torch.stack([e[2] for e in eps]); CF = torch.stack([e[3] for e in eps])
        CLP = torch.stack([e[4] for e in eps]); NLP = torch.stack([e[5] for e in eps])
        H = torch.cat([e[6] for e in eps], dim=1)          # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = t.emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)
            out, _ = t.brain.motor_gru(emb, hb)
            z, kl, rc = t.brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + t.nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = t.block_pe(pred, NLP[idx]) + kl + rc       # 学習ループと同じ基準
            t.learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(t.learner.brain.parameters(),
                                           t.learner.grad_clip)
            t.learner.optimizer.step()

    # ------------------------------------------- 測定を「無かったこと」にする
    #
    # 【なぜ要るか、2026-07-30】自己モデルの測定（`e_probes.evaluate`）は
    #   **見るだけではなく実際に動かす**（n_eval=80 判断＝8秒ぶん）。
    #   そのため測定を1回するたびに：
    #     ・太郎の体が 8秒（agency も測るなら 14秒）ぶん進む
    #     ・学習が「測定後の体」から再開する
    #     ・チェックポイント間隔を変えると学習そのものが変わる（項80）
    #     ・同じシードでも結果がばらつく入り口になる（項79）
    #   実測：測定を外すと4回すべて完全一致。入れると step 1 で既に食い違う。
    #        食い違うのは体の状態（関節・前庭・触覚・視覚）で、**脳の重みは一致**。
    #
    # ⇒ 測定の前にすべてを控え、測定が終わったら**元に戻す**。
    #    学習から見て測定は「無かったこと」になる。
    #
    # 注意：これで過去の実験とは学習の軌道が変わる（＝数値が比較できなくなる）。
    #   それでも直すのは、過去の数値がそもそも再現しないため（項79）。
    # 注意：測定**値**のばらつきは残りうる（測定の中の非決定性は消えない）。
    #   ただしそれは「測るときの誤差」で、学習が分岐して増幅するのとは別物。

    def _snapshot(self):
        """体・内臓・乱数の状態をまるごと控える。"""
        import copy
        import random
        u = self.env.unwrapped
        d = u.data
        # --- 体（MuJoCo）を丸ごと控える ----------------------------------
        #   qpos/qvel だけでは足りない。加速度（qacc）・ソルバの前回解
        #   （qacc_warmstart）・センサの値も次の計算に影響する。
        #   注意：手で並べると必ず取りこぼす（実測で obs_out だけ食い違い続けた）。
        #   注意：`mujoco.mj_copyData` は Python に露出していない（mujoco 3.3.0 で確認）。
        #     なので**書き換えられる配列を機械的に全部**控える。
        snap = {
            "mjarr": _mj_arrays(d),
            "time": float(d.time),
            # --- 学習ループが持つ「いま」-----------------------------------
            "state": {"obs": self.state["obs"],
                      "hidden": self.state["hidden"].clone(),
                      "prev_a": self.state["prev_a"].clone()},
            # --- 乱数（3系統＋Python標準）測定が引いた分を巻き戻す ----------
            "torch_rng": torch.get_rng_state(),
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        }
        # --- 環境の乱数（gym が別に管理している。reset の姿勢の揺らぎに使う）---
        npr = getattr(u, "np_random", None)
        if npr is not None:
            snap["env_rng"] = npr.bit_generator.state
        # --- 内臓（HybridEnv）。空腹・眠気・不快・覚醒・泣いている残り時間 ----
        organs = {}
        for nm in ("internal_state", "stomach", "blood_vessel", "adenosine",
                   "homeostasis", "_step_in_second", "_feeding"):
            if hasattr(self.env, nm):
                organs[nm] = copy.deepcopy(getattr(self.env, nm))
        snap["organs"] = organs
        # --- 筋の状態（MuJoCo の外・Python 側にある）------------------------
        #   MuscleModel は活性化ダイナミクス（力がじわっと立ち上がる一次遅れ）を
        #   `self.activity` という**Python の配列**で持っている。d.act ではない。
        #   注意：ここを戻さないと「測定中に力んだ状態」から学習が再開する。
        #     2026-07-30 の実測では、これが obs_out（環境を進めた結果）が
        #     食い違う原因だった。元の学習ループにも同じ穴がある。
        snap["actu"] = self._numeric_attrs(getattr(u, "actuation_model", None))
        # --- 環境が持つ数値の状態（おもちゃのタイマー・視覚のキャッシュなど）---
        #   列挙漏れを防ぐため**数値と配列は機械的に全部**控える。
        #   （`_t_since_reset` `_toy_pending` `_glow_until` `_vision_cache` など、
        #     手で並べると必ず取りこぼす）
        snap["envnum"] = self._numeric_attrs(u)
        # --- 反射（前庭動眼反射・視線誘導反射）は内部状態を持つオブジェクト ----
        refl = {}
        for nm in ("_vor", "_orienting"):
            v = getattr(u, nm, None)
            if v is not None:
                refl[nm] = copy.deepcopy(v)
        snap["reflex"] = refl
        # --- 視覚のキャッシュ（辞書なので上の数値の走査では拾えない）-----------
        #   `e_toy_env` は「前回の描画から VISION_MIN_DT 経っていなければ使い回す」
        #   ため、描いた画像（辞書）と時刻を持っている。
        #   注意：戻さないと**測定中に描いた画像**が学習に混ざりうる。
        snap["vis"] = {nm: copy.deepcopy(getattr(u, nm, None))
                       for nm in ("_vision_cache", "_vision_t")
                       if hasattr(u, nm)}
        if getattr(u, "vision", None) is not None:
            snap["vis_out"] = copy.deepcopy(getattr(u.vision, "sensor_outputs", None))
        return snap

    @staticmethod
    def _numeric_attrs(obj):
        """obj が持つ「数値・配列」の属性を控える。オブジェクトは触らない。

        注意：`model` `data` は MuJoCo の本体なので除く（ここで触ると壊れる）。
        注意：配列は必ず copy する（`data` のビューだと戻す意味がなくなる）。
        """
        if obj is None:
            return {}
        out = {}
        for nm, v in vars(obj).items():
            if nm in ("model", "data", "_model", "_data"):
                continue
            if isinstance(v, np.ndarray):
                out[nm] = v.copy()
            elif isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                out[nm] = v
            elif isinstance(v, bool):
                out[nm] = v
        return out

    def _restore(self, snap):
        """控えた状態に戻す。測定が体を進めた分を巻き戻す。"""
        import mujoco
        import random
        u = self.env.unwrapped
        d = u.data
        # 体を戻す。順序が大事：
        #   ①全部の配列を戻す ②派生量を作り直す（mj_forward）
        #   ③ソルバの前回解（qacc_warmstart）と加速度を**もう一度**戻す
        #     — mj_forward がこれらを上書きするので、後から入れ直さないと
        #       物理が「測定の続き」から解かれて静かに軌道が変わる。
        arr = snap["mjarr"]
        for nm, v in arr.items():
            cur = getattr(d, nm, None)
            if isinstance(cur, np.ndarray) and cur.shape == v.shape:
                cur[:] = v
        d.time = snap["time"]
        mujoco.mj_forward(u.model, d)
        for nm in ("qacc_warmstart", "qacc", "act_dot", "sensordata"):
            v = arr.get(nm)
            cur = getattr(d, nm, None)
            if v is not None and isinstance(cur, np.ndarray) and cur.shape == v.shape:
                cur[:] = v
        self.state["obs"] = snap["state"]["obs"]
        self.state["hidden"] = snap["state"]["hidden"]
        self.state["prev_a"] = snap["state"]["prev_a"]
        torch.set_rng_state(snap["torch_rng"])
        np.random.set_state(snap["np_rng"])
        random.setstate(snap["py_rng"])
        if "env_rng" in snap and getattr(u, "np_random", None) is not None:
            u.np_random.bit_generator.state = snap["env_rng"]
        for nm, v in snap["organs"].items():
            setattr(self.env, nm, v)
        # 筋の状態（活性化ダイナミクス）を戻す。ここが抜けると力んだまま再開する
        am = getattr(u, "actuation_model", None)
        if am is not None:
            for nm, v in snap.get("actu", {}).items():
                cur = getattr(am, nm, None)
                if isinstance(cur, np.ndarray) and isinstance(v, np.ndarray) \
                        and cur.shape == v.shape:
                    cur[:] = v          # 中身を書き戻す（参照を差し替えない）
                else:
                    setattr(am, nm, v)
        for nm, v in snap.get("envnum", {}).items():
            cur = getattr(u, nm, None)
            if isinstance(cur, np.ndarray) and isinstance(v, np.ndarray) \
                    and cur.shape == v.shape:
                cur[:] = v
            else:
                setattr(u, nm, v)
        for nm, v in snap.get("reflex", {}).items():
            setattr(u, nm, v)
        for nm, v in snap.get("vis", {}).items():
            setattr(u, nm, v)          # 視覚のキャッシュと描画時刻
        if "vis_out" in snap and getattr(u, "vision", None) is not None:
            u.vision.sensor_outputs = snap["vis_out"]

    # ------------------------------------------------------------ 記録
    def _record(self, step):
        """全プラグインの値＋太郎の状態を**1行**にして残す。

        【なぜ1行にまとめるか、2026-07-30】以前は各プラグインが自分で `ctx.log(行)` を
        呼んでいたので、道具を2つ以上使うと**CSVの行が道具ごとに分裂**していた。
        1行に揃えると、グラフを描く側は**列を決め打ちしなくてよい**
        （`run/tools/dashboard.py` が列を自動で見つけて全部描く）。

        注意：ここに入れるのは「記録」だけ。学習の数値は変えない。
        """
        t = self.taro
        row = {"step": step,
               # 太郎が生きた時間（判断×K×DT）。実時間ではない
               "life_min": round(step * self.cfg.K * DT / 60.0, 3),
               "age_months": round(float(self._cur_age), 4),
               "real_min": round((time.time() - self._t0) / 60.0, 2),
               # 探索の強さ（ノルアドレナリン由来）。0.05 + ne*0.45
               "noise": round(0.05 + t.ne.get_ne_level() * 0.45, 5),
               "ne_maturation": round(float(t.ne.maturation), 5)}
        if self._act_accum:      # 力の出し具合（0付近＝フリーズの警告）
            row["act_abs"] = round(float(np.mean(self._act_accum[-200:])), 5)
        if self._caps_accum:     # 行動の変化量（下がりすぎ＝固まった）
            row["d_action2"] = round(float(np.mean(self._caps_accum[-200:])), 6)
        if self._eff_accum:      # 努力（代謝コスト）
            row["effort"] = round(float(np.mean(self._eff_accum[-200:])), 5)
        if self.cfg.cerebellum:  # 小脳の馴染み度（自動化がどれだけ効くか）
            row["cereb_err"] = round(float(t.cereb.err_ema.item()), 5)
        for p in self.plugins:
            m = p.metrics(self.ctx)
            if m:
                row.update(m)
        self._log_row(row)

    # ------------------------------------------------------- チェックポイント
    def _checkpoint(self, step):
        """測るのはプラグイン。ここは呼ぶだけ。

        注意：測定は体を進めるので、前後で状態を控えて戻す（上の注記を参照）。
          `finally` で戻すのは、測定が例外で落ちても学習を汚さないため。
        """
        self.ctx.step = step
        snap = self._snapshot()
        try:
            for p in self.plugins:
                p.on_checkpoint(self.ctx)
        finally:
            self._restore(snap)
        self._record(step)
        life_min = step * self.cfg.K * DT / 60.0
        real_min = (time.time() - self._t0) / 60.0
        t = self.taro
        noise = 0.05 + t.ne.get_ne_level() * 0.45
        tags = [p.line(self.ctx) for p in self.plugins]
        tags = [x for x in tags if x]
        cereb_tag = (f"cereb=on(err={t.cereb.err_ema.item():.2f})"
                     if self.cfg.cerebellum else "cereb=off")
        # 【taro-C5】|行動|＝力の出し具合（0付近＝フリーズ警告）
        extra = ""
        if self._act_accum:
            extra += f" |act|={np.mean(self._act_accum[-200:]):.3f}"
            if self.cfg.effort_cost and self._eff_accum:
                extra += f" effort={np.mean(self._eff_accum[-200:]):.3f}(lam={self.cfg.effort_cost})"
        # 注意：タグはASCIIのみ（Windowsのcp932で出せない文字を混ぜると print が例外を投げ
        #   学習が途中で落ちる。上付き2・絵文字で実際に2回踏んだ＝落とし穴 項35）
        if self._caps_accum:
            extra += f" da2={np.mean(self._caps_accum[-200:]):.4f}(lam={self.cfg.caps})"
        print(f"[seed{self.cfg.seed} rew={self.cfg.reward} age={self._cur_age:.2f}mo] "
              f"life={life_min:.0f}min | " + "  ".join(tags) +
              f" | noise={noise:.3f}(mat={t.ne.maturation:.2f}) {cereb_tag}{extra}"
              f" real={real_min:.0f}min", flush=True)

    # ------------------------------------------------------------ 本体
    def run(self):
        """学習ループ。e_growth_train.py 1175〜1304行の写し。"""
        cfg, t = self.cfg, self.taro
        env = self.env
        state = self.state
        n_train = cfg.steps
        ckpt = cfg.checkpoint
        self._act_accum, self._eff_accum, self._caps_accum = [], [], []
        from learning_progress import smoothness_cost
        pe_fast, pe_slow = t.lp.pe_fast, t.lp.pe_slow
        reach_goal, reach_prev_dist = None, 0.0

        if cfg.grows:
            print(f"[body-growth] 月齢 {float(cfg.age_months or 0.0)} → {float(cfg.age_to)}／"
                  f"開始{cfg.age_start}回目・{cfg.age_ramp or '一瞬'}回かけて・"
                  f"{cfg.age_every}回ごとに見直し", flush=True)

        self._checkpoint(0)
        for i in range(n_train):
            # ---- 体を育てる（age_to が無ければ何もしない）----------------------
            if cfg.grows and cfg.age_every > 0 and i > 0 and i % cfg.age_every == 0:
                na = cfg.age_at(i)
                if abs(na - self._cur_age) > 1e-9:
                    self._regrow(na)
                    print(f"[body-growth] {i}回目：月齢 {self._cur_age:.3f} → {na:.3f} ヶ月",
                          flush=True)
                    self._cur_age = na
                    env = self.env            # 作り直したので持ち替える
            # ---- 感覚を受け取り、内部表現を作る -------------------------------
            obs_in = state["obs"]       # 環境を進める**前**の観測（原因追跡用に控える）
            sv = t.fusion.encode(state["obs"])
            cf = t.target_fusion.encode(state["obs"]).detach()
            clp = t.encode_target(state["obs"])
            z, kl, rc, hn = t.infer_latent(sv, state["prev_a"], cf, state["hidden"].detach())
            # ---- 行動を作る（運動野＋小脳のブレンド）--------------------------
            mean, std, _w_c, e_c = t.motor_drive(z)
            # ---- 目標指向の探索（Goal Babbling）------------------------------
            # 切替：fixed=i%2固定／ne=ノルアドレナリン／pe=予測誤差+NE（既定）
            goal_step = False
            if cfg.goal_babbling and len(self.goal_buf) >= 64:
                if cfg.goal_switch == "fixed":
                    goal_step = (i % 2 == 0)
                elif cfg.goal_switch == "ne":
                    goal_step = torch.rand(1).item() < (1.0 - t.ne.get_ne_level())
                else:      # "pe"：驚きを主役＋NEを下駄
                    # 【逸脱/工学近似 注意】向き（驚き大→探索）はEFE/LC-NE/予測符号化に基づく
                    # 人間模倣だが、"足し算・等重み・この正規化"という式には根拠なし＝恣意的。
                    rel = min(pe_fast / (pe_slow + 1e-6), 2.0) / 2.0
                    explore_drive = min(t.ne.get_ne_level() + rel, 1.0)
                    goal_step = torch.rand(1).item() < (1.0 - explore_drive)
            if goal_step:
                if cfg.closed_loop_reach:
                    # 1つの目標を「届く/停滞」まで保持してにじり寄る（held goal）
                    if reach_goal is None:
                        reach_goal = self.goal_buf[
                            torch.randint(len(self.goal_buf), (1,)).item()].clone()
                        reach_prev_dist = mse(clp, reach_goal).item()
                    g = reach_goal
                else:
                    g = self.goal_buf[torch.randint(len(self.goal_buf), (1,)).item()]
                mean = t.infer_goal_action(z.detach(), clp, mean, g)
            else:
                reach_goal = None            # 探索に切替 → リーチ終了
            a, lp = t.brain.explore(mean, std)
            self.goal_buf.append(clp.detach())
            if len(self.goal_buf) > 2000:
                # 注意：【逸脱・2026-07-30】先行研究は経験のカウントを**一度も捨てない**
                #   （離散版は 30,000ステップ通して累積し、それで「馴化」が起きる）。
                #   太郎は古い方から捨てるので、昔の経験の頻度情報が消える。
                self.goal_buf.pop(0)
            pred = clp + t.nat_head(torch.cat([z, a.detach()], dim=-1))
            # 拮抗筋モード：a(n_joint) → to_env_action で筋活性化へ写像。OFFなら a_env==a
            a_env = t.brain.to_env_action(a)
            state["obs"], term = self.step_k(rescale_action(a_env, env.action_space))
            nlp = t.encode_target(state["obs"])
            if cfg.closed_loop_reach and reach_goal is not None:
                nd = mse(nlp, reach_goal).item()
                if nd >= reach_prev_dist:
                    reach_goal = None        # 近づかなくなった＝停止条件
                else:
                    reach_prev_dist = nd
            if cfg.replay:
                t.hippo.record(sv.detach(), state["prev_a"].detach(), a.detach(),
                               cf.detach(), clp.detach(), nlp.detach(),
                               state["hidden"].detach())
            # ---- 測る（プラグイン）------------------------------------------
            # 注意：判断ごとに1回（K tick ぶんに1回）呼ぶ。元の実装と同じ頻度。
            #   接触のような一瞬の事象は tick 単位で見た方が正確だが、
            #   まず元と同じ数値が出ることを確かめるため頻度も合わせる。
            self.ctx.step = i + 1
            # このステップの内部の値を「置いておく」だけ（プラグインは読むだけ）。
            #   同じシードで結果がばらつく原因を追うのに使う（落とし穴 項79）。
            #   注意：参照を入れるだけなので計算はしない＝学習の数値は変わらない。
            self.ctx.last = {"obs_in": obs_in, "obs_out": state["obs"], "sv": sv,
                             "cf": cf, "clp": clp, "z": z, "mean": mean, "std": std,
                             "a": a, "pred": pred, "nlp": nlp}
            for p in self.plugins:
                p.on_step(self.ctx)
            # ---- 学習 --------------------------------------------------------
            pe = t.block_pe(pred, nlp)
            progress = t.lp.update(pe.item())
            pe_fast, pe_slow = t.lp.pe_fast, t.lp.pe_slow
            # 内発的動機。progress＝学習進度（誤差が減っていれば正）
            rew_task = (progress if cfg.reward == "progress"
                        else t.brain.sensorimotor_reward(pe.item()))
            rew = rew_task
            if cfg.effort_cost:
                # 【taro-C5】努力コスト：活性化²の筋力重み付き平均を報酬から引く
                # （Selinger 2015 等の代謝最小化。注意二乗・λ・重みは近似＝感度確認対象）
                effort = float((a.detach() ** 2 * t.eff_w).sum())
                rew = rew_task - cfg.effort_cost * effort
                self._eff_accum.append(effort)
            # 【CAPS, Mysore et al. 2021】行動の急変にペナルティ。
            # λ=0 でも**必ず記録する**：OFF側の値が無いと「λが弱くて効かなかった」のか
            #   「元からこの値なのか」を切り分けられない（2026-07-25 に実際に困った）。
            smooth = smoothness_cost(a.detach(), state["prev_a"].detach())
            self._caps_accum.append(smooth)
            if cfg.caps:
                rew = rew - cfg.caps * smooth
            self._act_accum.append(float(a.detach().abs().mean().item()))
            # 方策の学習（ドーパミン）は努力コスト込みの報酬を見る＝「疲れは損」を学ぶ
            pl = t.learner.learn_action([lp], t.dop.compute_rpe(rew))
            hl = t.homeo.homeostatic_loss(sv); t.homeo.observe(sv)
            t.learner.update(pe + hl + kl + rc, pl)
            if cfg.cerebellum:
                # 小脳は方策とは別に、実際に行った運動を教師なしで真似て自動化する
                closs = t.cereb.imitation_loss(z.detach(), a.detach())
                t.cere_opt.zero_grad(); closs.backward(); t.cere_opt.step()
                t.cereb.observe(e_c)          # 馴染み度の基準を更新（自己正規化）
            t.dev_clock.tick()                # ③発達年齢を進める（覚醒中の学習1回）
            # 【NE decoupling】NE（探索）にはタスク報酬だけを見せる（努力コスト抜き）。
            # 努力コストで下がった報酬をNEに見せると「失敗した→探索せよ」と誤読し、
            # 大振幅ノイズが努力コストを打ち消す自滅ループになる（実測 noise 0.095→0.5）。
            t.ne.observe_reward(rew_task); t.ne.release_ne()
            if term:
                reach_goal = None
            if cfg.mature:
                # 成熟は sim秒でなく発達年齢（学習回数）で駆動する
                t.ne.mature(t.dev_clock.progress(n_train))
            state["hidden"] = hn.detach(); state["prev_a"] = a.detach()
            if term:
                self.reset_state()
            if cfg.replay and (i + 1) % ckpt == 0:
                self.consolidate()            # 睡眠：この間の経験を再生して定着
            if (i + 1) % ckpt == 0:
                self._checkpoint(i + 1)

        # ---- 後片付け --------------------------------------------------------
        out = {}
        for p in self.plugins:
            rep = p.report(self.ctx)
            if rep:
                out[p.name] = rep
        if cfg.save:
            self.taro.save(os.path.join(_ROOT, cfg.save) if not os.path.isabs(cfg.save)
                           else cfg.save,
                           extra={"age_final": self._cur_age})
        # 注意：env を閉じるのは呼び出し側（train / view）の finally に任せる。
        #   ここで閉じると、途中で例外が出たときに閉じられないまま残る。
        return out


def close_env(env):
    """環境を閉じる。閉じる処理そのものの失敗で本来の例外を隠さない。"""
    if env is None:
        return
    try:
        _release_renderers(env)     # 描画用メモリを先に返す
        env.close()
    except Exception as e:      # noqa: BLE001
        # 注意：握りつぶすが黙らない。閉じ損ねはメモリが残るだけで結果は汚さない。
        print(f"注意[close] 環境を閉じるときに失敗: {type(e).__name__}: {e}", flush=True)


def train(cfg, *, plugins=(), verbose=True, log_row=None):
    """設定から学習を1本回す。これが新しい経路の本体。

    注意：途中で例外が出ても環境を必ず閉じる（`finally`）。閉じ損ねると
      MuJoCo の描画コンテキストが残り、次の実行が不安定になる。
    """
    tr = Trainer(cfg, plugins=plugins, verbose=verbose, log_row=log_row)
    try:
        tr.build()
        return tr.run()
    finally:
        close_env(getattr(tr, "env", None))
