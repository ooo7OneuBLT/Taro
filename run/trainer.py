"""★学習ループ本体。太郎を動かし、学習させ、プラグインに測らせる。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
1175〜1304行（学習ループ）＋1069〜1098行（checkpoint）＋1128〜1148行（睡眠リプレイ）
＋1150〜1183行（体を育てる）を**そのまま写した**もの。

元は930行の1つの関数の中にあり、測定（自己モデル・接触・手が視野に入る割合）も
同じ場所に書かれていたため、「測り方を変える」と「学習を変える」が区別できなかった。
→ 測るのは**プラグイン**に渡す（`run/plugins/`）。ここは学習だけを持つ。

【役割の分担】
    run/config.py       設定（実験ファイルから）
    run/taro_setup.py   ★太郎の中身（脳・学習器・神経調節・小脳）
    run/trainer.py      ★ここ。太郎と環境を噛み合わせて回す
    run/plugins/        外から測る道具（太郎を変えない）

⚠️★元の実装と数値が一致するかを必ず確かめる（`run/tools/compare_legacy.py`）。
  乱数を消費する順序が1つ違うだけで同じシードでも別の学習になる（落とし穴 項3）。
"""
import os
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


class Trainer:
    """太郎に「生きて学ぶ」をさせる。

    ⚠️★このクラスは**測らない**。測るのはプラグイン。
      ここが測り始めると、また「学習ループの中に測定が埋まる」状態に戻る。
    """

    def __init__(self, cfg, *, plugins=(), verbose=True, log_row=None):
        self.cfg = cfg
        self.plugins = list(plugins)
        self.verbose = verbose
        self._log_row = log_row or (lambda row: None)
        self.env = None
        self.taro = None

    # ------------------------------------------------------------ 組み立て
    def build(self):
        """環境と太郎を作る。★乱数の順序を元の実装に合わせる。"""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        from run.plugins.common import scene as scene_mod
        # ★月齢は「学習ループ側の月齢」で上書きする（体を育てると学習中に変わるので、
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
        self.goal_buf = []          # 目標指向の探索が使う（過去に経験した感覚）
        self._t0 = time.time()
        self._build_probe_ctx()
        self._build_ctx()
        for p in self.plugins:
            p.setup(self.ctx)
        return self

    # -------------------------------------------------------- 環境の操作
    def step_k(self, a):
        """1回の判断で K 物理ステップ進める（＝同じ命令を K tick 保持する）。"""
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
        """`E/scripts/e_probes.py` に渡す入れ物。★判定の実体は e_probes 側にある。"""
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
            log_dir=os.path.dirname(self.cfg.log or "") or ".")

    def _build_ctx(self):
        """プラグインに渡す入れ物。★プラグインは読むだけ。"""
        u = self.env.unwrapped
        dt = float(u.model.opt.timestep) * int(u.frame_skip) * self.cfg.K
        self.ctx = Ctx(env=self.env, spec={"scene": self.cfg.scene, "taro": self.cfg._taro,
                                           "run": self.cfg._run},
                       scene=self.scene, n_steps=self.cfg.steps, dt=dt,
                       brain=self.taro.brain, log=self._log_row)
        # ★自己モデルのプラグインが測るのに使う（e_probes への入れ物）
        self.ctx.probe_ctx = self.probe_ctx
        self.ctx.taro = self.taro

    # ---------------------------------------------------------- 体を育てる
    def _regrow(self, new_age):
        """★保存を挟まずに env だけ作り直す（脳・経験・オプティマイザは残る）。

        【なぜ、2026-07-29】以前は「0ヶ月で学ぶ → 保存 → 別プロセスで4ヶ月として
        読み込み」で体を切り替えていた。ところが保存されるのは**脳の重みだけ**で、
        オプティマイザの状態・経験バッファ・馴化は消える（落とし穴 項75）。
        段階的成長を9回の保存で作ると、その条件だけ内部状態が8回消える＝交絡。
        """
        cfg = self.cfg
        old = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        self.env.close()
        from run.plugins.common import scene as scene_mod
        spec = dict(self._taro_spec)
        spec["age_months"] = float(new_age)
        self.env, self.scene, self.hands = scene_mod.build(
            cfg.scene, taro=spec, seed=cfg.seed, verbose=False, hybrid=True)
        new = (int(self.env.observation_space["observation"].shape[0]),
               int(self.env.action_space.shape[0]))
        # ⚠️触覚ONではセンサ点が月齢で変わる（0ヶ月1734→4ヶ月4274）ので次元が合わなくなる。
        #   黙って壊れると「体を育てたのに学習が進まない」と読み違えるのでここで止める。
        assert old == new, (f"体を作り直したら観測/行動の次元が変わった {old} -> {new}。"
                            "触覚ONではこの方式は使えない（落とし穴 項75）")
        self.probe_ctx.env = self.env       # 測定器も新しい体を見る
        self.ctx.env = self.env
        u = self.env.unwrapped
        self.ctx.model, self.ctx.data = u.model, u.data
        # ★プラグインに知らせる（geom の id を引き直させる）。
        #   ⚠️元の実装はここが無く、さらに `env.unwrapped` を作り直し前のまま
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
            loss = t.block_pe(pred, NLP[idx]) + kl + rc       # ★学習ループと同じ基準
            t.learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(t.learner.brain.parameters(),
                                           t.learner.grad_clip)
            t.learner.optimizer.step()

    # ------------------------------------------------------- チェックポイント
    def _checkpoint(self, step):
        """★測るのはプラグイン。ここは呼ぶだけ。"""
        self.ctx.step = step
        for p in self.plugins:
            p.on_checkpoint(self.ctx)
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
        # ⚠️タグはASCIIのみ（Windowsのcp932で出せない文字を混ぜると print が例外を投げ
        #   学習が途中で落ちる。上付き2・絵文字で実際に2回踏んだ＝落とし穴 項35）
        if self._caps_accum:
            extra += f" da2={np.mean(self._caps_accum[-200:]):.4f}(lam={self.cfg.caps})"
        print(f"[seed{self.cfg.seed} rew={self.cfg.reward} age={self._cur_age:.2f}mo] "
              f"life={life_min:.0f}min | " + "  ".join(tags) +
              f" | noise={noise:.3f}(mat={t.ne.maturation:.2f}) {cereb_tag}{extra}"
              f" real={real_min:.0f}min", flush=True)

    # ------------------------------------------------------------ 本体
    def run(self):
        """★学習ループ。e_growth_train.py 1175〜1304行の写し。"""
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
                    env = self.env            # ★作り直したので持ち替える
            # ---- 感覚を受け取り、内部表現を作る -------------------------------
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
                    # 【逸脱/工学近似 ⚠️】向き（驚き大→探索）はEFE/LC-NE/予測符号化に基づく
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
            # ---- ★測る（プラグイン）------------------------------------------
            # ⚠️判断ごとに1回（K tick ぶんに1回）呼ぶ。元の実装と同じ頻度。
            #   接触のような一瞬の事象は tick 単位で見た方が正確だが、
            #   まず元と同じ数値が出ることを確かめるため頻度も合わせる。
            self.ctx.step = i + 1
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
                # （Selinger 2015 等の代謝最小化。⚠️二乗・λ・重みは近似＝感度確認対象）
                effort = float((a.detach() ** 2 * t.eff_w).sum())
                rew = rew_task - cfg.effort_cost * effort
                self._eff_accum.append(effort)
            # 【CAPS, Mysore et al. 2021】行動の急変にペナルティ。
            # ★λ=0 でも**必ず記録する**：OFF側の値が無いと「λが弱くて効かなかった」のか
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
        self.env.close()
        return out


def train(cfg, *, plugins=(), verbose=True, log_row=None):
    """設定から学習を1本回す。★これが新しい経路の本体。"""
    tr = Trainer(cfg, plugins=plugins, verbose=verbose, log_row=log_row).build()
    return tr.run()
