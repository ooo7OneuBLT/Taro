"""太郎一式（脳・学習器・神経調節・小脳）を組み立てる。

【なぜ切り出したか、2026-07-30】これは `E/scripts/e_growth_train.py` の
562〜700行を**そのまま写した**もの。元は930行の関数の中に埋まっていて、
53個のモジュール変数（`_LR` `_MUSCLE` …）を直接読んでいたため
実験ファイルから設定を渡せなかった。→ `run/config.py` の Config を受け取る形にする。

注意：写すときに守ったこと：**乱数を消費する順序を変えない**。
  順序が変わると同じシードでも違う初期値になり、過去の実験と比較できなくなる
  （落とし穴チェックリスト 項3「reset を1回足すだけで乱数列がずれる」）。
  順序は：env作成 → fusion → target_fusion → env.reset(seed) → 予測対象の次元を測る
        → brain → 脊髄CPG → learner → 神経調節 → 小脳 → モデル読み込み

【この中に「測り方」は入れない】測るのはプラグインの仕事（`run/plugins/`）。
【この中に「環境の作り方」も入れない】環境は `run/plugins/common/scene.py` だけが作る。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
_CORE = os.path.join(_ROOT, "taro_core")
for _sub in ("wrapper", "senses", "brain", "body"):
    _p = os.path.join(_CORE, "src", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _p in (os.path.join(_CORE, "tests"), os.path.join(_ROOT, "E", "scripts"),
           os.path.join(_ROOT, "D", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
import numpy as np                                              # noqa: E402

from fusion import MinimalFusion                                # noqa: E402
from taro_brain_motor import TaroBrainWithMotor                 # noqa: E402
from basal_ganglia import TaroLearner                           # noqa: E402
from dopamine import Dopamine                                   # noqa: E402
from locus_coeruleus import LocusCoeruleus                      # noqa: E402
from developmental_clock import DevelopmentalClock              # noqa: E402
from cerebellum_motor import MotorCerebellum                    # noqa: E402
from learning_progress import LearningProgress                  # noqa: E402
from homeostatic_scaling import HomeostaticScaling              # noqa: E402
from test_phase8_motor_learning import CombinedParams, rescale_action, to_tensor  # noqa: E402

mse = torch.nn.functional.mse_loss

# シナジーの関節index（MIMo身体の配列。d_c5_motor_quality.py と同一）
# 注意：【逸脱・Tier3・2026-07-30】「まとめてしか動かせない」を**配線で作っている**。
#   人間（および國吉研の胎児モデル）は、振動子を互いに繋がず、
#   身体・床・羊水を介した物理的な力の伝達で位相が揃う（引き込み現象）。
#   ＝「まとめて動く」は**結果**であって入力ではない。
#   国吉 & Sangawa 2006（Biological Cybernetics 95(6)）は
#   「複数筋を協調させる回路を一切事前に組み込まずに」全身協調運動を創発させている。
#   ⇒ 人間模倣からの逸脱リスト「2026-07-30 シナジーを配線で作っている」を参照。
#     注意：太郎の環境には羊水も子宮壁もないので、外すと単に消える可能性がある（要検証）。
LEG_R = [72, 73, 75, 76]
LEG_L = [81, 82, 84, 85]
ARM_R = [14, 15, 17, 19]
ARM_L = [43, 44, 46, 48]


class Taro:
    """太郎そのもの。脳・学習器・神経調節・小脳・海馬を持つ。

    注意：ここは「太郎の中身」だけ。環境（env）は組み立てに必要なので受け取るが、
      **保持しない**（体を作り直しても Taro は作り直さないため）。
      env が要る操作（step / reset）は呼び出し側（run/trainer.py）が持つ。

    属性:
        brain, fusion, target_fusion, nat_head, emb_proj
        learner, dop, ne, homeo, dev_clock, cereb, cere_opt, hippo, lp
        n_act      脳が出す行動の次元（拮抗筋モードでは環境の半分）
        out_dim    予測対象の次元
        blocks     予測対象のブロック境界 [(start, end, 名前), ...]
    """

    def __init__(self, cfg, env, *, seed, verbose=True):
        self.cfg = cfg
        self.seed = int(seed)
        self.blocks = None            # ln_prop が最初の呼び出しで作る

        # ---- 触覚の次元（reset せずに読める。reset を足すと乱数列がずれる）----
        # 注意：get_sensor_count() は「センサ点の数」で、観測は1点あたり力の3成分。
        touch_dim = int(env.observation_space["touch"].shape[0]) if cfg.touch else 0
        # 固有感覚の次元は駆動モードで変わる（関節モード=621 / 筋肉モード=801）
        prop_dim_space = int(env.observation_space["observation"].shape[0])

        # ---- 視覚の解像度（fusion に渡す）------------------------------------
        # 注意：シーンを使う場合、環境は**常に**視覚を持つ（e_scene が vision_params を渡す）。
        #   vision=False は「脳が視覚を無視する」アブレーションになる（環境は変わらない）。
        vres = 0
        if cfg.vision:
            from e_toy_env import VISION_RES
            vres = VISION_RES

        # ---- 体性感覚系（触覚ONのときだけ）----------------------------------
        soma_layout = None
        if cfg.somatosensory and cfg.touch:
            from somatosensory_cortex import build_sensor_layout
            soma_layout, soma_total = build_sensor_layout(env.unwrapped.model,
                                                          env.unwrapped.touch)
            assert soma_total == touch_dim, f"soma total {soma_total} != {touch_dim}"
            if verbose:
                print(f"[体性感覚系] SomatosensoryCortex 有効：部位数={len(soma_layout)}"
                      f" 触覚総次元={touch_dim}", flush=True)

        # ---- ① 融合層（感覚をまとめる）--------------------------------------
        # target_fusion は**凍結した別インスタンス**（RND式）。予測側と正解側が
        #   同じ学習中の層だと「出力を平坦にすれば当たる」抜け道で崩壊する（目標Cで実際に踏んだ）。
        self.fusion = MinimalFusion(touch_dim, vision_res=vres,
                                    proprio_dim=prop_dim_space,
                                    somatosensory_layout=soma_layout)
        self.target_fusion = MinimalFusion(touch_dim, vision_res=vres,
                                           proprio_dim=prop_dim_space,
                                           somatosensory_layout=soma_layout).freeze()
        if cfg.somatosensory and cfg.touch and self.fusion.touch is not None and verbose:
            print(self.fusion.touch.summary(), flush=True)

        # ---- ② 行動の次元 ---------------------------------------------------
        n_env_act = env.action_space.shape[0]
        # 拮抗筋モード：脳・CPG・prev_a はすべて n_joint 次元、環境には to_env_action で
        # 2*n_joint に写像して渡す。既定は n_act == n_env_act。
        self.n_act = n_env_act // 2 if (cfg.is_muscle and cfg.antagonist) else n_env_act

        # ---- ③ 最初の reset（必ず seed を渡す）------------------------------
        # 環境の乱数（env.unwrapped.np_random）は gym が別に管理しており、
        # torch.manual_seed も np.random.seed も効かない。シードなし reset だと
        # 毎回ちがう姿勢から始まり再現できない（2026-07-15 のバグ）。
        obs, _ = env.reset(seed=self.seed)
        self.sdim = self.fusion.encode(obs).shape[0]
        self.prop_dim = to_tensor(obs["observation"]).shape[0]

        # ---- ④ 予測対象の次元を**測る**（手計算だとズレる）--------------------
        self.out_dim = int(self.encode_target(obs).shape[0])
        if cfg.target_has_vision and verbose:
            bd = "／".join(f"{nm}:{e - s}" for (s, e, nm) in (self.blocks or []))
            print(f"[予測対象] {cfg.target_kind} → 全{self.out_dim}次元  内訳 {bd}")
            print(f"     誤差はブロックごとに平均してから足す（次元数の影響を除く。"
                  f"λ_v={cfg.lam_v}）")

        # ---- ⑤ 脳 -----------------------------------------------------------
        self.brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=self.sdim,
                                        n_actuators=self.n_act, proprio_dim=self.out_dim)
        # 【運動性喃語（脊髄CPG）】noise=colored のとき太郎の中で色付き探索を有効化する。
        # 既定 white では呼ばれない＝spinal_cpg=None＝白色ガウス＝従来と数値完全一致。
        if str(cfg.noise) == "colored":
            # 注意：シナジーの index は90-actuator前提。筋肉モードでは pair_offset を入れて
            #   対になる筋（伸ばす側）に符号反転で同じシナジーを混ぜる（spinal_cord/cpg.py）。
            pair_offset = (self.n_act // 2) if (cfg.is_muscle and not cfg.antagonist) else 0
            self.brain.enable_spinal_babble(
                self.n_act, leg_r=LEG_R, leg_l=LEG_L, arm_r=ARM_R, arm_l=ARM_L,
                beta=cfg.beta, synergy=bool(cfg.synergy), syn_w=cfg.syn_w, seed=self.seed,
                antagonist=(cfg.is_muscle and cfg.antagonist),
                co_activation=cfg.coactivation, pair_offset=pair_offset)
            if verbose:
                print(f"[脊髄CPG] 色付き探索ON: β={cfg.beta} synergy={bool(cfg.synergy)}"
                      f" syn_w={cfg.syn_w}"
                      f"{f' pair_offset={pair_offset}' if pair_offset else ''}"
                      f"{' 【拮抗筋モードON】coactivation=' + str(cfg.coactivation) if cfg.is_muscle and cfg.antagonist else ''}",
                      flush=True)
        # 【2026-07-25】D-a/D-b の層は**太郎の中（core）のもの**を使う。
        # 旧実装はここで別に作っており、core の層は作られるだけで使われていなかった
        # （＝脳が二重に存在していた）。旧名を別名として残す（参照箇所が多いため）。
        self.emb_proj = self.brain.motor_input_proj
        self.nat_head = self.brain.forward_model_head

        # ---- ⑥ 学習器と神経調節 ----------------------------------------------
        self.learner = TaroLearner(CombinedParams(self.brain, self.fusion), lr=cfg.lr)
        self.dop = Dopamine()
        self.ne = LocusCoeruleus(relative=bool(cfg.ne_relative))
        self.homeo = HomeostaticScaling(dim=self.sdim)
        self.dev_clock = DevelopmentalClock()   # ③発達年齢（累積学習回数）
        # 運動小脳。ON/OFFで乱数列を揃えるため cfg.cerebellum に関わらず**常に構築**する
        # （使う/学習するのは ON のときだけ。gate/imitation は乱数を消費しない）。
        self.cereb = MotorCerebellum(self.brain.latent_dim, self.n_act)
        self.cere_opt = torch.optim.Adam(self.cereb.parameters(), lr=cfg.lr)
        self.hippo = self.brain.hippocampus     # 睡眠リプレイのバッファ（core に一元化済み）
        self.lp = LearningProgress()            # 予測誤差の速い/遅い走行平均

        # ---- ⑦ 続きから学習する（形が合う層だけ）------------------------------
        if cfg.model:
            self._load(cfg.model, verbose=verbose)

        # ---- ⑧ 努力コストの重み ---------------------------------------------
        # 筋力（最大トルク）が大きい筋ほど動かすとコストが高い（代謝の標準：活性化²×筋サイズ）。
        # 注意：体を作り直しても**更新していない**（元の実装もそうだった）。
        #   体を育てる実験で努力コストを使うときは、ここが古い体の値であることに注意。
        gear = np.abs(env.unwrapped.model.actuator_gear[:self.n_act, 0]).astype(np.float32)
        self.eff_w = torch.tensor(gear / (gear.sum() + 1e-8))
        self.first_obs = obs

    # ------------------------------------------------------------ 読み込み
    def _load(self, path, *, verbose=True):
        def _match(module, sd, tag):
            own = module.state_dict()
            matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            module.load_state_dict(matched, strict=False)
            if verbose:
                print(f"  [{tag}] ロード{len(matched)}/{len(own)}層", flush=True)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"続きから学習：{os.path.basename(path)} を読み込み", flush=True)
        _match(self.brain, blob["brain"], "脳")
        self.fusion.insula.load_state_dict(blob["fusion_insula"])
        self.fusion.proprio.load_state_dict(blob["fusion_proprio"])
        self.fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
        # 触覚のエンコーダ。save は保存していたのに**読み戻していなかった**
        #   （2026-07-30 の点検で発覚）。黙って白紙に戻るので、
        #   「続きから学習できている」ように見えて触覚だけ学習しなおしになる。
        if "fusion_touch" in blob:
            if self.fusion.touch is not None:
                _match(self.fusion.touch, blob["fusion_touch"], "触覚")
            else:
                print("注意[load] 保存されたモデルは触覚あり、いまの設定は触覚なしです。"
                      "触覚のエンコーダは読み込みません（taro.touch を確認）", flush=True)
        elif self.fusion.touch is not None:
            print("注意[load] いまの設定は触覚ありですが、保存されたモデルに触覚が"
                  "ありません。触覚のエンコーダは白紙から学習します", flush=True)
        if "emb_proj" in blob:
            print("  [注意] 旧形式のチェックポイント（emb_proj/nat_head が別層）です。"
                  "層の構成が変わったため、その2層は読み込まれません＝学習しなおしになります。",
                  flush=True)
        if self.cfg.cerebellum and "cereb" in blob:
            _match(self.cereb, blob["cereb"], "小脳")

    # -------------------------------------------------------- 予測の対象
    def encode_target(self, obs):
        """予測する対象を作る（元 `ln_prop`）。既定は固有感覚のみ。

        2つの設計判断（詳細は E/scripts/e_target.py）：
          (1) **凍結した別インスタンス**のエンコーダを使う（RND式）。学習中のエンコーダを
              正解側に使うと「出力を平坦にすれば当たる」抜け道で崩壊する。
          (2) **固有感覚と視覚を別々に layer_norm** してから連結する。全体を一度に
              正規化すると621次元が平均・分散を支配して64次元が埋もれる（3回踏んだ罠）。
        """
        ln = torch.nn.functional.layer_norm
        cfg = self.cfg
        v = to_tensor(obs["observation"])
        # 【B案】somatosensory=True のときは生の触覚を混ぜず、S1相当のembedをブロックで足す
        if cfg.touch and cfg.touch_mode == "target" and not cfg.somatosensory:
            v = torch.cat([v, to_tensor(obs["touch"])])
        parts = [ln(v, v.shape).detach()]
        names = ["prop"]
        f = self.target_fusion
        if cfg.somatosensory and cfg.touch and cfg.touch_mode == "target":
            if getattr(f, "touch", None) is not None and "touch" in obs:
                with torch.no_grad():
                    e = f.touch(to_tensor(obs["touch"]))
                parts.append(ln(e, e.shape)); names.append("touch_embed")
        if cfg.target_has_vision:
            with torch.no_grad():             # 正解側は勾配を流さない（RND式）
                # 内受容は入れない（2026-07-20 の文献調査による判断）：人間は内受容の
                #   予測誤差を自律反射（心拍・血管）で解消するが、太郎にはその出力が無い
                #   ＝**誤差を減らす手段が構造的に存在しない**ので progress報酬が生まれない。
                if cfg.target_has_all:
                    for nm, enc, key in (("vest", getattr(f, "vestibular", None), "vestibular"),
                                         ("touch", getattr(f, "touch", None), "touch")):
                        if enc is None or key not in obs:
                            continue
                        e = enc(to_tensor(obs[key]))
                        parts.append(ln(e, e.shape)); names.append(nm)
                if getattr(f, "vision", None) is not None and "eye_left" in obs:
                    e = f.vision(obs["eye_left"], obs["eye_right"])
                    parts.append(ln(e, e.shape)); names.append("vision")
        if self.blocks is None or len(self.blocks) != len(parts):
            self.blocks, o = [], 0
            for nm, p in zip(names, parts):
                self.blocks.append((o, o + int(p.shape[-1]), nm))
                o += int(p.shape[-1])
        return torch.cat(parts, dim=-1).detach() if len(parts) > 1 else parts[0]

    def block_pe(self, pred, target):
        """予測誤差＝**ブロックごとに平均してから足す**（次元数の影響を除く）。

        【なぜ】従来は連結したベクトル全体を1回で平均していたので、寄与が次元数比で
        決まっていた（固有感覚621 + 視覚64 → 視覚の寄与は 9.3%）。これは「視覚が
        重要でない」という判断ではなく、**次元数という無関係な量**が重みを決めている状態。
        同じ罠を目標D0とE1で計3回踏んだ。
        根拠＝Ohata & Tani 2020（`1/(2Rp)`）、Idei et al. 2025（1,150倍差を正規化のみで処理）、
        Ichiwara & Ogata 2022（`1/(H·W·C)`）。［参考文献リスト §目標E-17］
        注意：(1+λ_v) で割るのは全体のスケールを保つため（割らないと「視覚を足した効果」と
          「学習率が実質変わった効果」が混ざる＝交絡）。
        """
        if not self.blocks or len(self.blocks) <= 1:
            return mse(pred, target)          # 固有感覚のみ＝従来と完全に同一
        tot, wsum = 0.0, 0.0
        for (s, e, nm) in self.blocks:
            lam = self.cfg.lam_v if nm == "vision" else 1.0
            tot = tot + lam * mse(pred[..., s:e], target[..., s:e])
            wsum += lam
        return tot / max(wsum, 1e-9)

    # ------------------------------------------------------------ 脳の操作
    def infer_latent(self, sv, prev_a, cf, h):
        """[感覚, 前回の行動] → 内部表現 z（元 `zc`）。太郎の infer_latent を呼ぶだけ。"""
        return self.brain.infer_latent(sv, prev_a, cf, h)

    def act_mean(self, z):
        """決定的な行動平均（評価・agency用）。太郎の act_deterministic を呼ぶだけ。"""
        return self.brain.act_deterministic(
            z, cerebellum=(self.cereb if self.cfg.cerebellum else None))

    def motor_drive(self, z):
        """行動の平均と揺らぎ。運動野の精密制御＋小脳の自動化ブレンド。"""
        return self.brain.motor_drive(
            z, self.ne.get_ne_level(),
            cerebellum=(self.cereb if self.cfg.cerebellum else None))

    def infer_goal_action(self, z, clp, init_mean, g, n_steps=15, lr_inf=0.1):
        """目標指向：凍結した順モデルを反転し、望む感覚 g に届く行動を推論する。

        注意：2026-07-30 の実測でこの機構は**有害**と判明（config.py の注記参照）。
          原典（Rolf, Steil & Gienger 2010）は目標を**手先位置の低次元**に取るが、
          ここは固有感覚621次元まるごとを目標にしている＝別物。
        """
        target = (g - clp).detach()
        raw = torch.atanh(torch.clamp(init_mean, -0.999, 0.999)).detach().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr_inf)
        for _ in range(n_steps):
            opt.zero_grad()
            ((self.nat_head(torch.cat([z, torch.tanh(raw)], dim=-1)) - target) ** 2).mean().backward()
            opt.step()
        return torch.tanh(raw).detach()

    def init_state(self, obs):
        """学習ループが持つ「いまの状態」の初期値。"""
        return {"obs": obs, "hidden": self.brain.init_motor_hidden(),
                "prev_a": torch.zeros(self.n_act)}

    def save(self, path, *, extra=None):
        """確立した自己モデルを保存する。条件も一緒に入れる。

        注意：条件を入れないと、あとから「どの設定で保存されたか」が分からない
          （2026-07-25 に感度分析のモデルが同一だと分かっても原因を追えなかった）。
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {"brain": self.brain.state_dict(),
                "fusion_insula": self.fusion.insula.state_dict(),
                "fusion_proprio": self.fusion.proprio.state_dict(),
                "fusion_vestibular": self.fusion.vestibular.state_dict(),
                "cereb": self.cereb.state_dict(),
                "config": dict(self.cfg.as_dict(),
                               sdim=self.sdim, prop_dim=self.prop_dim,
                               out_dim=self.out_dim, n_act=self.n_act,
                               **(extra or {}))}
        if self.fusion.touch is not None:
            blob["fusion_touch"] = self.fusion.touch.state_dict()
        torch.save(blob, path)
        print(f"SAVED MODEL {path}", flush=True)
