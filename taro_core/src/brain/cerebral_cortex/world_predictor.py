# -*- coding: utf-8 -*-
"""連合野の世界予測器（M7a・測るだけ／M7b-1・物ごとに動かす）。

【出典】仕様_M7a_世界の予測器_測るだけ_2026-09-08.md「後半：実装担当向け技術付録」1節。
文献：F/docs/二語文/文献調査/2026-09-08_予測の対象と階層_人間側.md・..._AI側.md
（Kiebel 2008 の速い/遅い段=0.25秒/2秒、Tani MTRNN の τ=5/70 を目安にした
工学的選択。当てはめの数値そのものは[Tier3]）。

【どこに置くか】大脳皮質の連合野に相当。速い層＝頭頂連合野（物の位置・見た目の
0.5秒先）、遅い層＝前頭前野（数秒の出来事のまとまり）。既存の運動用の予測
（`nat_head`等）は一切触らない・書き換えない。

【v1の範囲】測るだけ。太郎の他の部品（報酬・行動・学習）には一切書き込まない。
毎tick、前tickの予測と今の観測の誤差で勾配を1回流すだけ（この予測器自身の
重みだけが動く）。睡眠の再生は無し。精度の重み付け（注意）は入れない
（決定6：「注意中の物1つだけを予測する」ことで代用）。

【入力 x（1本に連結、全て float。呼び出し側=run/trainer.pyが組み立てる）】
    obj_state (6)  = [present, visible, vanished, pos_x, pos_y, area]
                     pos は画像幅・高さ(224px、object_files.py:438のCENTER_X/Y=112.0
                     から逆算)で0..1に正規化、area は log1p(area)/10。
                     attended_object が None なら全部0（present=0）。
    obj_vec  (obj_vec_dim) = last_seen_vec（None なら0ベクトル）をLayerNormしたもの
                     （このモジュール内でLayerNormする＝呼び出し側は生の値を渡してよい）。
    parent   (1+chunk_emb+1) = [spoke_now,
                     Embedding(n_chunks+1, chunk_emb)[first_chunk_id+1]
                     （喋っていなければindex0=零固定・padding_idx=0で実現）,
                     min(time_since_parent_spoke, 10)/10]
    act      (act_dim) = 直前の運動 a（tanh範囲のまま、そのまま渡す）

【なぜobj_vec_dim=384か、2026-09-08】object_files.py:470-476のself._attended_last_seen_vec
は ctx.last_vision_vec（run/trainer.py:836）をそのままコピーしたものであり、
ctx.last_vision_vecは_vision_channels(o)の出力（同ファイル296-334行）。この値は
run/taro_setup.py:597-598でVisualProjection(in_dim=384, ...)に渡されている値と同一の
経路（同じ_vision_backend_encode出力）なので384次元で固定した[実装判断・Tier3]。
仕様書は「分からなければ最初の非None値で遅延構築」を許可しているが、384は
コード上の複数箇所から一意に決まる値であり、遅延構築の複雑さ（GRU入力次元が
可変になり保存互換性が壊れやすい）を避けるためこちらを採った。想定と異なる次元の
ベクトルが来たら早期にValueErrorで止まるようにした（黙って壊れるより安全）。

【追記2026-09-08】遅い層にも頭head_ctx（h_slow→h_fast次元）を足した。的は
h_fast（detach）。err_slowとしてerr_totalとは別に戻す。詳細は_forward_tick
のdocstring参照。

【追記2026-09-09・M7b-1・物ごとの記憶】仕様_M7b-1_物ごとの予測器と驚きの配線_
2026-09-09.md「後半」1節。重み（学習）は1本を共有したまま、隠れ状態・前tickの
予測・誤差の基準線だけを物体ファイルごとに持てるようにした（`_ObjState`／
`self._states: dict[file_id -> _ObjState]`）。既存の単一物用の呼び出し
（`step(input_now, target_now)`）は `step_for(None, input_now, target_now)` の
別名として残しており、`cfg.world_predictor.multi_object` を使わない実験
（既定）ではこのファイルの計算経路は1ビットも変わらない。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ObjState:
    """物1つぶんの「記憶」（学習される重みではない）。

    速い層・遅い層の隠れ状態、直前tickに作った予測、tbptt窓の累積損失、
    誤差の基準線（err_total用・err_state用）をまとめて持つ。file_id=None の
    ときは「注意中の物1つだけ」という従来（M7a）の単一状態に相当する。
    """

    __slots__ = ("h_fast", "h_slow", "pred", "tick_count",
                 "loss_accum", "accum_count",
                 "baseline", "var_ema",            # err_total の基準線（従来どおり）
                 "baseline_state", "var_state")    # err_state だけの基準線（M7b-1で追加）

    def __init__(self):
        self.h_fast = None
        self.h_slow = None
        self.pred = None
        self.tick_count = 0
        self.loss_accum = None
        self.accum_count = 0
        self.baseline = None
        self.var_ema = None
        self.baseline_state = None
        self.var_state = None


class WorldPredictor(nn.Module):
    """連合野の世界予測器（MTRNN型2段）。速い層=頭頂連合野、遅い層=前頭前野に相当。

    測るだけ（v1）。太郎の他の部品には一切書き込まない。M7b-1で「物ごとに
    記憶を持つ」多物モードを足したが、重み・optimizer自体は1本のまま共有する。
    """

    def __init__(self, obj_vec_dim, n_chunks, act_dim, h_fast=64, h_slow=32,
                 tau_fast=5, tau_slow=40, chunk_emb=16, lr=1e-3, tbptt=10,
                 grad_clip=1.0, block_norm=False, multi_object=False, max_files=4):
        super().__init__()
        # 【block_norm・2026-09-09・F2-91】入力4ブロック（状態の印6／見た目384／親18／運動180）を
        #   それぞれ √次元 で割り、網に入るときの声の大きさを対等にする（感覚皮質の割り算による
        #   正規化＝divisive normalization の工学版）。F2-89/90 の実測：印の列の重みが育つのに
        #   4000回かかった原因の候補＝見た目が印の8倍の大きさで入っていた。既定False＝既定不変
        self.block_norm = bool(block_norm)
        self.block_norm_alpha = 0.01      # 大きさの移動平均の速さ（約100tick＝10秒で馴染む）
        self.block_norm_eps = 1e-3
        self.register_buffer("_blk_ms", torch.ones(4))   # ブロックごとの ||x||² の移動平均（初期1＝最初は割らないのと同じ）
        self.obj_vec_dim = int(obj_vec_dim)
        self.n_chunks = int(n_chunks)
        self.act_dim = int(act_dim)
        self.h_fast_dim = int(h_fast)
        self.h_slow_dim = int(h_slow)
        self.tau_fast = float(tau_fast)
        self.tau_slow = float(tau_slow)
        self.chunk_emb_dim = int(chunk_emb)
        self.tbptt = int(tbptt)
        self.grad_clip = float(grad_clip)

        # 【M7b-1・2026-09-09】多物モード（物ごとに記憶を持つ）。仕様「後半」1節：
        #   「多物モードは tbptt=1 を要求。理由：tbptt 窓の flush（backward＋
        #   optimizer.step）は物ごとに独立に起きるので、窓をまたぐ計算グラフが
        #   物どうしで混ざると_build_input docstringのin-place事故が再発する」。
        #   既定False・呼び出し側がstep()（=step_for(None,...)）しか使わなければ
        #   1ビットも既存挙動は変わらない。
        self.multi_object = bool(multi_object)
        self.max_files = int(max_files)
        if self.multi_object and self.tbptt != 1:
            raise ValueError(
                "WorldPredictor: multi_object=True のとき tbptt は1でなければ"
                f"ならない（実際={self.tbptt}）。仕様_M7b-1「後半」1節参照。")

        # ---- 入力の組み立て -------------------------------------------------
        self.obj_vec_norm = nn.LayerNorm(self.obj_vec_dim)
        # padding_idx=0で「喋っていない」印(index0)を常にゼロベクトルに固定する
        # （spec「index0=零固定」。学習でも勾配が入らずゼロのまま保たれる）。
        self.chunk_embedding = nn.Embedding(
            self.n_chunks + 1, self.chunk_emb_dim, padding_idx=0)
        self.x_dim = 6 + self.obj_vec_dim + (2 + self.chunk_emb_dim) + self.act_dim

        # ---- MTRNN型2段（漏れ積分GRU）----------------------------------------
        self.gru_fast = nn.GRUCell(self.x_dim + self.h_slow_dim, self.h_fast_dim)
        self.gru_slow = nn.GRUCell(self.h_fast_dim, self.h_slow_dim)

        # ---- 頭（fast hから）--------------------------------------------------
        self.head_state = nn.Linear(self.h_fast_dim, 6)
        self.head_vec = nn.Linear(self.h_fast_dim, self.obj_vec_dim)
        self.head_speak = nn.Linear(self.h_fast_dim, 1)
        self.head_chunk = nn.Linear(self.h_fast_dim, self.n_chunks)

        # ---- 頭（slow hから）：追記2026-09-08 -----------------------------------
        # 予測符号化の型（各段は一段下の活動を予測する）に合わせ、遅い層にも
        # 頭を1つ足す。的は速い層の状態h_fast（detachした値。速い層を歪めない
        # ため勾配は遅い層側だけに流す）。
        self.head_ctx = nn.Linear(self.h_slow_dim, self.h_fast_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # 【M7b-1・2026-09-09】物ごとの記憶の入れ物。file_id=None＝「注意中の物
        # 1つだけ」という従来（M7a）の単一状態。多物モードでも重み・optimizer・
        # _blk_msは上のとおり1本のまま（ここに置くのは記憶だけ）。
        self._states = {}

        # 誤差の基準線（慣れの確認用。学習には使わない、測るだけ）は_ObjState側へ
        # 移した（file_id=Noneの状態として同じ辞書に入る）。

        # 【M7b-1・2026-09-09・実装判断、仕様に無かった点】多物モード専用の
        # 「1tickに1回だけ・全物まとめてbackward+optimizer.step」用の共有累積。
        # なぜ必要か（机上確認で実測したバグ）：仕様「後半」1節は「tbptt=1なら
        # 毎tick flush＝グラフは1tickで閉じる」としているが、物ごとに独立に
        # （＝1呼び出しの中でbackward+optimizer.stepまで完結させる形で）flush
        # すると、ある物Aのflush（optimizer.step、重みを直接その場でin-place更新）が、
        # まだflushしていない物Bの「前tickに作った予測」の計算グラフが参照している
        # 同じ重みテンソルを書き換えてしまい、Bを後でbackwardしようとした瞬間に
        # `RuntimeError: one of the variables needed for gradient computation has
        # been modified by an inplace operation`になる（実測・obj_vec_dim=8の
        # ミニ検証で再現、_build_input docstring記載の既知の落とし穴と同型）。
        # 対策：1tickの中で「全物の誤差を集めてから1回だけbackward+step」する
        # 二段構え（observe_for×全物→flush_pending×1回→predict_for×全物）に
        # した。これにより物どうしの間で重みが書き換わるタイミングが1tickに
        # ちょうど1回（flush_pending呼び出し時）だけになり、単一状態(step_for)
        # と同じ「生成してから消費するまでの間に他の誰も重みを書き換えない」
        # という不変条件を保てる。
        self._multi_loss_accum = None
        self._multi_touched = set()

    # ------------------------------------------------------------------
    def _get_state(self, file_id):
        state = self._states.get(file_id)
        if state is None:
            state = _ObjState()
            self._states[file_id] = state
        return state

    def drop(self, file_id):
        """その物の記憶を捨てる（重みには触らない）。"""
        self._states.pop(file_id, None)

    def active_ids(self):
        """今記憶を持っている物のidの一覧（file_id=Noneの単一状態を含む）。"""
        return list(self._states.keys())

    def reset_hidden(self):
        """隠れ状態と直前の予測を全物ぶんリセットする（エピソード境界などで呼ぶ）。

        重み・optimizer・baselineは変えない（reset_hiddenの名の通り"隠れ状態"だけ
        ……ではあるが、物ごとの記憶自体を作り直す方が単純なため、ここでは
        _statesを空にする＝次のstep_for呼び出しで作り直される）。
        """
        self._states = {}
        self._multi_loss_accum = None
        self._multi_touched = set()

    # ------------------------------------------------------------------
    # 【M7b-1・2026-09-09】多物モード専用の二段API（observe_for→flush_pending→
    # predict_for）。クラス冒頭のコメント「なぜ必要か」参照。呼び出し順は
    # 1tickにつき必ず「選ばれた全物についてobserve_for」→「flush_pendingを1回」
    # →「選ばれた全物についてpredict_for」（run/trainer.py._world_predictor_step
    # が守る）。single-object（step/step_for、file_id=None）は一切変更なし。
    def observe_for(self, file_id, target_now):
        """多物モードphase1：state[file_id]に前tickの予測(state.pred)が
        あれば、今tickの実測(target_now)と比べて誤差を計算し、
        self._multi_loss_accum（共有・このtick分）へ積む（まだbackwardしない）。
        state.predがまだ無ければ（このfile_idを初めて見る）誤差はNoneのまま
        何も積まずに返す（従来のstep()の「初回はerr系がNone」と同じ）。

        戻り値：{"err_state","err_vec","err_parent","err_total","baseline",
                "z","z_state"}（意味はstep_for参照）。
        """
        state = self._get_state(file_id)
        self._init_hidden_if_needed(state)
        device = self._device()

        result = {"err_state": None, "err_vec": None, "err_parent": None,
                  "err_total": None, "baseline": None, "z": None, "z_state": None}
        if state.pred is None:
            return result

        state_target = torch.tensor(
            list(target_now["obj_state"]), dtype=torch.float32,
            device=device).view(1, -1)
        bce_part = F.binary_cross_entropy_with_logits(
            state.pred["state_logit"][:, :3], state_target[:, :3])
        mse_part = F.mse_loss(state.pred["state_logit"][:, 3:], state_target[:, 3:])
        err_state = bce_part + mse_part

        vec_target_raw = target_now.get("obj_vec")
        if vec_target_raw is None:
            vec_target = torch.zeros(1, self.obj_vec_dim, device=device)
        else:
            vec_target = torch.tensor(
                list(vec_target_raw), dtype=torch.float32,
                device=device).view(1, -1)
        err_vec = F.mse_loss(state.pred["vec_pred"], vec_target)

        speak_target = torch.tensor(
            [[float(target_now.get("parent_spoke", 0.0))]],
            dtype=torch.float32, device=device)
        err_speak = F.binary_cross_entropy_with_logits(
            state.pred["speak_logit"], speak_target)
        chunk_id_plus1 = target_now.get("parent_chunk_id_plus1")
        if (target_now.get("parent_spoke", 0.0) and chunk_id_plus1 is not None
                and 1 <= int(chunk_id_plus1) <= self.n_chunks):
            chunk_target = torch.tensor(
                [int(chunk_id_plus1) - 1], dtype=torch.long, device=device)
            err_chunk = F.cross_entropy(state.pred["chunk_logit"], chunk_target)
        else:
            err_chunk = torch.zeros((), device=device)
        err_parent = err_speak + err_chunk

        err_total = err_state + err_vec + err_parent

        err_state_v = float(err_state.item())
        err_total_v = float(err_total.item())
        result["err_state"] = err_state_v
        result["err_vec"] = float(err_vec.item())
        result["err_parent"] = float(err_parent.item())
        result["err_total"] = err_total_v

        if state.baseline is None:
            state.baseline = err_total_v
            state.var_ema = 0.0
        else:
            state.baseline = 0.99 * state.baseline + 0.01 * err_total_v
            dev2 = (err_total_v - state.baseline) ** 2
            state.var_ema = 0.99 * state.var_ema + 0.01 * dev2
        std_ema = state.var_ema ** 0.5 if state.var_ema is not None else 0.0
        result["baseline"] = state.baseline
        result["z"] = (err_total_v - state.baseline) / (std_ema + 1e-6)

        if state.baseline_state is None:
            state.baseline_state = err_state_v
            state.var_state = 0.0
        else:
            state.baseline_state = 0.99 * state.baseline_state + 0.01 * err_state_v
            dev2_state = (err_state_v - state.baseline_state) ** 2
            state.var_state = 0.99 * state.var_state + 0.01 * dev2_state
        std_state = state.var_state ** 0.5 if state.var_state is not None else 0.0
        result["z_state"] = (err_state_v - state.baseline_state) / (std_state + 1e-6)

        self._multi_loss_accum = (err_total if self._multi_loss_accum is None
                                   else self._multi_loss_accum + err_total)
        self._multi_touched.add(file_id)
        return result

    def flush_pending(self):
        """多物モードphase1と2の間：このtickぶんobserve_forへ積んだ誤差
        （＝前tickのpredict_forが積んだerr_slow全部＋今tickのobserve_forが
        積んだerr_total全部。同じ重みバージョンで作られたものだけが混ざる）を
        1回だけbackward+optimizer.stepし、触った物のhiddenをdetachする
        （次tickへ計算グラフを持ち越さない＝tbptt=1の実体）。
        何も積まれていなければ何もしない（1tick目、まだ誰もobserve_forで
        誤差が出ていないときなど）。
        """
        if self._multi_loss_accum is None:
            return
        self.optimizer.zero_grad()
        self._multi_loss_accum.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()
        for fid in self._multi_touched:
            st = self._states.get(fid)
            if st is not None and st.h_fast is not None:
                st.h_fast = st.h_fast.detach()
                st.h_slow = st.h_slow.detach()
        self._multi_loss_accum = None
        self._multi_touched = set()

    def predict_for(self, file_id, input_now):
        """多物モードphase2：flush_pendingの"後"に呼ぶ（更新済みの重みで
        次tickの予測を作る）。今tickの入力からx_nowを組み立て、次tickの
        予測(state.pred/h_fast/h_slow)を作る。遅い層の予測誤差(err_slow)は
        self._multi_loss_accum（次tickのflush_pending向けの新しい積み立て）へ
        積む。戻り値はerr_slow（ログ用のfloat）。
        """
        state = self._get_state(file_id)
        self._init_hidden_if_needed(state)
        x_now = self._build_input(
            input_now["obj_state"], input_now.get("obj_vec"),
            input_now.get("parent_spoke", 0.0), input_now.get("chunk_id_plus1", 0),
            input_now.get("time_since_parent", 1.0), input_now.get("act"))
        pred, h_fast_new, h_slow_new = self._forward_tick(state, x_now)

        err_slow = F.mse_loss(pred["ctx_pred"], h_fast_new.detach())
        self._multi_loss_accum = (err_slow if self._multi_loss_accum is None
                                   else self._multi_loss_accum + err_slow)
        self._multi_touched.add(file_id)

        state.pred = pred
        state.h_fast = h_fast_new
        state.h_slow = h_slow_new
        state.tick_count += 1
        return float(err_slow.item())

    def _device(self):
        return next(self.parameters()).device

    def _init_hidden_if_needed(self, state):
        if state.h_fast is None:
            device = self._device()
            state.h_fast = torch.zeros(1, self.h_fast_dim, device=device)
            state.h_slow = torch.zeros(1, self.h_slow_dim, device=device)

    def _forward_tick(self, state, x_now):
        """x_now(1,x_dim)を流し、次tickの予測（生のlogit/値）と新しい隠れ状態を返す。

        state: そのtickを進める物の_ObjState（h_fast/h_slowを読み書きする）。

        式（仕様書1節）：
          h_f <- (1-1/tau_fast)*h_f + (1/tau_fast)*GRUCell_f([x, h_s], h_f)
          h_s <- (1-1/tau_slow)*h_s + (1/tau_slow)*GRUCell_s(h_f_new, h_s)
        （遅い層は「速い層のまとめを受け」＝更新した後の速い層h_f_newを使う）。
        """
        f_in = torch.cat([x_now, state.h_slow], dim=-1)
        h_fast_gru = self.gru_fast(f_in, state.h_fast)
        h_fast_new = ((1.0 - 1.0 / self.tau_fast) * state.h_fast
                      + (1.0 / self.tau_fast) * h_fast_gru)
        h_slow_gru = self.gru_slow(h_fast_new, state.h_slow)
        h_slow_new = ((1.0 - 1.0 / self.tau_slow) * state.h_slow
                      + (1.0 / self.tau_slow) * h_slow_gru)

        pred = {
            "state_logit": self.head_state(h_fast_new),   # (1,6)
            "vec_pred": self.head_vec(h_fast_new),          # (1,obj_vec_dim)
            "speak_logit": self.head_speak(h_fast_new),     # (1,1)
            "chunk_logit": self.head_chunk(h_fast_new),     # (1,n_chunks)
            # 【追記2026-09-08・実装判断、なぜ「今tickの中」で作るか】
            # 仕様の文面は「的は次tickのh_fast」（他の頭と同じく、tick k で
            # 作った予測をtick k+1で観測と比べる形）を示唆する。しかしそれを
            # 素直にやると、head_ctx/gru_slow/gru_fastのパラメータが、
            # 予測を作ったtick kの直後～答え合わせするtick k+1の間に
            # tick k+1側の別windowのflush（optimizer.step）で更新され得て、
            # 「予測を作った時の重み」と「答え合わせでbackwardする時の重み」
            # がずれ、in-place更新エラーになる（このファイル_build_inputの
            # docstringで既に実測済みの同種のバグと同じ構造）。
            # そのため、state.h_slow（今tickの入力を取り込む"前"の遅い層＝
            # 前tickまでの文脈）から作った予測 head_ctx(state.h_slow) を、
            # 同じtick内で計算されるh_fast_new（今tickの入力を取り込んだ"後"の
            # 速い層）と比べる形に変えた。「前tickまでの文脈から、今tickの
            # 入力でfast層がどう変わるかを当てる」という意味では仕様の
            # 意図（遅い層＝速い層の続きを予測）を保っており、かつ予測と
            # 答え合わせが同一tick内で完結するため上記のバグを避けられる
            # [Tier3・工学的選択]。
            "ctx_pred": self.head_ctx(state.h_slow),        # (1,h_fast_dim)
        }
        return pred, h_fast_new, h_slow_new

    def _build_input(self, obj_state, obj_vec, parent_spoke, chunk_id_plus1,
                      time_since_parent, act):
        """呼び出し側(run/trainer.py)が渡す生の値からx(1,x_dim)を組み立てる。

        obj_state: 長さ6のfloat列（既にpos/areaの正規化済み）
        obj_vec: 長さobj_vec_dimのfloat列（Noneなら0ベクトル）
        parent_spoke: 0.0/1.0
        chunk_id_plus1: int（0=喋っていない/未知、1..n_chunksが実際の塊idの+1）
        time_since_parent: min(...,10)/10 済みのfloat
        act: 長さact_dimのfloat列（Noneなら0ベクトル）

        【なぜstep_for()の中でしか呼ばないか、2026-09-08・実装時に実測したバグ】
        obj_vec_norm（LayerNorm）とchunk_embeddingは学習対象のパラメータであり、
        ここで作った出力(x)は次tickの予測(_forward_tick)の入力として使われ、
        その予測の誤差はさらに"あとの"tbptt窓でbackwardされる。もしx_nowを
        step_for()の"外"（flushより前）で作ってしまうと、「flushでこのtick中に
        optimizer.step()が起きた直後」に、そのflush"前"の重みで作ったxの
        計算グラフが後の窓のbackwardに紛れ込み、
        `RuntimeError: one of the variables needed for gradient computation
        has been modified by an inplace operation`（実測・obj_vec_dim=8の
        ミニ検証で再現）になる。そのためstep_for()は生の値を受け取り、
        flush（backward+optimizer.step）が終わった"後"にこの関数を呼ぶ
        （このtickの他のパラメータ更新が全部済んでから入力を作る）。
        """
        device = self._device()
        obj_state_t = torch.tensor(list(obj_state), dtype=torch.float32,
                                    device=device).view(1, -1)
        if obj_state_t.shape[-1] != 6:
            raise ValueError(f"obj_state は6次元のはず（実際={obj_state_t.shape[-1]}）")
        if obj_vec is None:
            vec_t = torch.zeros(1, self.obj_vec_dim, device=device)
        else:
            vec_t = torch.tensor(list(obj_vec), dtype=torch.float32,
                                  device=device).view(1, -1)
            if vec_t.shape[-1] != self.obj_vec_dim:
                raise ValueError(
                    f"obj_vecの次元が構築時の想定({self.obj_vec_dim})と違う "
                    f"（実際={vec_t.shape[-1]}）。世界の予測器は384次元の"
                    "last_seen_vecを前提に組んでいる。")
        vec_t = self.obj_vec_norm(vec_t)
        idx = max(0, min(int(chunk_id_plus1), self.n_chunks))
        chunk_vec = self.chunk_embedding(
            torch.tensor([idx], dtype=torch.long, device=device))
        parent_t = torch.cat([
            torch.tensor([[float(parent_spoke)]], dtype=torch.float32, device=device),
            chunk_vec,
            torch.tensor([[float(time_since_parent)]], dtype=torch.float32, device=device),
        ], dim=-1)
        if act is None:
            act_t = torch.zeros(1, self.act_dim, device=device)
        else:
            act_t = torch.tensor(list(act), dtype=torch.float32,
                                  device=device).view(1, -1)
            if act_t.shape[-1] != self.act_dim:
                raise ValueError(
                    f"actの次元が構築時の想定({self.act_dim})と違う "
                    f"（実際={act_t.shape[-1]}）")
        if self.block_norm:
            # 各ブロックを「そのブロックの実際の大きさ（ノルムの二乗の移動平均の平方根）」で割る。
            # √次元で割るだけでは各次元の値の大きさ（見た目≈1、印は0/1、親は喋らないと0、運動は小さい）が
            # 違うので比率がそろわない（ユーザー指摘、2026-09-09）。実測の大きさで割れば走行中にそろう
            blocks = [obj_state_t, vec_t, parent_t, act_t]
            out = []
            with torch.no_grad():
                for i, b in enumerate(blocks):
                    self._blk_ms[i] = (1.0 - self.block_norm_alpha) * self._blk_ms[i] \
                        + self.block_norm_alpha * float((b.detach() ** 2).sum())
            for i, b in enumerate(blocks):
                out.append(b / torch.sqrt(self._blk_ms[i] + self.block_norm_eps))
            obj_state_t, vec_t, parent_t, act_t = out
        x = torch.cat([obj_state_t, vec_t, parent_t, act_t], dim=-1)
        return x

    def step_for(self, file_id, input_now, target_now):
        """1tickぶん進める（file_idの物の記憶を使う）。

        file_id: Noneなら従来（M7a）の単一状態。多物モードでは物体ファイルの
            id（int）を渡す。初めて見るfile_idなら新しい_ObjStateを作る
            （隠れ状態ゼロ・基準線未確定から始まる＝新しい物を初めて見るのと同じ）。
        input_now・target_now: 従来のstep()と同じ形（クラス冒頭・_build_input参照）。
        戻り値：{"err_state","err_vec","err_parent","err_slow","err_total",
                "baseline","z","z_state"}
                （z_stateはM7b-1で追加。err_stateだけのz値、baseline_state/
                var_state・α=0.01。err_totalのzは従来どおり"z"に残す）
                （float。初回（state.predがまだ無い）はerr系がNone）
        """
        state = self._get_state(file_id)
        self._init_hidden_if_needed(state)
        device = self._device()

        result = {"err_state": None, "err_vec": None, "err_parent": None,
                  "err_slow": None, "err_total": None, "baseline": None,
                  "z": None, "z_state": None}

        if state.pred is not None:
            state_target = torch.tensor(
                list(target_now["obj_state"]), dtype=torch.float32,
                device=device).view(1, -1)
            # present/visible/vanished(先頭3) = BCE(logit)、pos/area(残り3) = MSE。
            bce_part = F.binary_cross_entropy_with_logits(
                state.pred["state_logit"][:, :3], state_target[:, :3])
            mse_part = F.mse_loss(state.pred["state_logit"][:, 3:], state_target[:, 3:])
            err_state = bce_part + mse_part

            vec_target_raw = target_now.get("obj_vec")
            if vec_target_raw is None:
                vec_target = torch.zeros(1, self.obj_vec_dim, device=device)
            else:
                vec_target = torch.tensor(
                    list(vec_target_raw), dtype=torch.float32,
                    device=device).view(1, -1)
            # 頭の予測は「LayerNorm前の生の値」を当てる対象にする（学習対象が
            # 呼び出し側ごとに変わらないよう、内部標準化を予測側の目標には掛けない
            # [Tier3・工学的選択]）。
            err_vec = F.mse_loss(state.pred["vec_pred"], vec_target)

            speak_target = torch.tensor(
                [[float(target_now.get("parent_spoke", 0.0))]],
                dtype=torch.float32, device=device)
            err_speak = F.binary_cross_entropy_with_logits(
                state.pred["speak_logit"], speak_target)
            chunk_id_plus1 = target_now.get("parent_chunk_id_plus1")
            if (target_now.get("parent_spoke", 0.0) and chunk_id_plus1 is not None
                    and 1 <= int(chunk_id_plus1) <= self.n_chunks):
                chunk_target = torch.tensor(
                    [int(chunk_id_plus1) - 1], dtype=torch.long, device=device)
                err_chunk = F.cross_entropy(state.pred["chunk_logit"], chunk_target)
            else:
                err_chunk = torch.zeros((), device=device)
            err_parent = err_speak + err_chunk

            err_total = err_state + err_vec + err_parent

            # ---- ログ用の値（学習の更新頻度に関係なく毎tick出す）------------------
            err_state_v = float(err_state.item())
            err_total_v = float(err_total.item())
            result["err_state"] = err_state_v
            result["err_vec"] = float(err_vec.item())
            result["err_parent"] = float(err_parent.item())
            result["err_total"] = err_total_v

            # ---- 基準線（慣れの確認用。学習には使わない）------------------------
            if state.baseline is None:
                state.baseline = err_total_v
                state.var_ema = 0.0
            else:
                state.baseline = 0.99 * state.baseline + 0.01 * err_total_v
                dev2 = (err_total_v - state.baseline) ** 2
                state.var_ema = 0.99 * state.var_ema + 0.01 * dev2
            std_ema = state.var_ema ** 0.5 if state.var_ema is not None else 0.0
            result["baseline"] = state.baseline
            result["z"] = (err_total_v - state.baseline) / (std_ema + 1e-6)

            # ---- err_state だけの基準線（M7b-1・物ごとの驚き）--------------------
            # 【なぜerr_totalと別に持つか】仕様「決めたこと2」：親の発話の誤差
            # （err_parentに含まれる）は驚きに含めない（当てられないものなので、
            # 砂嵐に釘付けになる問題を避ける）。err_stateだけを見ることで、
            # 「その物の状態（見えた/消えた/位置/大きさ）」の跳ねだけを拾う。
            if state.baseline_state is None:
                state.baseline_state = err_state_v
                state.var_state = 0.0
            else:
                state.baseline_state = 0.99 * state.baseline_state + 0.01 * err_state_v
                dev2_state = (err_state_v - state.baseline_state) ** 2
                state.var_state = 0.99 * state.var_state + 0.01 * dev2_state
            std_state = state.var_state ** 0.5 if state.var_state is not None else 0.0
            result["z_state"] = (err_state_v - state.baseline_state) / (std_state + 1e-6)

            # ---- tbptt窓へ累積（この時点ではまだbackward/optimizer.stepしない）----
            state.loss_accum = (err_total if state.loss_accum is None
                                 else state.loss_accum + err_total)
            state.accum_count += 1
            if state.accum_count >= self.tbptt:
                self.optimizer.zero_grad()
                state.loss_accum.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()
                # 窓の境界＝ここでhiddenをdetachし、累積をリセットする
                # （次の窓の逆伝播がこの窓の外へ伸びないようにする）。
                state.h_fast = state.h_fast.detach()
                state.h_slow = state.h_slow.detach()
                state.loss_accum = None
                state.accum_count = 0

        # ---- 次tickの予測を作る（x_nowを組み立てて流す）------------------------
        # 【なぜここで組み立てるか】上のflush（backward+optimizer.step）が
        # 終わった"後"でないと、x_nowに埋め込まれる学習対象パラメータ
        # （obj_vec_norm・chunk_embedding）の計算グラフが、更新前の重みの
        # まま次の窓に紛れ込みin-placeエラーになる（_build_inputのdocstring参照）。
        x_now = self._build_input(
            input_now["obj_state"], input_now.get("obj_vec"),
            input_now.get("parent_spoke", 0.0), input_now.get("chunk_id_plus1", 0),
            input_now.get("time_since_parent", 1.0), input_now.get("act"))
        pred, h_fast_new, h_slow_new = self._forward_tick(state, x_now)

        # ---- 遅い層の予測誤差（追記2026-09-08）----------------------------------
        # ctx_pred（state.h_slow=前tickまでの文脈から作った予測）と、今tick
        # 実際に得られたh_fast_new（詳細は_forward_tickのdocstring参照）を比べる。
        # target側をdetach()し、速い層(h_fast_new)を歪めないよう勾配は
        # head_ctx・遅い層側にだけ流す。err_totalには含めない（速い段の誤差と
        # 分けて見る、仕様の指示どおり）。state.predがNoneの初回tick(k=0)でも
        # 計算できる（ctx_predは前tickの予測ではなく同tick内の自己完結した値の
        # ため、他の頭のような「初回はNone」の制約が無い）。
        err_slow = F.mse_loss(pred["ctx_pred"], h_fast_new.detach())
        result["err_slow"] = float(err_slow.item())
        state.loss_accum = (err_slow if state.loss_accum is None
                             else state.loss_accum + err_slow)

        state.pred = pred
        state.h_fast = h_fast_new
        state.h_slow = h_slow_new

        state.tick_count += 1
        return result

    def step(self, input_now, target_now):
        """従来（M7a・単一状態）のstep()。`step_for(None, input_now, target_now)`の別名
        （既定不変：multi_objectを使わない呼び出し側の挙動は1ビットも変わらない）。"""
        return self.step_for(None, input_now, target_now)

    def state_dict(self, *args, **kwargs):  # noqa: D401
        """モジュールの重みだけを保存する（隠れ状態・baselineは含まない）。"""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """塊名簿の大きさが保存時と違っても読めるようにする（2026-09-08、F2-89b の失敗を受けて）。

        学習中に塊名簿が伸びる（F2-89 では 18→20）ので、保存した chunk_embedding / head_chunk の行数と
        今の名簿の大きさが食い違う。脳本体の resize_embedding と同じ「先頭コピー＋残りは新規初期化」で合わせる。
        """
        sd = dict(state_dict)
        own = self.state_dict()
        if "_blk_ms" not in sd:            # block_norm 導入前（F2-89/90）の保存物も読める
            sd["_blk_ms"] = own["_blk_ms"]
        for k in ("chunk_embedding.weight", "head_chunk.weight", "head_chunk.bias"):
            if k in sd and k in own and tuple(sd[k].shape) != tuple(own[k].shape):
                new = own[k].clone()
                n = min(int(sd[k].shape[0]), int(new.shape[0]))
                new[:n] = sd[k][:n].to(new.device, new.dtype)
                sd[k] = new
        return super().load_state_dict(sd, *args, **kwargs)


# =============================================================================
# 【M7b-1改・2026-09-09】ポート型の世界の予測器（感覚ごとの速い層＋共有の遅い層）。
#
# 【出典】仕様_M7b-1改_ポート型の世界の予測器_2026-09-09.md「後半：実装担当向け
# 技術付録」1節。前の仕様（仕様_M7b-1_物ごとの予測器と驚きの配線_2026-09-09.md）が
# 導入した WorldPredictor.multi_object（物ごとに全部＝場面の記憶まで複製する）が、
# F2-92/F2-94 の実測（見た目の誤差 0.006→0.722、物の状態 0.153→0.586）で
# 「場面の記憶まで物ごとに空から作り直す」問題を起こしたのを受け、遅い層（＝場面の
# 記憶）だけ1本共有し、速い層（＝各感覚の直近の記憶）を「ポート」として複数
# 差し込む構造に変えた。視覚ポートだけ物ごとに複製（重み共有・記憶は物ごと）、
# 聴覚・体ポートは常に1つ。
#
# 【WorldPredictor（既存）とのすみ分け】既存クラスは無改修。ports=True の実験
# だけが下のPortWorldPredictorを使う（run/taro_setup.pyがcfg.world_predictor.ports
# で分岐。既定False＝既存WorldPredictorのまま、1ビットも変わらない）。
# =============================================================================


class _PortSpec:
    """ポートの規格（手で決める側。中身の学習はPortWorldPredictor側に任せる）。

    仕様「決めたこと」4節：「手で決めるのは『どの感覚のポートがあるか』
    『ポートの規格（まとめの大きさ・時定数）』だけ」。in_dimはポートの入力の
    連結後の次元（block_normのブロック分けの単位でもある）。head_namesは
    このポートが持つ頭（PortWorldPredictor.headsのキー）の一覧（ドキュメント・
    ログ用途。計算そのものはPortWorldPredictor側が直接headsを引いて行う）。
    """
    __slots__ = ("kind", "in_dim", "head_names")

    def __init__(self, kind, in_dim, head_names):
        self.kind = kind
        self.in_dim = int(in_dim)
        self.head_names = list(head_names)


class _PortMem:
    """1ポートぶんの「記憶」（学習される重みではない）。

    速い層の隠れ状態hと直前tickに作った予測predを持つ。視覚ポート（物ごとに
    複製）だけ、err_total・err_stateそれぞれのEMA基準線（baseline/var_ema・
    baseline_state/var_state）も持つ（仕様「後半」1節「各ポート（視覚は物ごと）の
    前tickの予測とbaseline_state/var_state（視覚のみ、z_state用）」）。

    【実装判断・2026-09-09、仕様に無かった点】baseline/var_ema（err_total用）は
    仕様本文が明示していないが、「戻り値：…baseline/z（err_total、従来どおり）」
    という記述があり、"従来"＝M7b-1の多物モード（_ObjState.baseline、物ごとに
    err_total=err_state+err_vec+err_parentのEMAを追う）を指すと解釈した。
    ポート型では聴覚（err_parent）が視覚と別ポートになったため、各視覚ファイルの
    err_totalは「そのファイルのerr_state+err_vec + そのtick共通のerr_parent」で
    計算し、ファイルごとに別々のEMAを追う（M7b-1の多物モードと同じ形を踏襲）。
    """
    __slots__ = ("h", "pred", "baseline", "var_ema", "baseline_state", "var_state")

    def __init__(self):
        self.h = None
        self.pred = None
        self.baseline = None
        self.var_ema = None
        self.baseline_state = None
        self.var_state = None


class PortWorldPredictor(nn.Module):
    """速い層＝ポート（視覚・聴覚・体）、遅い層＝1つ共有、のMTRNN型世界予測器。

    仕様_M7b-1改_ポート型の世界の予測器_2026-09-09.md「後半」1節。

    【構造】
      - 視覚ポート：物ごとに複製（重みは共有＝self.gru["vision"]等は1組、
        隠れ状態self._vision_statesだけfile_idごと）。入力＝[obj_state(6),
        LayerNorm(obj_vec(384))]。頭＝head_state(6)・head_vec(384)。
      - 聴覚ポート：常に1つ。入力＝[spoke(1), Embedding(chunk)(chunk_emb),
        time_since(1)]。頭＝head_speak(1)・head_chunk(n_chunks)。
      - 体ポート：常に1つ。入力＝[act(act_dim)]。頭＝head_act(act_dim)。
      - 遅い層：1つ共有（gru_slow）。入力＝全ポートのsummary（summary_dim次元に
        揃えた各ポートのh）の和を√(ポート数)で割ったもの。head_ctxが次tickの
        「和のまとめ」を当てる（detach）→err_slow。

    【1tickの手順（仕様「後半」1節）】呼び出し側（run/trainer.py）は
    observe_all(...) → predict_all(...) の2回だけ呼ぶ（内部で三段：
    ①全ポートのobserve+1回のbackward+Adam step、②全ポートのpredict）。
    tbptt=1固定（毎tick、flush直後に全hiddenをdetach）。WorldPredictor.observe_for
    /flush_pending/predict_for（多物モード）と同じ理由（1tick内で複数の"物"が
    同じ重みを共有しながら独立にflushするとin-placeエラーになる、
    WorldPredictor._multi_loss_accumのdocstring参照）で、1tickにつき
    ちょうど1回だけbackward+optimizer.stepする二段構えにしている。
    """

    def __init__(self, n_chunks, act_dim, obj_vec_dim=384, h_port=64, h_slow=32,
                 summary_dim=32, tau_fast=5, tau_slow=40, chunk_emb=16, lr=1e-3,
                 grad_clip=1.0, max_files=4):
        super().__init__()
        self.obj_vec_dim = int(obj_vec_dim)
        self.n_chunks = int(n_chunks)
        self.act_dim = int(act_dim)
        self.h_port = int(h_port)
        self.h_slow_dim = int(h_slow)
        self.summary_dim = int(summary_dim)
        self.tau_fast = float(tau_fast)
        self.tau_slow = float(tau_slow)
        self.chunk_emb_dim = int(chunk_emb)
        self.grad_clip = float(grad_clip)
        self.max_files = int(max_files)

        vision_in = 6 + self.obj_vec_dim
        hearing_in = 1 + self.chunk_emb_dim + 1
        body_in = self.act_dim
        self.specs = {
            "vision": _PortSpec("vision", vision_in, ["head_state", "head_vec"]),
            "hearing": _PortSpec("hearing", hearing_in, ["head_speak", "head_chunk"]),
            "body": _PortSpec("body", body_in, ["head_act"]),
        }

        # ---- 入力の組み立て（視覚・聴覚で共有する部品）--------------------------
        self.obj_vec_norm = nn.LayerNorm(self.obj_vec_dim)
        self.chunk_embedding = nn.Embedding(
            self.n_chunks + 1, self.chunk_emb_dim, padding_idx=0)

        # ---- ポートの重み（種類ごとに1組）---------------------------------------
        self.gru = nn.ModuleDict({
            kind: nn.GRUCell(spec.in_dim + self.h_slow_dim, self.h_port)
            for kind, spec in self.specs.items()
        })
        self.summary = nn.ModuleDict({
            kind: nn.Linear(self.h_port, self.summary_dim) for kind in self.specs
        })
        self.heads = nn.ModuleDict({
            "head_state": nn.Linear(self.h_port, 6),
            "head_vec": nn.Linear(self.h_port, self.obj_vec_dim),
            "head_speak": nn.Linear(self.h_port, 1),
            "head_chunk": nn.Linear(self.h_port, self.n_chunks),
            "head_act": nn.Linear(self.h_port, self.act_dim),
        })

        # ---- 遅い層（1つ共有）----------------------------------------------------
        self.gru_slow = nn.GRUCell(self.summary_dim, self.h_slow_dim)
        self.head_ctx = nn.Linear(self.h_slow_dim, self.summary_dim)

        # 【block_norm・実装判断、仕様に無かった点】既存WorldPredictorのblock_norm
        #   （4ブロック：状態6/見た目384/親18/運動act_dim、既定Falseで切替可能）と
        #   同じ式・同じ4分割（視覚state・視覚vec・聴覚・体）をここでは常時オンにした。
        #   ポート型の実験（F2-96）は最初からblock_norm:trueで設計されており
        #   （仕様「後半」6節のconfig）、切るモードを別途持つ意味が薄いため
        #   固定にした[Tier3・工学的選択]。
        self.register_buffer("_blk_ms", torch.ones(4))
        self.block_norm_alpha = 0.01
        self.block_norm_eps = 1e-3

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # ---- 記憶（学習される重みではない）--------------------------------------
        self._vision_states = {}          # file_id -> _PortMem
        self._hearing_state = _PortMem()
        self._body_state = _PortMem()
        self._h_slow = None                # 遅い層の隠れ状態(1,h_slow_dim)。前tick分。

        # 1tickの中で「観測時に積んだ誤差＋前tickのpredictが積んだerr_slow」を
        # まとめて1回だけbackward+stepするための共有累積（既存WorldPredictorの
        # _multi_loss_accumと同じ役目）。
        self._pending_loss = None

    # ------------------------------------------------------------------
    def _device(self):
        return next(self.parameters()).device

    def drop(self, file_id):
        """その物の視覚ポートの記憶を捨てる（重みには触らない）。"""
        self._vision_states.pop(file_id, None)

    def active_ids(self):
        """今、視覚ポートの記憶を持っている物のidの一覧。"""
        return list(self._vision_states.keys())

    def reset_hidden(self):
        """全ポート・遅い層の隠れ状態と直前の予測をリセットする（エピソード境界などで呼ぶ）。

        重み・optimizer・block_normの統計（_blk_ms）は変えない。
        """
        self._vision_states = {}
        self._hearing_state = _PortMem()
        self._body_state = _PortMem()
        self._h_slow = None
        self._pending_loss = None

    def _get_vision(self, file_id):
        st = self._vision_states.get(file_id)
        if st is None:
            st = _PortMem()
            self._vision_states[file_id] = st
        return st

    def _init_slow_if_needed(self):
        if self._h_slow is None:
            self._h_slow = torch.zeros(1, self.h_slow_dim, device=self._device())

    # ------------------------------------------------------------------
    # 入力の組み立て（ブロック正規化つき。既存WorldPredictor._build_inputの
    # block_norm節と同じ式：各ブロックを実測の大きさ（||x||²のEMAの平方根）で割る）。
    def _block_norm_apply(self, blocks_and_idx):
        out = []
        with torch.no_grad():
            for b, i in blocks_and_idx:
                self._blk_ms[i] = ((1.0 - self.block_norm_alpha) * self._blk_ms[i]
                                    + self.block_norm_alpha * float((b.detach() ** 2).sum()))
        for b, i in blocks_and_idx:
            out.append(b / torch.sqrt(self._blk_ms[i] + self.block_norm_eps))
        return out

    def _build_vision_x(self, obj_state, obj_vec):
        """block_ms index: 0=state(6), 1=vec(384)。"""
        device = self._device()
        state_t = torch.tensor(list(obj_state), dtype=torch.float32,
                                device=device).view(1, -1)
        if state_t.shape[-1] != 6:
            raise ValueError(f"obj_state は6次元のはず（実際={state_t.shape[-1]}）")
        if obj_vec is None:
            vec_t = torch.zeros(1, self.obj_vec_dim, device=device)
        else:
            vec_t = torch.tensor(list(obj_vec), dtype=torch.float32,
                                  device=device).view(1, -1)
            if vec_t.shape[-1] != self.obj_vec_dim:
                raise ValueError(
                    f"obj_vecの次元が構築時の想定({self.obj_vec_dim})と違う "
                    f"（実際={vec_t.shape[-1]}）")
        vec_t = self.obj_vec_norm(vec_t)
        state_t, vec_t = self._block_norm_apply([(state_t, 0), (vec_t, 1)])
        return torch.cat([state_t, vec_t], dim=-1)

    def _build_hearing_x(self, parent_spoke, chunk_id_plus1, time_since_parent):
        """block_ms index: 2=聴覚（spoke+chunk_emb+timeを1ブロックとして正規化）。"""
        device = self._device()
        idx = max(0, min(int(chunk_id_plus1), self.n_chunks))
        chunk_vec = self.chunk_embedding(
            torch.tensor([idx], dtype=torch.long, device=device))
        parent_t = torch.cat([
            torch.tensor([[float(parent_spoke)]], dtype=torch.float32, device=device),
            chunk_vec,
            torch.tensor([[float(time_since_parent)]], dtype=torch.float32, device=device),
        ], dim=-1)
        (parent_t,) = self._block_norm_apply([(parent_t, 2)])
        return parent_t

    def _build_body_x(self, act):
        """block_ms index: 3=体。"""
        device = self._device()
        if act is None:
            act_t = torch.zeros(1, self.act_dim, device=device)
        else:
            act_t = torch.tensor(list(act), dtype=torch.float32,
                                  device=device).view(1, -1)
            if act_t.shape[-1] != self.act_dim:
                raise ValueError(
                    f"actの次元が構築時の想定({self.act_dim})と違う "
                    f"（実際={act_t.shape[-1]}）")
        (act_t,) = self._block_norm_apply([(act_t, 3)])
        return act_t

    def _fast_step(self, kind, h_prev, x):
        """1ポートぶんの速い層を1tick進める（漏れ積分GRU）。h_prevがNoneなら零から。"""
        if h_prev is None:
            h_prev = torch.zeros(1, self.h_port, device=self._device())
        f_in = torch.cat([x, self._h_slow], dim=-1)
        h_gru = self.gru[kind](f_in, h_prev)
        h_new = ((1.0 - 1.0 / self.tau_fast) * h_prev
                  + (1.0 / self.tau_fast) * h_gru)
        return h_new

    # ------------------------------------------------------------------
    def observe_all(self, vision_targets, hearing_target, body_target):
        """1tickの①：全ポートのobserve（前tickの予測と今の観測を比べる）＋
        1回のbackward+Adam step（仕様「後半」1節①）。

        vision_targets: {file_id: {"obj_state":[6の列], "obj_vec":[384の列] or None}}
        hearing_target: {"parent_spoke":0.0/1.0, "parent_chunk_id_plus1": int or None}
        body_target: {"act": [act_dimの列] or None}

        戻り値：{"by_file": {file_id: {"err_state","err_vec","z_state",
                "err_total","baseline","z"}}, "err_parent", "err_body"}
                （まだ誰も予測していない物・聴覚・体は初回のみNone。
                err_slowはこの中には無い＝predict_allが今tickぶんを返す）
        """
        device = self._device()
        loss_terms = []

        # ---- 聴覚（親の発話。全ポート共通の対象）--------------------------------
        err_parent_v = None
        if self._hearing_state.pred is not None:
            speak_target = torch.tensor(
                [[float(hearing_target.get("parent_spoke", 0.0))]],
                dtype=torch.float32, device=device)
            err_speak = F.binary_cross_entropy_with_logits(
                self._hearing_state.pred["speak_logit"], speak_target)
            chunk_id_plus1 = hearing_target.get("parent_chunk_id_plus1")
            if (hearing_target.get("parent_spoke", 0.0) and chunk_id_plus1 is not None
                    and 1 <= int(chunk_id_plus1) <= self.n_chunks):
                chunk_target = torch.tensor(
                    [int(chunk_id_plus1) - 1], dtype=torch.long, device=device)
                err_chunk = F.cross_entropy(
                    self._hearing_state.pred["chunk_logit"], chunk_target)
            else:
                err_chunk = torch.zeros((), device=device)
            err_parent = err_speak + err_chunk
            err_parent_v = float(err_parent.item())
            loss_terms.append(err_parent)

        # ---- 体（自分の運動）----------------------------------------------------
        err_body_v = None
        if self._body_state.pred is not None:
            act_target_raw = body_target.get("act")
            if act_target_raw is None:
                act_target = torch.zeros(1, self.act_dim, device=device)
            else:
                act_target = torch.tensor(list(act_target_raw), dtype=torch.float32,
                                           device=device).view(1, -1)
            err_body = F.mse_loss(self._body_state.pred["act_pred"], act_target)
            err_body_v = float(err_body.item())
            loss_terms.append(err_body)

        # ---- 視覚（物ごと）-------------------------------------------------------
        by_file = {}
        for file_id, target in vision_targets.items():
            st = self._get_vision(file_id)
            if st.pred is None:
                by_file[file_id] = {"err_state": None, "err_vec": None,
                                     "z_state": None, "err_total": None,
                                     "baseline": None, "z": None}
                continue
            state_target = torch.tensor(list(target["obj_state"]), dtype=torch.float32,
                                         device=device).view(1, -1)
            bce_part = F.binary_cross_entropy_with_logits(
                st.pred["state_logit"][:, :3], state_target[:, :3])
            mse_part = F.mse_loss(st.pred["state_logit"][:, 3:], state_target[:, 3:])
            err_state = bce_part + mse_part

            vec_target_raw = target.get("obj_vec")
            if vec_target_raw is None:
                vec_target = torch.zeros(1, self.obj_vec_dim, device=device)
            else:
                vec_target = torch.tensor(list(vec_target_raw), dtype=torch.float32,
                                           device=device).view(1, -1)
            err_vec = F.mse_loss(st.pred["vec_pred"], vec_target)

            err_state_v = float(err_state.item())
            err_vec_v = float(err_vec.item())
            loss_terms.append(err_state + err_vec)

            # ---- err_total/baseline/z（従来どおり。err_parentはこのtick共通の値）----
            err_total_v = err_state_v + err_vec_v + (err_parent_v or 0.0)
            if st.baseline is None:
                st.baseline = err_total_v
                st.var_ema = 0.0
            else:
                st.baseline = 0.99 * st.baseline + 0.01 * err_total_v
                dev2 = (err_total_v - st.baseline) ** 2
                st.var_ema = 0.99 * st.var_ema + 0.01 * dev2
            std_ema = st.var_ema ** 0.5 if st.var_ema is not None else 0.0
            baseline_v = st.baseline
            z_v = (err_total_v - st.baseline) / (std_ema + 1e-6)

            # ---- err_state だけの基準線（驚き→青斑核の配線用、M7b-1と同じ）---------
            if st.baseline_state is None:
                st.baseline_state = err_state_v
                st.var_state = 0.0
            else:
                st.baseline_state = 0.99 * st.baseline_state + 0.01 * err_state_v
                dev2_state = (err_state_v - st.baseline_state) ** 2
                st.var_state = 0.99 * st.var_state + 0.01 * dev2_state
            std_state = st.var_state ** 0.5 if st.var_state is not None else 0.0
            z_state_v = (err_state_v - st.baseline_state) / (std_state + 1e-6)

            by_file[file_id] = {"err_state": err_state_v, "err_vec": err_vec_v,
                                 "z_state": z_state_v, "err_total": err_total_v,
                                 "baseline": baseline_v, "z": z_v}

        # ---- flush（①の後半：このtickぶん＋前tickのpredict_allが積んだerr_slowを
        #      まとめて1回だけbackward+Adam step）------------------------------
        total_loss = None
        for term in loss_terms:
            total_loss = term if total_loss is None else total_loss + term
        if self._pending_loss is not None:
            total_loss = (self._pending_loss if total_loss is None
                           else total_loss + self._pending_loss)
        if total_loss is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
            self.optimizer.step()
            # 窓の境界（tbptt=1固定）＝ここで全hiddenをdetach（次tickへ計算グラフを持ち越さない）。
            for st in self._vision_states.values():
                if st.h is not None:
                    st.h = st.h.detach()
            if self._hearing_state.h is not None:
                self._hearing_state.h = self._hearing_state.h.detach()
            if self._body_state.h is not None:
                self._body_state.h = self._body_state.h.detach()
            if self._h_slow is not None:
                self._h_slow = self._h_slow.detach()
        self._pending_loss = None

        return {"by_file": by_file, "err_parent": err_parent_v, "err_body": err_body_v}

    def predict_all(self, vision_inputs, hearing_input, body_input):
        """1tickの②：全ポートのpredict（今の入力で状態を更新し次tickの予測を作る）。

        flush（observe_allの①後半）の"後"に呼ぶこと（更新済みの重みで次tickの
        予測を作る）。遅い層も更新し、今tickぶんのerr_slowを次のobserve_all呼び出し
        （＝次tickのflush）に積む。

        vision_inputs: {file_id: {"obj_state":[6], "obj_vec": [...] or None}}
        hearing_input: {"parent_spoke":..., "chunk_id_plus1":..., "time_since_parent":...}
        body_input: {"act": [...] or None}
        戻り値：{"err_slow": float}
        """
        self._init_slow_if_needed()
        h_slow_prev = self._h_slow

        summaries = []

        # ---- 視覚（物ごと）-------------------------------------------------------
        for file_id, inp in vision_inputs.items():
            st = self._get_vision(file_id)
            x = self._build_vision_x(inp["obj_state"], inp.get("obj_vec"))
            h_new = self._fast_step("vision", st.h, x)
            st.pred = {
                "state_logit": self.heads["head_state"](h_new),
                "vec_pred": self.heads["head_vec"](h_new),
            }
            st.h = h_new
            summaries.append(self.summary["vision"](h_new))

        # ---- 聴覚 -----------------------------------------------------------
        x_h = self._build_hearing_x(hearing_input.get("parent_spoke", 0.0),
                                     hearing_input.get("chunk_id_plus1", 0),
                                     hearing_input.get("time_since_parent", 1.0))
        h_hearing_new = self._fast_step("hearing", self._hearing_state.h, x_h)
        self._hearing_state.pred = {
            "speak_logit": self.heads["head_speak"](h_hearing_new),
            "chunk_logit": self.heads["head_chunk"](h_hearing_new),
        }
        self._hearing_state.h = h_hearing_new
        summaries.append(self.summary["hearing"](h_hearing_new))

        # ---- 体 -------------------------------------------------------------
        x_b = self._build_body_x(body_input.get("act"))
        h_body_new = self._fast_step("body", self._body_state.h, x_b)
        self._body_state.pred = {"act_pred": self.heads["head_act"](h_body_new)}
        self._body_state.h = h_body_new
        summaries.append(self.summary["body"](h_body_new))

        # ---- 遅い層（1つ共有。全ポートのsummaryの和を√ポート数で割る）------------
        n_ports = len(summaries)
        agg = sum(summaries) / (float(n_ports) ** 0.5)
        h_slow_gru = self.gru_slow(agg, h_slow_prev)
        h_slow_new = ((1.0 - 1.0 / self.tau_slow) * h_slow_prev
                       + (1.0 / self.tau_slow) * h_slow_gru)

        # 【追記2026-09-09・既存WorldPredictor._forward_tickのTier3判断を踏襲】
        #   head_ctx(h_slow_prev)＝前tickまでの文脈から作った予測を、今tick
        #   計算されたagg（今tickの入力を取り込んだ後の全ポートのまとめ、detach）
        #   と比べる。予測と答え合わせが同一tick内で完結するため、
        #   複数tickにまたがるbackwardのin-place事故を避けられる。
        ctx_pred = self.head_ctx(h_slow_prev)
        err_slow = F.mse_loss(ctx_pred, agg.detach())

        self._h_slow = h_slow_new
        self._pending_loss = (err_slow if self._pending_loss is None
                                else self._pending_loss + err_slow)

        return {"err_slow": float(err_slow.item())}

    def state_dict(self, *args, **kwargs):  # noqa: D401
        """モジュールの重みだけを保存する（隠れ状態・baselineは含まない）。"""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """塊名簿の大きさが保存時と違っても読めるようにする（既存WorldPredictorと同じ流儀）。"""
        sd = dict(state_dict)
        own = self.state_dict()
        if "_blk_ms" not in sd:
            sd["_blk_ms"] = own["_blk_ms"]
        for k in ("chunk_embedding.weight", "heads.head_chunk.weight", "heads.head_chunk.bias"):
            if k in sd and k in own and tuple(sd[k].shape) != tuple(own[k].shape):
                new = own[k].clone()
                n = min(int(sd[k].shape[0]), int(new.shape[0]))
                new[:n] = sd[k][:n].to(new.device, new.dtype)
                sd[k] = new
        return super().load_state_dict(sd, *args, **kwargs)
