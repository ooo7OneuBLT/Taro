"""測定器（probe類）— 太郎を外から観察するための計測装置。学習ループ本体からは独立。

【なぜ別ファイルか・2026-07-23】これらは太郎の脳・身体・環境ではなく、**研究者が太郎を
外から測るための道具**（人間の赤ちゃんには存在しない観測装置）。学習ループ
(e_growth_train.py)と混ざっていると「どこが太郎でどこが計測器か」が見分けにくくなるため
分離した（ユーザー指摘）。中身は C/scripts/run_c_metrics_ac_lr.py から**挙動を変えずに**
切り出したもの。

すべての関数は ProbeContext（太郎モデル・環境・モデル操作ヘルパー・設定を束ねたもの）を
受け取る。measurement は state を歩かせる副作用を持つ（元コードと同一）。

含まれるもの：
  evaluate           … 自己モデルの識別成績（classify/margin/corr/persist）
  agency_probe       … 自己主体感（自分の意図どおりか vs 外部指令か）
  inverse_probe      … 逆モデルStage1診断（順モデルを反転して行動を推論できるか）
  inverse_exec_probe … 逆モデルStage1.5（推論した行動を実際に実行して到達を測る）
  closed_loop_probe  … C4診断（閉ループ制御 vs 開ループの到達比較）
  record_video       … 学習後の太郎を等速で録画（指標が退化を高評価する罠への対抗＝目視）
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

mse = F.mse_loss


class ProbeContext:
    """太郎モデル・環境・モデル操作ヘルパー・設定を束ねて probe へ渡す入れ物。

    ヘルパー（zc/act_mean/step_k/ln_prop/reset_state/infer_goal_action）は学習ループと
    測定器の両方が使う「モデル/環境インターフェース」で、e_growth_train.py 側で定義された
    クロージャの参照をそのまま持つ（＝挙動は元と完全に同一）。
    """

    def __init__(self, *, brain, fusion, target_fusion, nat_head, env, rescale_action,
                 zc, act_mean, step_k, ln_prop, reset_state, infer_goal_action,
                 state, n_act, n_eval, K, DT, seed, log_dir):
        self.brain = brain
        self.fusion = fusion
        self.target_fusion = target_fusion
        self.nat_head = nat_head
        self.env = env
        self.rescale_action = rescale_action
        self.zc = zc
        self.act_mean = act_mean
        self.step_k = step_k
        self.ln_prop = ln_prop
        self.reset_state = reset_state
        self.infer_goal_action = infer_goal_action
        self.state = state
        self.n_act = n_act
        self.n_eval = n_eval
        self.K = K
        self.DT = DT
        self.seed = seed
        self.log_dir = log_dir


def evaluate(ctx):
    fusion, target_fusion, ln_prop = ctx.fusion, ctx.target_fusion, ctx.ln_prop
    zc, act_mean, nat_head = ctx.zc, ctx.act_mean, ctx.nat_head
    step_k, rescale_action, env = ctx.step_k, ctx.rescale_action, ctx.env
    reset_state, state = ctx.reset_state, ctx.state
    Zs, acts, nx, cu, self_err, ep = [], [], [], [], [], []
    pdel, adel = [], []
    for _ in range(ctx.n_eval):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
        z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
        a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        pd = nat_head(torch.cat([z, a], dim=-1)).detach()
        state["obs"], term = step_k(rescale_action(a, env.action_space))
        nlp = ln_prop(state["obs"])
        self_err.append(mse(clp + pd, nlp).item()); ep.append(mse(clp, nlp).item())
        pdel.append(pd.numpy()); adel.append((nlp - clp).numpy())
        Zs.append(z); acts.append(a); nx.append(nlp); cu.append(clp)
        state["hidden"] = hn.detach(); state["prev_a"] = a
        if term:
            reset_state()
    N = len(Zs); correct = total = 0; other_errs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            eo = mse((cu[i] + nat_head(torch.cat([Zs[i], acts[j]], dim=-1))).detach(), nx[i]).item()
            other_errs.append(eo)
            if self_err[i] < eo:
                correct += 1
            total += 1
    classify = correct / total * 100
    margin = (np.mean(other_errs) - np.mean(self_err)) / np.mean(other_errs) * 100
    persist = np.mean(self_err) / np.mean(ep) * 100
    P = np.concatenate([p.flatten() for p in pdel]); A = np.concatenate([a.flatten() for a in adel])
    corr = float(np.corrcoef(P, A)[0, 1])
    return classify, margin, corr, persist


def agency_probe(ctx, n=30):
    # 遠心性コピー(=太郎の意図a_self)は両トライアル共通。体が自分の意図どおり(自己)か
    # 外部指令(外因)かだけが違う。GRUに渡す前回行動は常に「太郎自身の意図」。
    fusion, target_fusion, ln_prop = ctx.fusion, ctx.target_fusion, ctx.ln_prop
    zc, act_mean, nat_head = ctx.zc, ctx.act_mean, ctx.nat_head
    step_k, rescale_action, env = ctx.step_k, ctx.rescale_action, ctx.env
    reset_state, state = ctx.reset_state, ctx.state
    self_errs, ext_errs, self_mag, ext_mag, self_acts = [], [], [], [], []
    for _ in range(n):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
        z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
        a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        pred = nat_head(torch.cat([z, a], dim=-1)).detach()
        state["obs"], term = step_k(rescale_action(a, env.action_space))
        nlp = ln_prop(state["obs"])
        self_errs.append(mse(clp + pred, nlp).item())
        self_mag.append((nlp - clp).abs().mean().item())
        self_acts.append(a); state["hidden"] = hn.detach(); state["prev_a"] = a
        if term:
            reset_state()
    perm = np.random.permutation(len(self_acts))
    for k in range(n):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach(); clp = ln_prop(state["obs"])
        z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
        a_self = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        a_ext = self_acts[perm[k]]
        pred = nat_head(torch.cat([z, a_self], dim=-1)).detach()
        state["obs"], term = step_k(rescale_action(a_ext, env.action_space))
        nlp = ln_prop(state["obs"])
        ext_errs.append(mse(clp + pred, nlp).item())
        ext_mag.append((nlp - clp).abs().mean().item())
        state["hidden"] = hn.detach(); state["prev_a"] = a_self  # 遠心性コピーは意図
        if term:
            reset_state()
    correct = total = 0
    for se in self_errs:
        for ee in ext_errs:
            correct += int(se < ee); total += 1
    agency = correct / total * 100
    mag_ratio = np.mean(ext_mag) / max(np.mean(self_mag), 1e-9) * 100
    return agency, mag_ratio


def inverse_probe(ctx, n=60, n_steps=40, n_restarts=3, lr_inf=0.1):
    """逆モデルStage1診断：凍結した順モデル(nat_head)を"反転"して、望む次感覚に当てる
    行動を推論できるか。goal＝実際に到達した次固有感覚(到達可能)。
    主眼＝①行動レバレッジ(err_random − err_infer：行動が予測にどれだけ効くか)、
    ②順精度の天井(err_areal)。副次＝復元相関(a*とa_realの一致、§5線形0.22と比較)。"""
    fusion, target_fusion, ln_prop = ctx.fusion, ctx.target_fusion, ctx.ln_prop
    zc, act_mean, nat_head = ctx.zc, ctx.act_mean, ctx.nat_head
    step_k, rescale_action, env = ctx.step_k, ctx.rescale_action, ctx.env
    reset_state, state, n_act, seed, LOG_DIR = ctx.reset_state, ctx.state, ctx.n_act, ctx.seed, ctx.log_dir
    ei, ea, er, astar_all, areal_all = [], [], [], [], []
    for _ in range(n):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
        clp = ln_prop(state["obs"])
        z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
        a_real = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        state["obs"], term = step_k(rescale_action(a_real, env.action_space))
        target = (ln_prop(state["obs"]) - clp).detach()  # 望む変化＝実際の次感覚−今

        def ferr(a):
            return ((nat_head(torch.cat([z, a], dim=-1)) - target) ** 2).mean()

        best_a, best_e = a_real, ferr(a_real).item()
        for r in range(n_restarts):
            raw = (torch.atanh(torch.clamp(a_real, -0.999, 0.999)).clone() if r == 0
                   else torch.empty(n_act).uniform_(-1.0, 1.0)).detach().requires_grad_(True)
            opt = torch.optim.Adam([raw], lr=lr_inf)
            for _ in range(n_steps):
                opt.zero_grad(); ferr(torch.tanh(raw)).backward(); opt.step()
            af = torch.tanh(raw).detach(); ef = ferr(af).item()
            if ef < best_e:
                best_e, best_a = ef, af
        ei.append(best_e); ea.append(ferr(a_real).item())
        er.append(ferr(torch.empty(n_act).uniform_(-1.0, 1.0)).item())
        astar_all.append(best_a.numpy()); areal_all.append(a_real.numpy())
        state["hidden"] = hn.detach(); state["prev_a"] = a_real
        if term:
            reset_state()
    A = np.concatenate(astar_all); B = np.concatenate(areal_all)
    rec = float(np.corrcoef(A, B)[0, 1])
    mi, ma, mr = float(np.mean(ei)), float(np.mean(ea)), float(np.mean(er))
    print(f"[INV seed{seed}] err_infer={mi:.4f} err_areal(順精度天井)={ma:.4f} "
          f"err_random={mr:.4f} | leverage(rand-infer)={mr - mi:.4f} "
          f"infer/random={mi / max(mr, 1e-9):.2f} infer/areal={mi / max(ma, 1e-9):.2f} "
          f"recover_corr(a*,a_real)={rec:.3f}", flush=True)
    with open(os.path.join(LOG_DIR, f"inv_probe_seed{seed}.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"err_infer,{mi}\nerr_areal,{ma}\nerr_random,{mr}\nleverage,{mr - mi}\n"
                 f"infer_over_random,{mi / max(mr, 1e-9)}\ninfer_over_areal,{mi / max(ma, 1e-9)}\n"
                 f"recover_corr,{rec}\n")
    return {"infer_over_random": mi / max(mr, 1e-9), "recover_corr": rec}


def inverse_exec_probe(ctx, n=40, n_steps=40, n_restarts=3, lr_inf=0.1):
    """逆モデルStage1.5＝実行テスト：推論a*を"実際にMIMoで実行"し、現実の次感覚が
    目標に届くか。同じ物理状態(qpos/qvel)から a_real / a* / random を実行して現実の
    到達誤差を比較（MuJoCo state save/restore でカウンターファクト）。
    goal＝基準行動a_realを実行した現実の次感覚。d_real2＝a_real再実行＝決定性の床(≈0期待)。"""
    fusion, target_fusion, ln_prop = ctx.fusion, ctx.target_fusion, ctx.ln_prop
    zc, act_mean, nat_head = ctx.zc, ctx.act_mean, ctx.nat_head
    step_k, rescale_action, env = ctx.step_k, ctx.rescale_action, ctx.env
    reset_state, state, n_act, seed, LOG_DIR = ctx.reset_state, ctx.state, ctx.n_act, ctx.seed, ctx.log_dir
    mj = env.unwrapped
    d_star, d_rand, d_floor, mi_model = [], [], [], []

    def real_rollout(a, qpos, qvel):
        mj.set_state(qpos.copy(), qvel.copy())
        o, _ = step_k(rescale_action(a, env.action_space))
        return ln_prop(o)

    for _ in range(n):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
        clp = ln_prop(state["obs"])
        z, _, _, hn = zc(sv, state["prev_a"], cf, state["hidden"]); z = z.detach()
        a_real = torch.clamp(act_mean(z), -1.0, 1.0).detach()
        qpos = mj.data.qpos.copy(); qvel = mj.data.qvel.copy()
        g = real_rollout(a_real, qpos, qvel)          # 目標＝a_realの現実の結果
        g2 = real_rollout(a_real, qpos, qvel)         # 再実行＝決定性の床
        target = (g - clp).detach()

        def ferr(a):
            return ((nat_head(torch.cat([z, a], dim=-1)) - target) ** 2).mean()

        best_a, best_e = a_real, ferr(a_real).item()
        for r in range(n_restarts):
            raw = (torch.atanh(torch.clamp(a_real, -0.999, 0.999)).clone() if r == 0
                   else torch.empty(n_act).uniform_(-1.0, 1.0)).detach().requires_grad_(True)
            opt = torch.optim.Adam([raw], lr=lr_inf)
            for _ in range(n_steps):
                opt.zero_grad(); ferr(torch.tanh(raw)).backward(); opt.step()
            af = torch.tanh(raw).detach(); ef = ferr(af).item()
            if ef < best_e:
                best_e, best_a = ef, af
        a_rand = torch.empty(n_act).uniform_(-1.0, 1.0)
        o_star = real_rollout(best_a, qpos, qvel)     # 推論a*を実行
        o_rand = real_rollout(a_rand, qpos, qvel)     # ランダムを実行
        d_star.append(mse(o_star, g).item())
        d_rand.append(mse(o_rand, g).item())
        d_floor.append(mse(g2, g).item())
        mi_model.append(best_e)
        # 実トラジェクトリを a_real で1歩進める（他プローブと同様に状態を歩かせる）
        state["obs"], term = step_k(rescale_action(a_real, env.action_space))
        state["hidden"] = hn.detach(); state["prev_a"] = a_real
        if term:
            reset_state()
    ds, dr, df = float(np.mean(d_star)), float(np.mean(d_rand)), float(np.mean(d_floor))
    mm = float(np.mean(mi_model))
    print(f"[INVEXEC seed{seed}] d_star(a*実行)={ds:.4f} d_random={dr:.4f} "
          f"d_floor(a_real再実行)={df:.4f} | star/random={ds / max(dr, 1e-9):.2f} "
          f"star/floor={ds / max(df, 1e-9):.2f} model_err(a*)={mm:.4f}", flush=True)
    with open(os.path.join(LOG_DIR, f"inv_exec_seed{seed}.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"d_star,{ds}\nd_random,{dr}\nd_floor,{df}\n"
                 f"star_over_random,{ds / max(dr, 1e-9)}\nstar_over_floor,{ds / max(df, 1e-9)}\n"
                 f"model_err_star,{mm}\n")
    return {"star_over_random": ds / max(dr, 1e-9), "model_err": mm}


def closed_loop_probe(ctx, goal_buf, n=30, max_reach=10):
    """C4診断：閉ループ制御 vs 開ループ で、目標姿勢へどれだけ届くか。
    目標g＝過去に経験した姿勢(goal_buf)。同じ物理状態(MuJoCo save/restore)から比較。
    補正の刻み k_inner ＝ K（順モデルの予測幅に合わせる。第1版はK//8でミスマッチ→負けた）。
    【リーチ長は固定しない＝根拠づけ】"補正しても目標に近づかなくなったら止める"（能動的
    推論＝誤差が減らせなくなったら停止）。max_reach は暴走防止の安全上限のみ（挙動を決める
    数字ではない）。開ループは閉ループが要した長さと同じで比較（公平）。"""
    fusion, target_fusion, ln_prop = ctx.fusion, ctx.target_fusion, ctx.ln_prop
    zc, act_mean, infer_goal_action = ctx.zc, ctx.act_mean, ctx.infer_goal_action
    step_k, rescale_action, env = ctx.step_k, ctx.rescale_action, ctx.env
    reset_state, state, n_act, seed, LOG_DIR, K = ctx.reset_state, ctx.state, ctx.n_act, ctx.seed, ctx.log_dir, ctx.K
    mj = env.unwrapped
    k_inner = K  # 補正の刻み＝順モデルの予測幅（ミスマッチ解消）
    dc, do, dr, dn, steps = [], [], [], [], []

    def const_rollout(a, nsteps, qpos, qvel):
        mj.set_state(qpos.copy(), qvel.copy())
        ra = rescale_action(a, env.action_space); o = state["obs"]
        for _ in range(nsteps):
            o, _, te, tr, _ = env.step(ra)
            if te or tr:
                break
        return ln_prop(o)

    def closed_rollout(g, z0, clp0, h0, pa0, qpos, qvel):
        mj.set_state(qpos.copy(), qvel.copy())
        z_now, clp_now, hcl, pacl = z0, clp0, h0, pa0
        o = state["obs"]; prev_d = mse(clp_now, g).item(); nstep = 0
        for _ in range(max_reach):
            a = infer_goal_action(z_now, clp_now, torch.clamp(act_mean(z_now), -1.0, 1.0), g)
            ra = rescale_action(a, env.action_space)
            for _ in range(k_inner):
                o, _, te, tr, _ = env.step(ra)
                if te or tr:
                    break
            nstep += 1
            clp_now = ln_prop(o)  # ← 実際に観測した状態から補正（閉ループの"閉じ"）
            d = mse(clp_now, g).item()
            if d >= prev_d:  # これ以上近づけない→停止（誤差最小化の自然な停止条件＝根拠）
                break
            prev_d = d
            sv_now = fusion.encode(o); cf_now = target_fusion.encode(o).detach()
            zt, _, _, hn = zc(sv_now, pacl, cf_now, hcl)
            z_now = zt.detach(); hcl = hn.detach(); pacl = a
        return ln_prop(o), nstep

    for _ in range(n):
        sv = fusion.encode(state["obs"]); cf = target_fusion.encode(state["obs"]).detach()
        clp0 = ln_prop(state["obs"])
        z0, _, _, hn0 = zc(sv, state["prev_a"], cf, state["hidden"]); z0 = z0.detach()
        g = goal_buf[torch.randint(len(goal_buf), (1,)).item()]
        qpos = mj.data.qpos.copy(); qvel = mj.data.qvel.copy()
        o_closed, nstep = closed_rollout(g, z0, clp0, state["hidden"], state["prev_a"], qpos, qvel)
        a_open = infer_goal_action(z0, clp0, torch.clamp(act_mean(z0), -1.0, 1.0), g)
        o_open = const_rollout(a_open, max(1, nstep) * k_inner, qpos, qvel)  # 閉ループと同じ長さで公平比較
        o_rand = const_rollout(torch.empty(n_act).uniform_(-1.0, 1.0), max(1, nstep) * k_inner, qpos, qvel)
        do.append(mse(o_open, g).item()); dc.append(mse(o_closed, g).item())
        dr.append(mse(o_rand, g).item()); dn.append(mse(clp0, g).item()); steps.append(nstep)
        mj.set_state(qpos, qvel)  # 実トラジェクトリを1リーチぶん進める
        state["obs"], term = step_k(rescale_action(a_open, env.action_space))
        state["hidden"] = hn0.detach(); state["prev_a"] = a_open
        if term:
            reset_state()
    mc, mo, mr, mn, ms = (float(np.mean(dc)), float(np.mean(do)), float(np.mean(dr)),
                          float(np.mean(dn)), float(np.mean(steps)))
    print(f"[CLOSEDLOOP seed{seed}] closed={mc:.4f} open={mo:.4f} random={mr:.4f} "
          f"nothing={mn:.4f} | closed/open={mc / max(mo, 1e-9):.2f} "
          f"closed/nothing={mc / max(mn, 1e-9):.2f} reach_len_avg={ms:.1f}", flush=True)
    with open(os.path.join(LOG_DIR, f"closed_loop_seed{seed}.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"closed,{mc}\nopen,{mo}\nrandom,{mr}\nnothing,{mn}\nreach_len_avg,{ms}\n"
                 f"closed_over_open,{mc / max(mo, 1e-9)}\nclosed_over_nothing,{mc / max(mn, 1e-9)}\n")


def record_video(ctx, mp4_path, make_render_env):
    """学習後の太郎を等速で録画する。

    【必須級・2026-07-15】太郎の指標は「予測がうまいか」を測るが、予測を最もうまくする方法は
    「何も面白いことをしない」こと。つまり**指標は行動の退化をむしろ高評価する**。数字だけでは
    構造的に検出できないので目視を残す。脳はK=100(0.5秒)に1回判断するので、判断の"間"のコマも
    撮って等速にする（render_everyステップ毎）。

    make_render_env: render_mode="rgb_array" の環境を1つ作って返す thunk（env構築は学習ループ側の
    設定に依存するため、呼び出し側から渡す）。
    """
    brain, fusion, target_fusion = ctx.brain, ctx.fusion, ctx.target_fusion
    zc, act_mean, K, rescale_action = ctx.zc, ctx.act_mean, ctx.K, ctx.rescale_action
    n_act, seed, DT = ctx.n_act, ctx.seed, ctx.DT
    try:
        import cv2
        renv = make_render_env()
        ro, _ = renv.reset(seed=seed)
        rh = brain.init_motor_hidden(); rpa = torch.zeros(n_act)
        frames = []; render_every = 4
        for _ in range(20):   # 20判断＝10秒ぶん
            sv = fusion.encode(ro); cf = target_fusion.encode(ro).detach()
            z, _, _, rhn = zc(sv, rpa, cf, rh); z = z.detach()
            a = torch.clamp(act_mean(z), -1.0, 1.0).detach()
            ctrl = rescale_action(a, renv.action_space)
            for k in range(K):
                ro, _, te, tr, _ = renv.step(ctrl)
                if k % render_every == 0:
                    f = renv.render()
                    if f is not None:
                        frames.append(f)
                if te or tr:
                    break
            rh = rhn.detach(); rpa = a
            if te or tr:
                ro, _ = renv.reset(); rh = brain.init_motor_hidden(); rpa = torch.zeros(n_act)
        renv.close()
        if frames:
            os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
            fps = (1.0 / DT) / render_every   # 等速再生になるfps
            hh, ww, _ = frames[0].shape
            vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"RECORDED {mp4_path} ({len(frames)}フレーム, 等速{fps:.0f}fps)", flush=True)
    except Exception as e:
        print(f"[警告] 録画に失敗（学習結果は保存済み）: {type(e).__name__}: {e}", flush=True)
