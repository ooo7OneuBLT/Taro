"""taro-C5：運動の質(なめらかさ)を、クリーンな仰向け環境で測る／見るハーネス。

【なぜ】太郎の運動が非人間的（Prechtl's GMAの基準ではカクつき過大＝異常寄り）と判明した。
最初の実験①（活性化ダイナミクス）はD視覚環境で試したが、交絡（倒れた姿勢・ベータ・視覚
レンダリング）が多すぎ、指標（眼球コマ間差分）も100ms間隔で35msの効果を検出できず判定不能
だった。ここでは**視覚なし・ベータなし・仰向け**のクリーンな環境で、GMAの"jerkiness"に直接
対応する**ジャーク（＝加速度の時間変化率）**を物理ステップ解像度で測る。これなら①の効果も
原理的に見える。

【測るもの】
  ・mean|jerk|：関節角加速度(qacc)の時間微分の絶対値平均。小さいほどなめらか（Flash&Hogan
    1985の最小ジャークの発想＝ヒトの滑らかな運動はジャークを最小化する）。
  ・境界ジャーク vs 内部ジャーク：1ティック(K=100)の「切り替わり目」と「保持中」でジャークを
    分けて集計。スナップ＆ホールドなら境界で跳ねるはず。①はこの境界の跳ねを抑えるのが狙い。
  ・per-tick 行動変化量：毎ティック行動がどれだけ"ジャンプ"するか（スナップの大きさ）。

【変えないもの】脳・方策・1秒に1回の判断。ACTION_SCALEのようなD側の後付けは使わない
（＝Cで実際に学習した方策そのものの運動を、素で測る）。

使い方:
  python d_c5_motor_quality.py view off          # 見る（従来トルク SpringDamperModel）
  python d_c5_motor_quality.py view on           # 見る（活性化ダイナミクス SmoothTorqueModel）
  python d_c5_motor_quality.py measure off [n]   # 測る（ヘッドレス, n=60ティック既定, 従来）
  python d_c5_motor_quality.py measure on  [n]   # 測る（活性化ダイナミクス）
  末尾に babble を付けると探索ノイズ(運動性喃語)込みで動かす（既定は決定的な方策平均）。

  【taro-C6 Step1】環境変数 C5_CTRL_M で制御の刻みを変える（既定=100=従来の1秒保持）：
    C5_CTRL_M=10 python d_c5_motor_quality.py view off      # 0.1秒ごとに感覚を見て出し直す＝速い連続制御
    C5_CTRL_M=10 python d_c5_motor_quality.py measure off 200
  ＝再学習なしで"実行だけ"速い制御に変え、jerkが下がるか（＝速い制御はなめらかにするか）を
  学習の交絡なしで測る（C4未実施の角度）。C5_CKPTで測るモデルを差替（推奨=c5_progress_seed0.pt）。
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import mujoco

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "taro_core"))
sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "C", "scripts"))  # fusion再エクスポート等
import paths
paths.setup_brain_path()
sys.path.insert(0, os.path.join(paths.SRC, "body"))   # smooth_actuation
sys.path.insert(0, paths.MIMO_DIR)

import mujoco.viewer
from hybrid_env import HybridEnv
from fusion import MinimalFusion, to_tensor
from taro_brain_motor import TaroBrainWithMotor
from cerebellum_motor import MotorCerebellum
from test_phase8_motor_learning import rescale_action
from d_supine_env import SupineMimoEnv
from mimoActuation.actuation import SpringDamperModel
from smooth_actuation import SmoothTorqueModel

# 既定は目標Cの学習済みモデル。C5_CKPT環境変数で別モデル（例：新生児＋努力コストで再学習した版）に差替可。
CKPT = os.environ.get("C5_CKPT",
                      os.path.join(_HERE, os.pardir, os.pardir, "C", "models", "c_pred_abs_seed0.pt"))
K = 100
# 【taro-C6 Step1】制御の刻み。C5_CTRL_M=10 で「毎10tick(0.1秒)ごとに感覚を見て行動を出し直す」
# ＝速い連続制御（閉ループ）。既定=K=100＝従来の1秒保持と完全に同一。再学習せず"実行だけ"変える
# ＝「速い制御は運動をなめらかにするか」を学習の交絡なしで測る（C4未実施の角度）。
CTRL_M = int(os.environ.get("C5_CTRL_M", str(K)))
# 体の月齢（成長モジュール）。C5_AGE=0 で新生児。空=18ヶ月児（従来）。
AGE = float(os.environ["C5_AGE"]) if os.environ.get("C5_AGE") else None
# 【目標E1】おもちゃ（押すと動く随伴対象）のある仰向け環境で見る/測る。E_TOY=1でopt-in。
# 既定OFF＝従来のD/C5と1バイトも変わらない。おもちゃは観測にも行動にも入らない（予測対象は
# 固有感覚のまま）ので、**C5の学習済みモデルをそのまま実行できる**＝再学習なしで目視できる。
E_TOY = os.environ.get("E_TOY", "0") == "1"


def load_matching(module, sd, tag):
    own = module.state_dict()
    matched = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in own if k not in matched]
    module.load_state_dict(matched, strict=False)
    note = f"（作り直し: {sorted(set(k.split('.')[0] for k in skipped))}）" if skipped else "（全層一致）"
    print(f"  [{tag}] ロード{len(matched)}層/作り直し{len(skipped)}層 {note}")


def build(mode_actuation, age=None):
    """age（月齢0〜24）を渡すと、MIMoの成長モジュールが体をその月齢に自動調整する
    （env内部で adjust_mimo_to_age を呼ぶ＝mimo_env.py）。既定 None＝18ヶ月児（従来）。"""
    seed = 0
    torch.manual_seed(seed); np.random.seed(seed)
    act_model = SmoothTorqueModel if mode_actuation == "on" else SpringDamperModel
    _kw = {"age": age} if age is not None else {}
    # 【2026-07-20 修正・重要】E1では**視覚と触覚を実際に脳へ繋ぐ**。
    # それまでは vision_params=None / touch_dim=0 で、太郎は内受容+固有感覚+前庭覚の
    # 3つだけ（sdim=192）で動いていた＝「視覚を入力に足した」という以前の記述は**誤り**で、
    # 環境側にパラメータを用意しただけで build から渡していなかった。
    # ⚠️sdimが192→320に変わるので、C5のチェックポイントは**一部の層が作り直しになる**
    #   （load_matching が形の合う層だけ読む）。E1は「視覚を使って自分の手を学ぶ」段階なので、
    #   視覚なしで学んだ重みをそのまま持ち越すほうが不自然、という判断で許容する。
    #   E_VISION=0 / E_TOUCH=0 で個別に切れる（アブレーション用）。E_TOY=0 なら従来と完全同一。
    _use_vision = E_TOY and os.environ.get("E_VISION", "1") == "1"
    _use_touch = E_TOY and os.environ.get("E_TOUCH", "1") == "1"
    if E_TOY:   # 目標E1：おもちゃ入りの仰向け環境（E/scripts/e_toy_env.py）
        sys.path.insert(0, os.path.join(_HERE, os.pardir, os.pardir, "E", "scripts"))
        from e_toy_env import ToySupineEnv, infant_vision_params, VISION_RES
        _vp = infant_vision_params() if _use_vision else None
        env = HybridEnv(ToySupineEnv(vision_params=_vp, actuation_model=act_model, **_kw))
        print("[E1] おもちゃ環境（ToySupineEnv）で実行")
    else:
        env = HybridEnv(SupineMimoEnv(vision_params=None, actuation_model=act_model, **_kw))
    obs, _ = env.reset(seed=seed)
    n_act = env.action_space.shape[0]
    _touch_dim = int(np.asarray(obs["touch"]).shape[0]) if (_use_touch and "touch" in obs) else 0
    _vres = VISION_RES if _use_vision else 0
    fusion = MinimalFusion(touch_dim=_touch_dim, vision_res=_vres)
    if E_TOY:
        print(f"[E1] 感覚: 視覚={'ON(' + str(_vres) + 'px)' if _vres else 'OFF'}"
              f"／触覚={'ON(' + str(_touch_dim) + '次元)' if _touch_dim else 'OFF'}")
    sdim = fusion.encode(obs).shape[0]
    prop_dim = to_tensor(obs["observation"]).shape[0]
    print(f"融合次元 sdim={sdim}／固有感覚 prop_dim={prop_dim}／行動 n_act={n_act}")
    brain = TaroBrainWithMotor(vocab_size=3, sensory_dim=sdim, n_actuators=n_act)
    emb_proj = nn.Linear(sdim + n_act, brain.sensory_proj.out_features)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"チェックポイント読込: {os.path.basename(CKPT)}")
    load_matching(brain, blob["brain"], "脳")
    fusion.insula.load_state_dict(blob["fusion_insula"])
    fusion.proprio.load_state_dict(blob["fusion_proprio"])
    fusion.vestibular.load_state_dict(blob["fusion_vestibular"])
    load_matching(emb_proj, blob["emb_proj"], "emb_proj")
    cereb = MotorCerebellum(brain.latent_dim, n_act)
    load_matching(cereb, blob["cereb"], "小脳")
    return env, brain, fusion, emb_proj, cereb, n_act


def actuated_dofs(model):
    """アクチュエータが駆動する関節のDOFアドレス（＝脳が動かす関節の集合）。"""
    dofs = []
    for i in range(model.nu):
        if model.actuator(i).name.startswith("beta_"):
            continue
        jid = model.actuator_trnid[i, 0]
        if jid >= 0:
            dofs.append(int(model.jnt_dofadr[jid]))
    return np.array(sorted(set(dofs)), dtype=int)


def make_policy(brain, fusion, emb_proj, cereb, n_act, babble):
    """1ティックぶんの決定的（または喃語込み）行動を返すクロージャ。"""
    ne_level = 0.095  # 学習後期のNE水準（ログ実測値）。babble時のノイズ幅に使う。

    def policy(obs, prev_a, hidden):
        sv = fusion.encode(obs); cf = sv.detach()
        emb = emb_proj(torch.cat([sv, prev_a], dim=-1)).unsqueeze(0).unsqueeze(0)
        out, nh = brain.motor_gru(emb, hidden)
        z, _, _ = brain.pc_latent.infer(hidden[-1, 0], out[0, -1], cf)
        z = z.detach()
        policy_m = torch.tanh(brain.motor_head(z))            # ★ACTION_SCALEなし＝Cで学習した素の方策
        w_c, cere_a, _ = cereb.gate(z, policy_m)
        mean = (1.0 - w_c) * policy_m + w_c * cere_a
        if babble:
            std = (0.05 + ne_level * 0.45) * (1.0 - w_c)
            a = torch.clamp(torch.distributions.Normal(mean, std).sample(), -1, 1)
        else:
            a = torch.clamp(mean, -1, 1)
        return a.detach(), nh.detach()

    return policy


class JerkMeter:
    """物理ステップごとのqaccを受け取り、ジャーク（qaccの時間微分）を集計する。"""

    def __init__(self, dofs, dt):
        self.dofs = dofs; self.dt = dt
        self.prev_qacc = None
        self.boundary, self.interior = [], []

    def observe(self, qacc, is_boundary):
        a = qacc[self.dofs]
        if self.prev_qacc is not None:
            jerk = np.abs(a - self.prev_qacc) / self.dt
            (self.boundary if is_boundary else self.interior).append(float(jerk.mean()))
        self.prev_qacc = a

    def summary(self):
        b = np.array(self.boundary) if self.boundary else np.array([0.0])
        it = np.array(self.interior) if self.interior else np.array([0.0])
        allj = np.concatenate([b, it])
        return {"mean": float(allj.mean()), "boundary": float(b.mean()),
                "interior": float(it.mean()), "max": float(allj.max())}


def run_view(mode_actuation, babble):
    """ライブビューア。C5_REALTIME=1 で**等倍速**（既定は計算任せ＝早送り）。

    等倍速でないと「人間の赤ちゃんと比べて速すぎ/遅すぎ」が判断できず目視の意味が薄れる。
    一時停止＝スペース、巻き戻し＝左パネルのHistory（MuJoCoビューアの標準機能）。
    """
    import time
    # 再生速度。1.0=等倍速、0=待たない（最速＝学習を早く進めたいとき）。
    # 実行中にキーで変えられる（等速だと学習の進みを見るのに時間がかかりすぎるため）：
    #   .（>） 2倍速く   ,（<） 2倍遅く   0 等倍に戻す   M 最速（待たない）
    speed = [1.0 if os.environ.get("C5_REALTIME", "0") == "1" else 0.0]

    def _key_cb(keycode):
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch in ".>":
            speed[0] = min(speed[0] * 2 if speed[0] > 0 else 64.0, 64.0)
        elif ch in ",<":
            speed[0] = max(speed[0] / 2 if speed[0] > 0 else 1.0, 0.0625)
        elif ch == "0":
            speed[0] = 1.0
        elif ch in "mM":
            speed[0] = 0.0
        else:
            return
        print(f"  [speed] x{speed[0]:.4g}" if speed[0] > 0 else "  [speed] MAX (no wait)",
              flush=True)
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    _ctrl_note = f"制御刻み={CTRL_M}tick({'速い連続制御' if CTRL_M < K else '1秒保持'})"
    print(f"\nビューア起動。仰向けの太郎が学習済みの脳で動きます（活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}／{_ctrl_note}）。スペース=一時停止、左History=巻き戻し。")
    print("  キー操作: . = 速く / , = 遅く / 0 = 等倍 / M = 最速（待たない）")

    def _show_speed(viewer, eff=0.0):
        """現在の再生倍率を**画面右上のUI**に出す。

        第1版はシーン内にラベルgeomを浮かせたが、3D空間の物体として描かれるので
        カメラを動かすと位置がずれ、見づらかった。ビューアのオーバーレイ(mjr_overlay)は
        passive viewer から直接触れないため、**ウィンドウのタイトル**に出す方式にする
        （常に画面上部に見え、カメラ操作の影響を受けない）。
        """
        # ★「要求倍率」と「実効倍率」を**両方**出す。第1版は要求だけを出していたが、
        #   実測で x2 以上は計算が追いつかず頭打ち（1物理ステップの計算6.46ms > sim 10ms/倍率）
        #   と判明した＝x64と表示しながら実際は約1.5倍速だった。要求だけの表示は嘘になる。
        req = (f"x{speed[0]:.4g}" if speed[0] > 0 else "MAX")
        txt = f"speed {req} (real x{eff:.1f})"
        if getattr(viewer, "_last_speed_txt", None) == txt:
            return
        viewer._last_speed_txt = txt
        try:
            # set_figures はビューアの右上に固定表示されるオーバーレイ（3D空間ではないので
            # カメラを動かしてもずれない）。mjvFigure のタイトルを速度表示に使う。
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.title = txt
            fig.flg_legend = 0
            fig.flg_ticklabel[:] = [0, 0]          # 配列フィールドは[:]で代入する
            fig.figurergba[:] = [0.0, 0.0, 0.0, 0.4]
            vp = viewer.viewport
            w = max(int(vp.width * 0.22), 180)
            h = 46
            rect = mujoco.MjrRect(int(vp.width - w - 10), int(vp.height - h - 10), w, h)
            viewer.set_figures([(rect, fig)])
        except Exception as e:
            print(f"  [{txt}] (overlay unavailable: {type(e).__name__})", flush=True)

    tick = 0; ctrl = None
    # 画面の更新は60Hzで十分（人間の目に見える上限）。第1版は**毎物理ステップ**（等倍で100Hz、
    # 倍速時はもっと）描画していて、それ自体が倍速を頭打ちにする原因の一つだった。
    SYNC_DT = 1.0 / 60.0
    with mujoco.viewer.launch_passive(m, d, key_callback=_key_cb) as viewer:
        t_wall = time.perf_counter()   # 次に物理を進めてよい実時刻（sleepの累積誤差を防ぐ）
        t_draw = 0.0
        eff_t0, eff_sim, eff = time.perf_counter(), 0.0, 0.0
        while viewer.is_running():
            for k in range(K):
                if k % CTRL_M == 0:   # 制御の刻みごとに、今の感覚を見て行動を出し直す（閉ループ）
                    a, hidden = policy(obs, prev_a, hidden)
                    ctrl = rescale_action(a, env.action_space)
                    prev_a = a
                obs, r, te, tr, info = env.step(ctrl)
                meter.observe(d.qacc.copy(), is_boundary=(k % CTRL_M == 0))
                now = time.perf_counter()
                eff_sim += dt_env
                if now - eff_t0 >= 0.5:       # 実効倍率＝直近0.5秒の「sim時間/実時間」
                    eff = eff_sim / (now - eff_t0)
                    eff_t0, eff_sim = now, 0.0
                if now - t_draw >= SYNC_DT:
                    _show_speed(viewer, eff)
                    viewer.sync()
                    t_draw = now
                if speed[0] > 0:
                    # 目標時刻を積み上げて追従する。第1版の「毎回 sleep(dt/speed)」は
                    # Windowsのsleep分解能(~1-15ms)ぶん必ず遅れ、その誤差が積もっていた。
                    t_wall += dt_env / speed[0]
                    lag = t_wall - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.25:         # 大きく遅れたら追いつくのを諦めて基準を引き直す
                        t_wall = time.perf_counter()
                else:
                    t_wall = now
                if te or tr:
                    break
            tick += 1
            if tick % 20 == 0:
                s = meter.summary()
                print(f"  tick {tick}: mean|jerk|={s['mean']:.1f} 境界={s['boundary']:.1f} "
                      f"内部={s['interior']:.1f} 最大={s['max']:.1f}", flush=True)


def run_measure(mode_actuation, n, babble):
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble)
    m, d = env.unwrapped.model, env.unwrapped.data
    dofs = actuated_dofs(m)
    dt_env = m.opt.timestep * env.unwrapped.frame_skip
    meter = JerkMeter(dofs, dt_env)

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    action_jumps = []
    _ctrl_note = f"制御刻み={CTRL_M}tick({'速い連続制御' if CTRL_M < K else '1秒保持'})"
    print(f"\n測定開始（{n}ティック・活性化ダイナミクス={mode_actuation.upper()}"
          f"／{'喃語込み' if babble else '決定的'}／{_ctrl_note}）")
    ctrl = None
    for tick in range(n):
        for k in range(K):
            if k % CTRL_M == 0:   # 制御の刻みごとに今の感覚を見て出し直す（閉ループ）
                a, hidden = policy(obs, prev_a, hidden)
                action_jumps.append(float(np.abs((a - prev_a).numpy()).mean()))
                ctrl = rescale_action(a, env.action_space)
                prev_a = a
            obs, r, te, tr, info = env.step(ctrl)
            meter.observe(d.qacc.copy(), is_boundary=(k % CTRL_M == 0))
            if te or tr:
                obs, _ = env.reset()
                hidden = brain.init_motor_hidden()
                break
    s = meter.summary()
    print(f"\n===== 結果（活性化={mode_actuation.upper()}／{_ctrl_note}）=====")
    print(f"mean|jerk|      = {s['mean']:.2f}   (小さいほどなめらか)")
    print(f"  境界ジャーク  = {s['boundary']:.2f}  (ティック切替の瞬間)")
    print(f"  内部ジャーク  = {s['interior']:.2f}  (行動保持中)")
    print(f"  最大ジャーク  = {s['max']:.2f}")
    print(f"行動ジャンプ = {np.mean(action_jumps):.3f}  (1判断ごとに行動がどれだけ跳ぶか)")
    return s


def run_eyeview(mode_actuation, n, babble):
    """【運動性喃語×egomotion】仰向け・新生児・相手なし・視覚ありで自発運動を動かし、
    第三者視点＋一人称(眼球)視界を録画する。＝C5の運動が"視界を揺らさないか(egomotion)"を、
    姿勢/体/相手の交絡なしで目視するための動画。脳は視覚を使わない（＝眼球カメラを"見るだけ"）。"""
    import cv2
    env, brain, fusion, emb_proj, cereb, n_act = build(mode_actuation, age=AGE)
    policy = make_policy(brain, fusion, emb_proj, cereb, n_act, babble)
    m, d = env.unwrapped.model, env.unwrapped.data
    # オフスクリーン描画のフレームバッファを広げる（既定500pxだと640×480が入らない）
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)

    third_ren = mujoco.Renderer(m, height=480, width=640)
    third_cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, third_cam)
    # E1（おもちゃ）は「手とおもちゃの位置関係」を見るのが目的なので寄る＋太郎を追う。
    # 既定の引きだと太郎が豆粒でおもちゃが見えない（実際に一度そうなった）。
    third_cam.distance *= 0.32 if E_TOY else 1.3
    if E_TOY:
        third_cam.elevation = -35.0
    try:
        eye_cid = int(m.camera("eye_left").id)
    except Exception:
        print("⚠️ eye_left カメラが見つからず一人称は録画できません（第三者のみ）"); eye_cid = None
    eye_ren = eye_cam = None
    if eye_cid is not None:
        eye_ren = mujoco.Renderer(m, height=64, width=64)
        eye_cam = mujoco.MjvCamera(); eye_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        eye_cam.fixedcamid = eye_cid

    obs, _ = env.reset(seed=0)
    hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
    third_frames, eye_frames = [], []
    SUB = 2  # 物理2ステップに1コマ録画（秒内の揺れが見える解像度）
    ctrl = None; step_i = 0
    _age = "新生児(0m)" if AGE == 0 else (f"{int(AGE)}m" if AGE is not None else "18m既定")
    print(f"\n録画開始（{n}ティック・仰向け・体={_age}・視覚あり(眼球)・相手なし・"
          f"{'喃語(運動性喃語)込み' if babble else '決定的'}／制御刻み={CTRL_M}tick）")
    for tick in range(n):
        for k in range(K):
            if k % CTRL_M == 0:
                a, hidden = policy(obs, prev_a, hidden)
                ctrl = rescale_action(a, env.action_space); prev_a = a
            obs, r, te, tr, info = env.step(ctrl)
            if step_i % SUB == 0:
                if E_TOY:   # 太郎は暴れて移動するのでカメラを体に追従させる
                    third_cam.lookat[:] = d.body("upper_body").xpos
                third_ren.update_scene(d, camera=third_cam)
                third_frames.append(third_ren.render().copy())
                if eye_ren is not None:
                    eye_ren.update_scene(d, camera=eye_cam)
                    eye_frames.append(cv2.resize(eye_ren.render().copy(), (240, 240),
                                                 interpolation=cv2.INTER_NEAREST))
            step_i += 1
            if te or tr:
                obs, _ = env.reset(); hidden = brain.init_motor_hidden(); prev_a = torch.zeros(n_act)
                break

    # 【重要】E1（おもちゃ環境）の動画は**別名・別フォルダ**に出す。
    # 同名で出すと従来のC5動画（おもちゃ無し）を上書きしてしまう（実際に一度やらかした）。
    # ログは消さず分類して保存する、という運用に合わせる。
    _age_tag = int(AGE) if AGE is not None else "def"
    _bab = "babble" if babble else "det"
    if E_TOY:
        out_dir = os.path.join(_HERE, os.pardir, os.pardir, "E", "logs", "video")
        tag = f"e1toy_age{_age_tag}_m{CTRL_M}_{_bab}"
    else:
        out_dir = os.path.join(_HERE, os.pardir, "logs", "video")
        tag = f"c5eye_age{_age_tag}_m{CTRL_M}_{_bab}"
    os.makedirs(out_dir, exist_ok=True)

    def _write(frames, name, size, fps=50):
        p = os.path.join(out_dir, f"{tag}_{name}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release(); return p

    p3 = _write(third_frames, "third", (640, 480))
    print("第三者視点mp4:", p3)
    # 静止画シート（等間隔に抜いた6コマ）。動画を再生せずに配置・接触を一目で確認するため。
    try:
        idx = np.linspace(0, len(third_frames) - 1, 6).astype(int)
        rows = [np.hstack([third_frames[i] for i in idx[:3]]),
                np.hstack([third_frames[i] for i in idx[3:]])]
        sheet = np.vstack(rows)
        sp = os.path.join(out_dir, f"{tag}_sheet.png")
        cv2.imwrite(sp, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("第三者視点シートpng:", sp)
    except Exception as e:
        print(f"[警告] シート画像の保存に失敗: {type(e).__name__}: {e}")
    if eye_frames:
        pe = _write(eye_frames, "eye", (240, 240))
        print("一人称(眼球)mp4:", pe)
    print("→ 一人称が激しく揺れるほど、運動が視界(egomotion)を汚している＝C6が要る兆候。")
    return p3


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    actuation = sys.argv[2] if len(sys.argv) > 2 else "off"
    rest = sys.argv[3:]
    babble = "babble" in rest
    nums = [int(x) for x in rest if x.isdigit()]
    n = nums[0] if nums else 60
    print(f"=== taro-C5 運動の質: mode={mode} 活性化ダイナミクス={actuation.upper()} "
          f"{'喃語込み' if babble else '決定的'} ===")
    if mode == "measure":
        run_measure(actuation, n, babble)
    elif mode == "eyeview":
        run_eyeview(actuation, n if nums else 15, babble)
    else:
        run_view(actuation, babble)
