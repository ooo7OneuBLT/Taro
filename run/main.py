"""シミュレーションシステム ── 実験・学習・目視の唯一の入口。

【なぜ作ったか、2026-07-30】目標Eの実験スクリプトが118本あり、そのうち66本が
**それぞれ独立に環境を組み立てていた**。そのため「学習は関節モード、測定・Viewerは
筋肉モード」という別の体で動いていた事故が起きた（ユーザーの目視で発覚。実測で
動きが人間の新生児の約3.3倍速かった）。
設計の全体像は `E/docs/実行基盤_設計.md`。

【使い方】
    .venv/Scripts/python.exe -m run.main E/experiments/<名前>.json
    .venv/Scripts/python.exe -m run.main E/experiments/<名前>.json --steps 600

【実験ファイル（JSON）の4つの欄】境界を混ぜない
    scene    どんな環境か（run/scenes/*.json の名前）
    taro     太郎の中身の設定（本能のON/OFF・月齢・駆動モード）
             → 実装は taro_core にある。ここは「どれを使うか」だけ
    run      動かし方（type / steps / seed）
    plugins  外から測る・見る道具（太郎を変えない）

注意：env を直接作るコードを**新しく書かない**。ここを通す。
  （検査は run/tools/check_entry.py）
"""
import argparse
import json
import os
import sys
import time
import warnings

# 【2026-07-30 に解決】「同じシードでも結果がばらつく」問題は直した。
#   原因は5つ、全部自分たちのコードだった：
#     ①内臓（泣く・寝る・うとうと）の乱数に種を撒いていなかった ← 本物のバグ
#       ＝乱数は4系統（torch / numpy / gym / Python標準）ある
#     ②筋の活性化（MuscleModel.activity＝Python側の配列）が復元されていなかった
#     ③物理の内部状態（qacc_warmstart＝ソルバの前回解）が復元されていなかった
#     ④視覚のキャッシュ（辞書）が復元されていなかった
#     ⑤測定が学習を8秒ぶん進める構造そのもの
#   対策＝測定の前後で状態を控えて戻す（`run/trainer.py` の _snapshot/_restore）。
#   結果 350ステップ × 5回すべて完全一致。
#   注意：過去の実験（margin +58.8 等）は同一シードでも再現しない条件での値。
#   注意：再現性が崩れていないかは `run/tools/check_divergence.py` で確かめられる。
#     → 検証の落とし穴チェックリスト 項79・項80

warnings.filterwarnings("ignore")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# プラグインの登録表。実験ファイルのキーとクラスの対応はここだけ。
#   新しい測る道具を作ったら、ここに1行足す。
PLUGINS = {}


def _register():
    """プラグインを登録する。import が重いものは使うときだけ読む。"""
    from run.plugins.common.toy_touch import ToyTouch
    from run.plugins.common.hand_in_view import HandInView
    from run.plugins.common.self_model import SelfModel
    from run.plugins.common.trace import Trace
    from run.plugins.common.dashboard import Dashboard
    from run.plugins.common.movement_units import MovementUnits
    from run.plugins.common.reach_success import ReachSuccess
    from run.plugins.common.double_touch import DoubleTouch
    from run.plugins.common.contact_reward import ContactReward
    from run.plugins.common.block_progress_probe import BlockProgressProbe
    from run.plugins.common.plateau_stop import PlateauStop
    from run.plugins.common.posture_probe import PostureProbe
    from run.plugins.common.toy_in_view import ToyInView
    from run.plugins.common.word_learning import WordLearning
    from run.plugins.common.model_snapshots import ModelSnapshots
    from run.plugins.common.view_video import ViewVideo
    from run.plugins.common.word_production import WordProduction
    from run.plugins.common.babble_probe import BabbleProbe
    from run.plugins.common.gaze_probe import GazeProbe
    from run.plugins.common.produce_snapshot import ProduceSnapshot
    from run.plugins.common.vergence_probe import VergenceProbe
    from run.plugins.common.object_files import ObjectFiles
    from run.plugins.common.world_predictor_log import WorldPredictorLog
    from run.plugins.common.world_predictor_record import WorldPredictorRecord
    PLUGINS["toy_touch"] = ToyTouch
    PLUGINS["toy_in_view"] = ToyInView   # おもちゃが視界に入っている割合（幾何・脳不要＝measure可）
    PLUGINS["hand_in_view"] = HandInView
    PLUGINS["self_model"] = SelfModel
    PLUGINS["movement_units"] = MovementUnits   # 動きが滑らかか（リーチングの指標）
    PLUGINS["reach_success"] = ReachSuccess     # 自己接触が実際に起きているか（直接測る）
    PLUGINS["double_touch"] = DoubleTouch       # ダブルタッチ（自己接触の一致）検出（報酬には未接続）
    PLUGINS["contact_reward"] = ContactReward   # 自己接触あり/なしで報酬・RPEを直接比較する
    PLUGINS["block_progress_probe"] = BlockProgressProbe   # ブロックごとのprogressを見るだけ（読むだけ）
    PLUGINS["plateau_stop"] = PlateauStop       # cereb_errの頭打ちを検出し学習を打ち切る（唯一ctx.stop_requestedを書く）
    PLUGINS["posture_probe"] = PostureProbe   # 頭の高さと報酬をCSVに残す（姿勢の学習の結果そのもの）
    PLUGINS["trace"] = Trace           # 内部の値の指紋を残す（原因追跡用）
    PLUGINS["dashboard"] = Dashboard   # 学習の様子の絵を自動で作り直す
    PLUGINS["word_learning"] = WordLearning   # 親のfollow-in labelingで語彙が育っているか（F1-3）
    PLUGINS["view_video"] = ViewVideo         # 太郎の視界＋第三者視点を走行中に動画へ（目視承認用・2026-09-03）
    PLUGINS["model_snapshots"] = ModelSnapshots   # checkpointごとに脳を途中保存（走行後にオフライン測定する用・F2-33）
    PLUGINS["word_production"] = WordProduction   # 太郎自身の発話（見た物の名前を言う、F2）
    PLUGINS["babble_probe"] = BabbleProbe   # 喃語モードの発話・帳面の成長を記録する（F2-1）
    PLUGINS["gaze_probe"] = GazeProbe   # 視線が的にどれだけ連続で留まるか（F2-9の判定設計用）
    PLUGINS["produce_snapshot"] = ProduceSnapshot   # 発話した瞬間の中心窩画像と視覚ベクトル（F2-11のりんご偏り究明）
    PLUGINS["vergence_probe"] = VergenceProbe   # 輻輳角の目標と実測（F2-15の切り分け）
    PLUGINS["object_files"] = ObjectFiles   # MobileSAM+物体ファイルを本番の走行で動かす（F2-67）
    PLUGINS["world_predictor_log"] = WorldPredictorLog   # 世界の予測器（M7a）の誤差をCSVに書く（測るだけ・2026-09-08）
    PLUGINS["world_predictor_record"] = WorldPredictorRecord   # 世界の予測器の入力を1回記録（切り分け実験用・2026-09-10）
    # 注意：self_model / trace / reach_success / double_touch は太郎の脳が要る
    #   （run.type=train のみ）。measure（脳を通さず環境だけ進める）では使えない。
    #   reach_success・double_touch はさらに taro.goal_space="reach_self" も要る。


def load_spec(path):
    """実験ファイルを読む。書き間違いをここで止める。"""
    with open(path, encoding="utf-8") as fp:
        spec = json.load(fp)
    # 【2026-09-11】expect＝「この走行は何を出すはずか」の宣言。走行そのものには
    #   使わない（run/tools/smoke.py が試し走行の結果と突き合わせる）。
    known = {"name", "note", "scene", "world", "taro", "run", "plugins", "expect"}
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"実験ファイルに知らない欄がある: {sorted(unknown)}\n"
                        f"  使える欄: {sorted(known)}")
    if "scene" not in spec:
        raise ValueError("実験ファイルに scene が無い（どの環境で回すか）")
    spec.setdefault("taro", {})
    spec.setdefault("run", {})
    spec.setdefault("plugins", {})
    return spec


def build_plugins(spec):
    """実験ファイルの plugins 欄から、使う道具を組み立てる。

    `run.csv` を指定した学習には、書かなくても `dashboard`（絵の自動更新）を足す。
      ⇒「実験を流せば勝手に絵ができる」状態にするため（2026-07-30 の要望）。
      切りたいときは実験ファイルに `"dashboard": false` と明示する。
    """
    r = spec.get("run") or {}
    if (str(r.get("type", "measure")).lower() == "train" and r.get("csv")
            and "dashboard" not in spec["plugins"]):
        spec["plugins"]["dashboard"] = True
    out = []
    for key, cfg in spec["plugins"].items():
        if cfg is False or cfg is None:
            continue
        if key not in PLUGINS:
            raise ValueError(f"知らないプラグイン: {key}\n"
                            f"  使えるもの: {sorted(PLUGINS)}\n"
                            f"  （足すには run/main.py の PLUGINS に登録する）")
        out.append(PLUGINS[key](cfg if isinstance(cfg, dict) else {}))
    return out


def _csv_logger(path, spec):
    """チェックポイントの記録をCSVに残す関数を作る。

    注意：列は**途中で増える**（学習前は「力の出し具合」がまだ無いなど）。
      追記だと最初の行で列が固定され、あとから増えた値が永久に落ちる。
      ⇒ 行を溜めて**毎回すべて書き直す**。記録は数十行なので軽く、
        走行中に読んでも常に整合した表になっている。
    """
    import csv
    if not path:
        return (lambda row: None)
    path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []

    def log_row(row):
        """1行分の記録 row を受け取り、これまでの行に足してCSVの全行を書き直す。新しい列が現れたら列を増やし、一時ファイルへ書いてから置き換える。戻り値は無い。"""
        if not isinstance(row, dict):
            return
        rows.append(dict(row))
        cols = []               # 出現した順に並べる（step が先頭に来る）
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        os.replace(tmp, path)   # 書き換え中の半端な表を読ませない
    return log_row


def _write_run_meta(spec, out_dir):
    """この走行が**実際に何で動いたか**を1本のJSONに残す（2026-09-14・ステップ1）。

    【なぜ本体で書くか】以前これを書いていたのは dashboard プラグイン
    （`run/plugins/common/dashboard.py`）で、①道具なので外せる ②`run.csv` が
    無いと書かない ③`except Exception` で黙って失敗する、の3点から
    **270フォルダ中123本にしかなかった**（2026-09-14 実測）。本番 F2-126 にも無い。

    【なぜ全部要るか】場面ファイルを直したり消したりすると、その場面で走った
    過去の走行が何だったか分からなくなる。ログ側に写しがあれば触れる。
    また 771/805 本の実験が過去のモデルを出発点にしているので、
    「そのモデルがどの環境で育ったか」はここからしか引けない
    （F2-129 が背景の壁なしで育っていた件は、これがあれば自動で分かる）。

    【環境変数】`E_` で始まる設定が 110 個あり、どこにも記録されていなかったため
    `F2-117pre_積み上げ0.2/0.5/1.0` の3本は**設定が完全に同一で再現できない**。
    ここで丸ごと控える。注意：控えるのは「環境に設定されていた値」であって、
    設定されていなければ現れない（そのときはコード側の既定値が効く）。

    【欄の増やし方】`name` は `run/tools/preflight.py` が「他人のフォルダでないか」の
    判定に読む。ほかにも `run/tools/dashboard.py`・`check_jitter.py` が読むので、
    **既存の欄は名前も意味も変えず、足すだけ**にする。
    """
    import datetime
    import subprocess
    from run.world import scene as scene_mod

    sc = scene_mod.resolve(spec["scene"], spec.get("taro") or {},
                           spec.get("world"))

    rev = None
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=_ROOT, capture_output=True, text=True,
                             timeout=5).stdout.strip() or None
    except Exception:       # noqa: BLE001  git が無くても走行は続ける
        rev = None

    meta = {
        # ---- 既存の欄（読み手がいるので変えない）--------------------------
        "name": spec.get("name"),
        "scene": spec.get("scene"),
        "world_override": spec.get("world"),   # 実験ファイルが上書きした分（ステップ3）
        "taro": spec.get("taro"),
        "run": spec.get("run"),
        "tools": sorted(k for k, v in (spec.get("plugins") or {}).items() if v),
        "scene_note": sc.get("note"),
        "world": sc.get("world"),
        "body": sc.get("body"),
        # ---- ここから 2026-09-14 に足した欄 --------------------------------
        "meta_schema": 2,
        "note": spec.get("note"),
        "plugins": spec.get("plugins"),
        "expect": spec.get("expect"),
        "setup": sc.get("setup"),
        "state": sc.get("state"),
        "noise_mode": sc.get("noise_mode"),
        "fingerprint": sc.get("fingerprint"),
        "scene_created": sc.get("created"),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("E_")},
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": rev,
    }
    d = out_dir if os.path.isabs(out_dir) else os.path.join(_ROOT, out_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "run.meta.json")
    # 注意：ここで失敗したら**止める**。以前は握りつぶしていたので、
    #   「記録が無い」ことに気づけなかった。
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    return path


def run(spec, *, steps_override=None, verbose=False):
    """実験仕様 spec（と任意の steps_override・verbose）を受け取り、run.type（train/view/edit/measure）に応じて学習・目視・編集用Viewer起動・脳を通さない測定のいずれかを行う。見出しの表示と run.meta.json の書き出し、プラグインの組み立てを行ったうえで各分岐へ渡す。プラグインの報告や終了コードをまとめた辞書を返す。
    """
    from run.context import Ctx
    from run.plugins.common import scene as scene_mod

    r = spec["run"]
    kind = str(r.get("type", "measure")).lower()
    n_steps = int(steps_override if steps_override is not None else r.get("steps", 600))
    seed = int(r.get("seed", 0))

    print("=" * 74)
    print(f" {spec.get('name', '(名前なし)')}")
    print("=" * 74)
    print(f"  シーン   {spec['scene']}")
    print(f"  太郎     {spec['taro'] or '(既定)'}")
    print(f"  動かし方 {kind} / {n_steps}ステップ / seed={seed}")
    print(f"  道具     {sorted(k for k, v in spec['plugins'].items() if v)}")

    # 【2026-09-14・ステップ1】走行の記録を最初に置く。道具を1つも付けない
    #   走行でも必ず残る（以前は dashboard プラグイン頼みで46%しか無かった）。
    _meta_path = _write_run_meta(spec, _out_dir_of(spec))
    print(f"  記録     {os.path.relpath(_meta_path, _ROOT)}")

    plugins = build_plugins(spec)

    # --- train：太郎の脳を通して学習する（run/trainer.py）------------------
    if kind == "train":
        from run.config import Config
        from run import trainer
        cfg = Config.from_spec(spec, steps_override=steps_override)
        print(f"  設定     {cfg.summary()}")
        log_row = _csv_logger(r.get("csv"), spec)
        out = trainer.train(cfg, plugins=plugins, verbose=verbose, log_row=log_row)
        print("-" * 74)
        for k, v in out.items():
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
        _check_learning_happened(spec, out)
        return out

    # --- view：等倍速で太郎を見る（run/viewer.py）----------------------------
    if kind == "view":
        from run.config import Config
        from run import viewer
        cfg = Config.from_spec(spec, steps_override=steps_override)
        print(f"  設定     {cfg.summary()}")
        return viewer.view(cfg, plugins=plugins, verbose=verbose)

    # --- edit：編集ウィンドウ付きの Viewer（run/viewer_tools/e_viewer.py）を開く ---
    #   【なぜ実験ファイルから開けるようにしたか、2026-07-31】
    #   e_viewer は環境変数で設定する古い作りで、「どの条件で開いたか」が
    #   コマンドの履歴にしか残らなかった。実験ファイルにすれば記録が残る。
    #   注意：中身は e_viewer.py のまま（統合の第4段階・ステップ2）。
    #     環境変数での起動も残す（急に壊さない）。
    #   【なぜ、2026-08-07】E/scripts/e_viewer.py から run/viewer_tools/e_viewer.py
    #     へ移設（設計：作業記録（非公開）
    #     2026-08-07_runSystem移設_統合版.md）。パス参照をここで更新。
    if kind == "edit":
        import subprocess
        t = spec.get("taro") or {}
        envv = dict(os.environ)
        envv["E_SCENE"] = str(spec["scene"])
        envv["PYTHONIOENCODING"] = "utf-8"
        for key, name in (("age_months", "E_AGE"),
                          ("model", "E_VIEW_MODEL"),
                          ("view_model_b", "E_VIEW_MODEL_B"),
                          ("view_std", "E_VIEW_STD")):
            v = t.get(key, r.get(key))
            if v is not None:
                envv[name] = str(v)
        # 【なぜ、2026-08-10】taro.actuation（駆動モード）が e_viewer.py に
        #   渡っていなかった。渡さないと e_viewer.py は常に筋肉モードで体を
        #   作ってしまう（監査：作業記録（非公開）
        #   2026-08-10_run系システムとViewerの型バグ横断監査.md「中2」）。
        #   既定（未指定）は run/plugins/common/scene.py と同じ "muscle"。
        #   ここで既定値をそのまま明示しても、e_viewer.py 側の分岐で
        #   "muscle" は今までどおり MuscleModel を選ぶだけなので、既存の
        #   挙動（全実験が muscle）は変わらない。
        if t.get("actuation") is not None:
            envv["E_ACTUATION"] = str(t["actuation"])
        # 【なぜ、2026-08-11】taro欄のキーを1つずつ列挙して環境変数へ転送する
        #   やり方は、新しいキー（例：伸張反射＋揺らぐ振動子の共通駆動＝
        #   spinal_drive_mode・common_drive_rho・common_drive_grouping・
        #   common_drive_osc_params）が増えるたびに転送漏れを繰り返す
        #   （2026-08-07 の taro.noise と同型の再発。
        #   E/docs/figures/駆動モード比較_2026-08-07/_録画スクリプト_参考.py 冒頭）。
        #   ⇒ taro欄（JSON化できる値だけ）を丸ごとJSON文字列にして渡す。
        #   e_viewer.py 側はこれを taro_spec の「土台」として展開し、そのうえで
        #   既存の個別計算（actuation・age_months・model・noise/beta/synergy/
        #   syn_wの上書き）を今までどおり適用する＝個別計算が必ず勝つ
        #   （2026-08-10の駆動モード食い違いチェックの前提を壊さないため）。
        #   注意：これにより、_load_brain がこれまで受け取っていなかった
        #   taro欄の他のキー（reward・touch等）も編集ウィンドウ側へ伝わるように
        #   なる（副次的な挙動の広がり。仕様2026-08-11_新機構バグ2件修正.md
        #   担当B節に明記のとおり、作業記録にも記載する）。
        if t:
            try:
                envv["E_TARO_SPEC_JSON"] = json.dumps(t, ensure_ascii=False)
            except (TypeError, ValueError) as _e:      # noqa: BLE001
                print(f"注意 taro欄をJSON化できませんでした（編集ウィンドウには"
                      f"個別項目のみ渡ります）: {type(_e).__name__}: {_e}", flush=True)
        script = os.path.join(_ROOT, "run", "viewer_tools", "e_viewer.py")
        print(f"  編集ウィンドウ付き Viewer を開きます\n"
              f"     シーン {envv['E_SCENE']}"
              + (f" ／ 月齢 {envv['E_AGE']}ヶ月" if "E_AGE" in envv else "")
              + (f"\n     脳A {envv['E_VIEW_MODEL']}" if "E_VIEW_MODEL" in envv else "")
              + (f"\n     脳B {envv['E_VIEW_MODEL_B']}" if "E_VIEW_MODEL_B" in envv else ""),
              flush=True)
        return {"edit": {"戻り値": subprocess.call([sys.executable, script], env=envv)}}

    if kind not in ("measure",):
        raise ValueError(f"run.type が不明: {kind}（measure / train / view / edit）")

    # --- measure：太郎の脳を通さず、環境をそのまま進めて測る ---
    import numpy as np
    env, sc, _hands = scene_mod.build(spec["scene"], taro=spec["taro"],
                                      seed=seed, verbose=verbose,
                                      world=spec.get("world"))
    u = env.unwrapped
    K = int(r.get("K", 10))
    dt = float(u.model.opt.timestep) * int(u.frame_skip) * K
    ctx = Ctx(env=env, spec=spec, scene=sc, n_steps=n_steps, dt=dt)
    for p in plugins:
        p.setup(ctx)
    obs, _ = env.reset(seed=seed)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    if float(env.action_space.low[0]) >= 0.0:
        zero[:] = 0.0            # 筋肉モードは [0,1]＝0が「力を入れない」
    ckpt = int(r.get("checkpoint", max(1, n_steps // 10)))
    t0 = time.time()
    for i in range(n_steps):
        for _ in range(K):
            obs, _rew, term, trunc, _info = env.step(zero)
            if term or trunc:
                obs, _ = env.reset()
        ctx.step = i + 1
        for p in plugins:
            p.on_step(ctx)
        if (i + 1) % ckpt == 0:
            for p in plugins:
                p.on_checkpoint(ctx)
            tags = [p.line(ctx) for p in plugins]
            tags = [t for t in tags if t]
            print(f"  [{i+1}/{n_steps}] sim={ctx.sim_sec:.0f}s  " + "  ".join(tags),
                  flush=True)

    print(f"\n  実時間 {time.time() - t0:.0f}秒 / sim {ctx.sim_sec:.0f}秒")
    print("-" * 74)
    out = {}
    for p in plugins:
        rep = p.report(ctx)
        if rep:
            out[p.name] = rep
            print(f"  {p.name}: {json.dumps(rep, ensure_ascii=False)}")
    # 注意：閉じ損ねると MuJoCo の描画コンテキストが残り、次の実行が不安定になる
    from run.trainer import close_env
    close_env(env)
    _check_learning_happened(spec, out)
    return out


# ---- 起動時ガード：テクスチャの解像度 ---------------------------------------
# 【2026-08-28・なぜ機械で止めるか】8/27に「テクスチャ512px化でメモリが3757→1524MBに
#   減る」と実測したのに、**適用せず忘れたまま1日走らせ続けた**。1プロセス3178MBのうち
#   2485MBがMuJoCoモデルで、その中身は顔2500x2500などのテクスチャ1025MB。太郎の目は
#   周辺128px・中心窩208pxなので20倍の過剰だった。
#   CLAUDE.md「同じミスが2回起きたら、文書ではなく機械で防ぐ」に従い、注意書きではなく
#   **起動を止める**。原本は MIMo/mimoEnv/assets/tex_original_2500px/ に退避してある。
#   画像のヘッダだけを読むので、この検査自体は数ミリ秒しかかからない。
TEXTURE_MAX_PX = 512
TEXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "MIMo", "mimoEnv", "assets", "tex")


def _check_texture_resolution():
    """テクスチャが上限を超えていたら、走らせずに止める。"""
    import glob
    try:
        from PIL import Image
    except ImportError:
        return   # PILが無い環境では検査を諦める（走行そのものは止めない）
    bad = []
    for f in sorted(glob.glob(os.path.join(TEXTURE_DIR, "*.png"))
                    + glob.glob(os.path.join(TEXTURE_DIR, "*.jpg"))):
        try:
            with Image.open(f) as im:
                w, h = im.size
        except Exception:
            continue
        if max(w, h) > TEXTURE_MAX_PX:
            bad.append((os.path.basename(f), w, h))
    if not bad:
        return
    lines = ["", "=" * 74,
             "  起動を中止：テクスチャの解像度が上限（%dpx）を超えています" % TEXTURE_MAX_PX,
             "=" * 74, ""]
    for nm, w, h in bad[:8]:
        lines.append("   %5d x %-5d  %s" % (w, h, nm))
    if len(bad) > 8:
        lines.append("   ほか %d枚" % (len(bad) - 8))
    lines += ["",
              "  高解像度のまま走らせるとメモリを1プロセスあたり2.3GB余分に食います",
              "  （実測：モデルの読み込みが 2486MB → 142MB）。太郎の目は周辺128px・",
              "  中心窩208pxなので、512pxでも十分に細かい絵です。",
              "",
              "  直し方： %s の画像を512px以下に縮小してください。" % TEXTURE_DIR,
              "  （2500px の原本は MIMo/mimoEnv/assets/tex_original_2500px/ にあります）",
              "=" * 74, ""]
    raise SystemExit(chr(10).join(lines))



def _check_learning_happened(spec, out):
    """親が名前を言うはずの学習で、1回も言えていなければ**その場で止める**。

    【2026-08-28・なぜ機械で止めるか】板を顔から0.086m→0.30mへ離したところ、
    親が名前を言う条件（視線10度以内を3秒連続）を一度も満たせなくなり、
    **24本・約1時間40分ぶんの学習がまるごとゼロ**になった。しかも鎖の最後まで
    走り切ってから気づいた。1本目の終了時点で分かる情報だったのに、次の本へ
    そのまま進んでしまった。原因は親の手の速さ follow_speed が距離によらず
    0.5m/s 固定で、板が太郎の視線に追いつけなかったこと（実測：0.5で0回／
    1.75で8回）。CLAUDE.md「同じミスが2回起きたら文書ではなく機械で防ぐ」に従う。

    判定は「親がラベリングするはずの設定なのに語彙イベントが0」だけ。
    喃語フェーズ（親なし＝parent_labeling が無効か respond_prob=0）は対象外なので、
    既存の実験の挙動は1つも変わらない。
    """
    rep = (out or {}).get("word_learning")
    if not isinstance(rep, dict):
        return
    scene = {}
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sp = os.path.join(root, "run", "scenes", str(spec.get("scene")) + ".json")
        with open(sp, encoding="utf-8") as fp:
            scene = json.load(fp)
    except Exception:
        return   # シーンが読めないときは黙って通す（この検査のために止めない）
    pl = ((scene.get("world") or {}).get("parent_labeling")) or {}
    if not pl.get("enabled", True):
        return
    if float(pl.get("respond_prob", 1.0)) <= 0.0:
        return
    # 【2026-09-15】toy1/toy2 だけを足していたため、8択の場面（親が toy6〜toy11 を
    #   出す）で「親が一度も喋っていない」と誤判定し、走行を殺していた。
    #   word_learning は全スロットを数えているので、その合計を見る。
    #   古いログとの互換のため、合計が無ければ従来どおり toy1/toy2 を足す。
    n = 0
    if rep.get("label_count_合計") is not None:
        try:
            n = int(rep.get("label_count_合計") or 0)
        except (TypeError, ValueError):
            n = 0
    else:
        for k in ("label_count_toy1", "label_count_toy2"):
            try:
                n += int(rep.get(k) or 0)
            except (TypeError, ValueError):
                pass
    if n > 0:
        return
    lines = ["", "=" * 74,
             "  中止：親が一度も名前を言っていません（学習が成立していない）",
             "=" * 74, "",
             "   実験    %s" % spec.get("name", "(無名)"),
             "   シーン  %s" % (scene.get("name") or spec.get("scene")),
             "   語彙イベント 0件（label_count_toy1=0 / toy2=0）", "",
             "  このまま次へ進めても、学習していないモデルを渡すだけです。",
             "  よくある原因（2026-08-28に実際に踏んだもの）：",
             "",
             "   ・板の距離 follow_dist を変えたのに 親の手の速さ follow_speed が既定0.5のまま",
             "     → 板が太郎の視線に追いつかず、視線角度が悪化して注視条件を満たせない",
             "     （0.086m:0.5 の比を保つ。0.30m なら 1.75）",
             "   ・gaze_hold_sec（何秒連続で見たら名前を言うか）に注視が届いていない",
             "   ・板が視界に入っていない（シーンの配置・姿勢）",
             "",
             "  確かめ方：同じシーンで --steps 600 を1本だけ回し、",
             "  label_count が0でないことを見てから本走行に入ってください。",
             "=" * 74, ""]
    raise SystemExit(chr(10).join(lines))



def _check_toy_distance(spec):
    """板・おもちゃの距離が、文献で棄却された近さになっていたら**警告**する。

    【2026-08-28・なぜ入れるか】`doc/参考文献リスト.md` の「結論②：おもちゃ8cmは棄却／
    推奨25cm前後」は2026-07-20の調査で出ていたのに、実装に反映されないまま**1ヶ月以上
    0.086m で走らせ続けた**。8cmは成人の輻輳近点(8-10cm)相当で、調節要求12.5D（乳児の
    実力は1.0〜3.0D）、5cm角の玩具が視野の34.7%を占める。実際、両目が別々の場所を見て
    語の意味が混ざる原因になっていた（F2-15）。

    止めずに警告にとどめるのは、**近距離を意図的に試す実験がありうる**ため
    （触覚が届く距離での実験など）。研究上の選択は残し、うっかりだけを拾う。
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sp = os.path.join(root, "run", "scenes", str(spec.get("scene")) + ".json")
        with open(sp, encoding="utf-8") as fp:
            scene = json.load(fp)
    except Exception:
        return
    world = scene.get("world") or {}
    pl = world.get("parent_labeling") or {}
    checks = []
    if pl.get("follow_gaze") and pl.get("follow_dist") is not None:
        checks.append(("follow_dist（親が差し出す距離）", float(pl["follow_dist"])))
    for key in ("toy", "toy2"):
        t = world.get(key) or {}
        if t.get("enabled") and t.get("dist") is not None:
            checks.append(("%s.dist" % key, float(t["dist"])))
    near = [(n, v) for n, v in checks if v < 0.15]
    if not near:
        return
    print("")
    print("  " + "!" * 68)
    print("  注意：板・おもちゃが文献で棄却された近さです")
    for n, v in near:
        print("     %-28s %.3f m" % (n, v))
    print("     doc/参考文献リスト.md「結論②：おもちゃ8cmは棄却／推奨25cm前後」")
    print("     8cmは成人の輻輳近点(8-10cm)相当。調節要求12.5D（乳児の実力1.0〜3.0D）。")
    print("     両目が別々の場所を見て、語の意味が混ざる原因になります（F2-15で実測）。")
    print("     意図した近距離実験ならこのまま進めて構いません。")
    print("  " + "!" * 68)
    print("")


def _out_dir_of(spec):
    """この走行のログ置き場を実験ファイルから決める（2026-09-12）。

    実験ファイルは出力先を `run.csv` / `taro.save` / 各道具の `*_out` に
    バラバラに書くので、そのどれかの**フォルダ**を借りる（ほぼ常に
    `F/logs/<実験名>/` になっている）。1つも無ければ名前から作る。
    """
    cands = []
    r = spec.get("run") or {}
    t = spec.get("taro") or {}
    cands += [r.get("csv"), t.get("save")]
    for cfg in (spec.get("plugins") or {}).values():
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and (k.endswith("_out") or k.endswith("_dir")):
                    cands.append(v)
    for c in cands:
        if isinstance(c, str) and c.strip():
            d = os.path.dirname(c)
            if d:
                return d
    name = str(spec.get("name") or "無名").replace("/", "_").replace("\\", "_")
    goal = "F" if name.startswith("F") else "E"
    return os.path.join(goal, "logs", name)


def main():
    """コマンドライン引数（実験ファイルのパス・--steps・--verbose・--skip-preflight）を読み取り、テクスチャ解像度検査・プラグイン登録・ログ設定・おもちゃ距離の検査・走行前点検を順に行ってから run() を呼ぶ。点検で止めた場合は1を、それ以外は0を返す。
    """
    ap = argparse.ArgumentParser(description="シミュレーションシステムの入口")
    ap.add_argument("spec", help="実験ファイル（JSON）のパス")
    ap.add_argument("--steps", type=int, default=None, help="ステップ数を上書きする")
    ap.add_argument("--verbose", action="store_true", help="環境構築のログも出す")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="走行前点検を飛ばす（事故の分かっている場合だけ）")
    a = ap.parse_args()
    _check_texture_resolution()
    _register()
    spec = load_spec(a.spec)
    # 【2026-09-12・ユーザー指示】「エラーログは一つにまとめてファイル化」
    #   「実行中はエラーログを絶対読むようにしよう」。
    #   この走行のフォルダに エラー.log／走行.log／未実装.log を作る。
    #   画面の一番下にエラー.logの中身を出す（最後に呼ぶ log_tail）ので、
    #   読み忘れが注意力ではなく**位置**で防がれる。詳細は run/log_setup.py
    from run.log_setup import setup_logging, log_tail
    log_paths = setup_logging(_out_dir_of(spec))
    _check_toy_distance(spec)
    # 【2026-09-11・ユーザー指示】構造のミスは走る前に止める。走行後に発覚して
    #   1本（9〜15分）を捨てるのを避ける。詳細は run/tools/preflight.py
    from run.tools.preflight import run_check
    if not run_check(spec, a.spec, skip=a.skip_preflight):
        log_tail(log_paths)
        return 1
    try:
        run(spec, steps_override=a.steps, verbose=a.verbose)
    except KeyboardInterrupt:
        raise
    except BaseException as _e:
        # 先に記録してから finally へ。excepthook は finally の**後**に発火するので、
        # ここで書かないと log_tail が「エラー.log：空」と嘘を言う（2026-09-12に実測）
        from run.log_setup import log_crash
        log_crash(_e)
        raise
    finally:
        # 落ちても必ずエラー.logを画面の最後に出す
        log_tail(log_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
