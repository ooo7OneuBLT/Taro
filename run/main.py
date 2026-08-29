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
    from run.plugins.common.word_production import WordProduction
    from run.plugins.common.babble_probe import BabbleProbe
    from run.plugins.common.gaze_probe import GazeProbe
    from run.plugins.common.produce_snapshot import ProduceSnapshot
    from run.plugins.common.vergence_probe import VergenceProbe
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
    PLUGINS["word_production"] = WordProduction   # 太郎自身の発話（見た物の名前を言う、F2）
    PLUGINS["babble_probe"] = BabbleProbe   # 喃語モードの発話・帳面の成長を記録する（F2-1）
    PLUGINS["gaze_probe"] = GazeProbe   # 視線が的にどれだけ連続で留まるか（F2-9の判定設計用）
    PLUGINS["produce_snapshot"] = ProduceSnapshot   # 発話した瞬間の中心窩画像と視覚ベクトル（F2-11のりんご偏り究明）
    PLUGINS["vergence_probe"] = VergenceProbe   # 輻輳角の目標と実測（F2-15の切り分け）
    # 注意：self_model / trace / reach_success / double_touch は太郎の脳が要る
    #   （run.type=train のみ）。measure（脳を通さず環境だけ進める）では使えない。
    #   reach_success・double_touch はさらに taro.goal_space="reach_self" も要る。


def load_spec(path):
    """実験ファイルを読む。書き間違いをここで止める。"""
    with open(path, encoding="utf-8") as fp:
        spec = json.load(fp)
    known = {"name", "note", "scene", "taro", "run", "plugins"}
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


def run(spec, *, steps_override=None, verbose=False):
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
                                      seed=seed, verbose=verbose)
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
    n = 0
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


def main():
    ap = argparse.ArgumentParser(description="シミュレーションシステムの入口")
    ap.add_argument("spec", help="実験ファイル（JSON）のパス")
    ap.add_argument("--steps", type=int, default=None, help="ステップ数を上書きする")
    ap.add_argument("--verbose", action="store_true", help="環境構築のログも出す")
    a = ap.parse_args()
    _check_texture_resolution()
    _register()
    spec = load_spec(a.spec)
    _check_toy_distance(spec)
    run(spec, steps_override=a.steps, verbose=a.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
