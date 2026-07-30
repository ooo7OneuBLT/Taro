"""シミュレーションシステム ── 実験・学習・目視の★唯一の入口。

【なぜ作ったか、2026-07-30】目標Eの実験スクリプトが118本あり、そのうち66本が
**それぞれ独立に環境を組み立てていた**。そのため「学習は関節モード、測定・Viewerは
筋肉モード」という別の体で動いていた事故が起きた（ユーザーの目視で発覚。実測で
動きが人間の新生児の約3.3倍速かった）。
設計の全体像は `E/docs/実行基盤_設計.md`。

【使い方】
    .venv/Scripts/python.exe -m run.main E/experiments/<名前>.json
    .venv/Scripts/python.exe -m run.main E/experiments/<名前>.json --steps 600

【実験ファイル（JSON）の3つの欄】★境界を混ぜない
    scene    どんな環境か（E/scenes/*.json の名前）
    taro     ★太郎の中身の設定（本能のON/OFF・月齢・駆動モード）
             → 実装は taro_core にある。ここは「どれを使うか」だけ
    run      動かし方（type / steps / seed）
    plugins  ★外から測る・見る道具（太郎を変えない）

⚠️env を直接作るコードを**新しく書かない**。ここを通す。
  （検査は run/tools/check_entry.py）
"""
import argparse
import json
import os
import sys
import time
import warnings

# ★★【最重要の未解決事項・2026-07-30】**同じ実験ファイル・同じ seed でも結果がばらつく。**
#   【実測】seed=0・100回学習・自己モデルの測定ONで7回回した classify（100回目）：
#       45.6 / 46.7 / 44.5 / 45.7 / 47.0 / 43.9 / 45.6  ＝★幅3.1ポイント
#     元の経路（E/scripts/e_growth_train.py）でも同じ（49.3 → 43.2）。
#     ＝**私の組み換えのせいではなく、元からある**。
#   【切り分けで分かったこと】
#     ・物理（MuJoCo）は決定的        脳を通さない measure は2回**完全一致**
#     ・学習だけなら決定的            測る道具を外すと2回**完全一致**
#     ・★自己モデルの測定を入れると出る  e_probes.evaluate（環境を80判断進める）
#     ・PYTHONHASHSEED も BLAS のスレッド数も**原因ではない**（固定しても一致しない）
#   【まだ分かっていないこと】どこで非決定性が生まれるのか。
#   【★実験するときの守り】「シード間の差」を見る前に、**同じシードを複数回**回して
#     シード内のばらつきを測る。それより小さい差は主張できない。
#     → 検証の落とし穴チェックリスト 項79

warnings.filterwarnings("ignore")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ★プラグインの登録表。実験ファイルのキーとクラスの対応はここだけ。
#   新しい測る道具を作ったら、ここに1行足す。
PLUGINS = {}


def _register():
    """プラグインを登録する。import が重いものは使うときだけ読む。"""
    from run.plugins.common.toy_touch import ToyTouch
    from run.plugins.common.hand_in_view import HandInView
    from run.plugins.common.self_model import SelfModel
    PLUGINS["toy_touch"] = ToyTouch
    PLUGINS["hand_in_view"] = HandInView
    PLUGINS["self_model"] = SelfModel
    # ⚠️self_model は★太郎の脳が要る（run.type=train）。脳側との接続は第2段階。
    #   いまの train は e_growth_train を包んでいるので、測定はそちらで行われる。


def load_spec(path):
    """実験ファイルを読む。★書き間違いをここで止める。"""
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
    """実験ファイルの plugins 欄から、使う道具を組み立てる。"""
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
    """チェックポイントの記録をCSVに残す関数を作る。★列は最初の行で決まる。"""
    import csv
    if not path:
        return (lambda row: None)
    path = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"cols": None}

    def log_row(row):
        if not isinstance(row, dict):
            return
        new = state["cols"] is None
        if new:
            state["cols"] = list(row)
            with open(path, "w", newline="", encoding="utf-8") as fp:
                csv.writer(fp).writerow(state["cols"])
        with open(path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([row.get(c, "") for c in state["cols"]])
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

    # --- ★train：太郎の脳を通して学習する（run/trainer.py）------------------
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
        return out

    # --- view：等倍速で太郎を見る（run/viewer.py）----------------------------
    if kind == "view":
        from run.config import Config
        from run import viewer
        cfg = Config.from_spec(spec, steps_override=steps_override)
        print(f"  設定     {cfg.summary()}")
        return viewer.view(cfg, plugins=plugins, verbose=verbose)

    if kind not in ("measure",):
        raise ValueError(f"run.type が不明: {kind}（measure / train / view）")

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
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description="シミュレーションシステムの入口")
    ap.add_argument("spec", help="実験ファイル（JSON）のパス")
    ap.add_argument("--steps", type=int, default=None, help="ステップ数を上書きする")
    ap.add_argument("--verbose", action="store_true", help="環境構築のログも出す")
    a = ap.parse_args()
    _register()
    spec = load_spec(a.spec)
    run(spec, steps_override=a.steps, verbose=a.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
