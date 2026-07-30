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
    PLUGINS["toy_touch"] = ToyTouch
    PLUGINS["hand_in_view"] = HandInView
    # ⚠️view（目視）と self_model は第2段階で足す（設計 §7）


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

    env, sc, _hands = scene_mod.build(spec["scene"], taro=spec["taro"],
                                      seed=seed, verbose=verbose)
    u = env.unwrapped
    # 1ステップの秒数。K は「1判断あたりの物理ステップ数」（学習ループと同じ既定10）
    K = int(r.get("K", 10))
    dt = float(u.model.opt.timestep) * int(u.frame_skip) * K

    plugins = build_plugins(spec)
    ctx = Ctx(env=env, spec=spec, scene=sc, n_steps=n_steps, dt=dt)
    for p in plugins:
        p.setup(ctx)

    if kind not in ("measure",):
        raise NotImplementedError(
            f"run.type={kind} はまだ（第1段階は measure だけ）。"
            "学習は第2段階で取り込む＝ E/docs/実行基盤_設計.md §7")

    # --- measure：太郎の脳を通さず、環境をそのまま進めて測る ---
    #   ⚠️脳を通す（学習する／学習済みモデルで動かす）のは第2段階。
    #     いまは「環境と測る道具が正しく繋がっているか」を確かめる最小版。
    import numpy as np
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
