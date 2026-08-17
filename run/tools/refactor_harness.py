"""設定配線の大改修（シーンJSON→環境構築、`run/scene_tools/e_scene.py` 363〜411行の
環境変数24個＋モジュール定数上書き15個を「設定オブジェクト1個」へ束ねる改修）に先立つ
bit-identical検証ハーネス。

【なぜ要るか】改修の各段階で挙動が1ビットも変わっていないことを機械的に確認するため。
人間が目視で比べるのではなく、ハッシュの一致/不一致で機械的に判定する。

【対象】`run/scenes/*.json` の現役シーン全部（`_archive` はサブフォルダなので
`*.json` の単純globには元々含まれない）。実行時にディレクトリを都度globするので、
シーンが増減しても数を数え直す必要はない。

【設定マトリクス】
    (1) 素のmeasure          taro={"actuation": "muscle"}
    (2) 反射あり              taro={"actuation": "muscle", "vision": true,
                                    "orienting_reflex": true}
組み立ては `run/plugins/common/scene.py` の `build()` 経由（実験本番と同じ経路）。

【1組み合わせあたりの記録】
    - 10ステップごと（0,10,20,...,200）のqpos全体のsha256
      （float64を np.round(qpos, 10) してから tobytes() でハッシュ）
    - 最終ステップ(200)のqpos/qvel先頭20要素の生値
      （ハッシュ不一致のとき、差の中身を見るため）
    - 環境構築直後の os.environ のうち E_ で始まる変数のスナップショット
      （配線が変わっていないかの裏取り）
seed=0・行動ゼロ（zero action）・200ステップ。

【なぜサブプロセスで回すか】MuJoCoのモデルは同一プロセス内で作り直すと状態を
引きずる場合がある。1回のビルド＝1回のプロセス起動という最も安全な粒度にし、
シーン単位どころか「シーン×設定」単位でプロセスを分ける（実装担当判断）。

使い方::

    .venv/Scripts/python.exe run/tools/refactor_harness.py baseline   # 現状の挙動を記録
    .venv/Scripts/python.exe run/tools/refactor_harness.py compare    # 現状とベースラインを突き合わせ

ベースラインは `run/tools/_refactor_baseline/{シーン名}_{設定番号}.json` に保存する。
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
SCENES_DIR = os.path.join(_ROOT, "run", "scenes")
BASELINE_DIR = os.path.join(_HERE, "_refactor_baseline")

SEED = 0
N_STEPS = 200
HASH_EVERY = 10

# 設定マトリクス（設定番号, taro辞書）。設定番号はベースラインのファイル名に使う。
CONFIGS = [
    (1, {"actuation": "muscle"}),
    (2, {"actuation": "muscle", "vision": True, "orienting_reflex": True}),
]


def _list_scenes():
    paths = glob.glob(os.path.join(SCENES_DIR, "*.json"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


# ============================================================================
# 子プロセス側（1回のビルド＋200step進行だけを行い、結果をJSONへ書く）
# ============================================================================
def _run_child(request_path):
    warnings.filterwarnings("ignore")
    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    scene_name = req["scene"]
    cfg = req["config"]
    cfg_idx = req["config_index"]
    seed = req["seed"]
    n_steps = req["n_steps"]
    hash_every = req["hash_every"]
    out_path = req["out_path"]

    result = dict(scene=scene_name, config_index=cfg_idx, config=cfg,
                  seed=seed, n_steps=n_steps, hash_every=hash_every)
    try:
        import numpy as np
        from run.plugins.common import scene as scene_mod

        env, sc, hands = scene_mod.build(scene_name, taro=dict(cfg), seed=seed,
                                         verbose=False, hybrid=False)
        u = env.unwrapped
        d = u.data
        # 組み立て直後（0step進める前）のE_環境変数スナップショット
        env_snapshot = {k: os.environ[k] for k in sorted(os.environ)
                        if k.startswith("E_")}
        zero = np.zeros(env.action_space.shape[0], dtype=np.float32)

        def _hash():
            arr = np.round(np.asarray(d.qpos, dtype=np.float64), 10)
            return hashlib.sha256(arr.tobytes()).hexdigest()

        qpos_hashes = {"0": _hash()}
        for t in range(1, n_steps + 1):
            env.step(zero)
            if t % hash_every == 0:
                qpos_hashes[str(t)] = _hash()

        final_qpos_head20 = np.asarray(d.qpos, dtype=np.float64)[:20].tolist()
        final_qvel_head20 = np.asarray(d.qvel, dtype=np.float64)[:20].tolist()
        env.close()

        result.update(ok=True, env_snapshot=env_snapshot, qpos_hashes=qpos_hashes,
                      final_qpos_head20=final_qpos_head20,
                      final_qvel_head20=final_qvel_head20)
    except Exception as e:
        import traceback
        result.update(ok=False, error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc())

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ============================================================================
# 親プロセス側
# ============================================================================
def _child_env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "2"
    return env


def _run_one(scene_name, cfg_idx, cfg):
    """1つの (シーン, 設定) をサブプロセスで実行し、結果の辞書を返す。"""
    fd, req_path = tempfile.mkstemp(suffix="_req.json", prefix="refactor_harness_")
    os.close(fd)
    out_path = req_path[:-9] + "_out.json"   # "_req.json" -> "_out.json"
    req = dict(scene=scene_name, config_index=cfg_idx, config=cfg,
               seed=SEED, n_steps=N_STEPS, hash_every=HASH_EVERY, out_path=out_path)
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(req, f, ensure_ascii=False)

    try:
        r = subprocess.run([PY, os.path.abspath(__file__), "_child", req_path],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=600, env=_child_env())
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                result = json.load(f)
        else:
            tail_out = (r.stdout or "")[-3000:]
            tail_err = (r.stderr or "")[-3000:]
            result = dict(scene=scene_name, config_index=cfg_idx, config=cfg, ok=False,
                          error=f"子プロセスが出力ファイルを作らなかった"
                                f"(returncode={r.returncode})",
                          stdout_tail=tail_out, stderr_tail=tail_err)
    except subprocess.TimeoutExpired:
        result = dict(scene=scene_name, config_index=cfg_idx, config=cfg, ok=False,
                      error="子プロセスがタイムアウト(600秒)")
    finally:
        for p in (req_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
    return result


def _baseline_path(scene_name, cfg_idx):
    return os.path.join(BASELINE_DIR, f"{scene_name}_{cfg_idx}.json")


# ============================================================================
# baseline
# ============================================================================
def cmd_baseline():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    scenes = _list_scenes()
    total = len(scenes) * len(CONFIGS)
    print(f"対象シーン数: {len(scenes)}  設定数: {len(CONFIGS)}  合計: {total}件")
    print("-" * 88)

    n_ok = 0
    n_fail = 0
    failures = []
    for scene_name in scenes:
        for cfg_idx, cfg in CONFIGS:
            print(f"[baseline] {scene_name} 設定{cfg_idx} ...", flush=True)
            result = _run_one(scene_name, cfg_idx, cfg)
            with open(_baseline_path(scene_name, cfg_idx), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if result.get("ok"):
                n_ok += 1
                last_hash = result["qpos_hashes"].get(str(N_STEPS), "?")
                print(f"  OK   最終step({N_STEPS})のqposハッシュ={last_hash[:16]}...")
            else:
                n_fail += 1
                failures.append((scene_name, cfg_idx, result.get("error")))
                print(f"  FAIL {result.get('error')}")

    print("-" * 88)
    print(f"生成件数: OK={n_ok}  FAIL={n_fail}  合計={n_ok + n_fail}/{total}")
    if failures:
        print()
        print("失敗一覧:")
        for s, c, e in failures:
            print(f"  {s} 設定{c}: {e}")
    return 0 if n_fail == 0 else 1


# ============================================================================
# compare
# ============================================================================
def _compare_result(base, cur):
    """(status, detail文字列) を返す。status は 'OK' か 'NG'。"""
    if not base.get("ok") and not cur.get("ok"):
        return "OK", (f"(両方ビルド失敗のまま一致: base={base.get('error')} / "
                      f"cur={cur.get('error')})")
    if bool(base.get("ok")) != bool(cur.get("ok")):
        return "NG", (f"ビルド成否が変化: base_ok={base.get('ok')} cur_ok={cur.get('ok')}"
                      f"  base_err={base.get('error')}  cur_err={cur.get('error')}")
    if not cur.get("ok"):
        return "NG", f"ビルド失敗: {cur.get('error')}"

    parts = []

    # ---- qposハッシュの列（10stepごと）を先頭から突き合わせ、最初の不一致を探す ----
    bh = base.get("qpos_hashes", {})
    ch = cur.get("qpos_hashes", {})
    steps = sorted(set(bh) | set(ch), key=lambda x: int(x))
    first_bad_step = None
    for s in steps:
        if bh.get(s) != ch.get(s):
            first_bad_step = s
            break

    if first_bad_step is not None:
        parts.append(f"qposハッシュがstep={first_bad_step}で最初に不一致"
                     f"(base={str(bh.get(first_bad_step))[:12]}..."
                     f" cur={str(ch.get(first_bad_step))[:12]}...)")
        bq, cq = base.get("final_qpos_head20"), cur.get("final_qpos_head20")
        bv, cv = base.get("final_qvel_head20"), cur.get("final_qvel_head20")
        if bq and cq:
            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(bq, cq)) if a != b]
            if diffs:
                sample = "; ".join(f"qpos[{i}] base={a:.10f} cur={b:.10f}"
                                   for i, a, b in diffs[:5])
                parts.append(f"最終step({N_STEPS})qpos先頭20要素の差(先頭5件): {sample}")
            else:
                parts.append(f"（最終step({N_STEPS})のqpos先頭20要素は一致。"
                             f"途中stepでのみ差がある）")
        if bv and cv:
            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(bv, cv)) if a != b]
            if diffs:
                sample = "; ".join(f"qvel[{i}] base={a:.10f} cur={b:.10f}"
                                   for i, a, b in diffs[:5])
                parts.append(f"最終step({N_STEPS})qvel先頭20要素の差(先頭5件): {sample}")

    # ---- E_環境変数スナップショットの差（配線そのものの裏取り）----
    be = base.get("env_snapshot", {})
    ce = cur.get("env_snapshot", {})
    env_diff = [f"{k}: base={be.get(k)!r} cur={ce.get(k)!r}"
               for k in sorted(set(be) | set(ce)) if be.get(k) != ce.get(k)]
    if env_diff:
        parts.append("E_環境変数の差: " + "; ".join(env_diff[:8]))

    if not parts:
        return "OK", ""
    return "NG", " / ".join(parts)


def cmd_compare():
    if not os.path.isdir(BASELINE_DIR):
        print(f"ベースラインが無い: {BASELINE_DIR}\n"
              f"先に `python run/tools/refactor_harness.py baseline` を実行すること。")
        return 1

    scenes = _list_scenes()
    total = len(scenes) * len(CONFIGS)
    print(f"対象シーン数: {len(scenes)}  設定数: {len(CONFIGS)}  合計: {total}件")
    print("-" * 88)

    rows = []
    n_match = 0
    n_mismatch = 0
    n_baseline_missing = 0
    for scene_name in scenes:
        for cfg_idx, cfg in CONFIGS:
            base_path = _baseline_path(scene_name, cfg_idx)
            if not os.path.isfile(base_path):
                rows.append((scene_name, cfg_idx, "NG", "ベースラインファイルが無い"))
                n_baseline_missing += 1
                continue
            with open(base_path, encoding="utf-8") as f:
                base = json.load(f)
            print(f"[compare] {scene_name} 設定{cfg_idx} ...", flush=True)
            cur = _run_one(scene_name, cfg_idx, cfg)
            status, detail = _compare_result(base, cur)
            rows.append((scene_name, cfg_idx, status, detail))
            if status == "OK":
                n_match += 1
            else:
                n_mismatch += 1
            print(f"  {status}  {detail}")

    print()
    print("=" * 110)
    print(f"{'シーン':<42}{'設定':<6}{'結果':<6}詳細")
    print("=" * 110)
    for scene_name, cfg_idx, status, detail in rows:
        print(f"{scene_name:<42}{cfg_idx:<6}{status:<6}{detail}")
    print("=" * 110)
    print(f"一致={n_match}  不一致={n_mismatch}  ベースライン無し={n_baseline_missing}"
          f"  合計={len(rows)}/{total}")
    return 0 if (n_mismatch == 0 and n_baseline_missing == 0) else 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "_child":
        _run_child(sys.argv[2])
        return 0
    if len(sys.argv) < 2 or sys.argv[1] not in ("baseline", "compare"):
        print(__doc__)
        return 1
    if sys.argv[1] == "baseline":
        return cmd_baseline()
    return cmd_compare()


if __name__ == "__main__":
    sys.exit(main())
