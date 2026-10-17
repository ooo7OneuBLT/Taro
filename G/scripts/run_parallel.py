"""スミスとユーの実験（core_v2/tools/smith_yu.py）を、種を分けて何本か同時に回し、最後にまとめる。

【2026-10-17新設・軽量化】1本ずつだと1人約3.3分。2本同時に回すと、同じ種の結果は1本ずつのときと
1ビットも変わらず（種0・1で確かめた。G/logs/並列確かめ_2026-10-17）、2人で約5.0分（1本ずつなら約6.6分）。
同時に回せるのは、MIMo の一時ファイル名にプロセスIDが入っているから（MIMo/mimoGrowth/scene.py の
temp_path、2026-10-16 再適用）。

ハードウェアの見張り（起動の前と、回している間ずっと）：
* GPU のメモリ：1本あたり約0.5GB（実測）。空きが --min-gpu-free-mb（既定1500）を切ったら、新しい本を起動しない
* 空きメモリ（RAM）：1本あたり約1GB（実測）。空きが --min-ram-free-gb（既定4）を切ったら、新しい本を起動しない
* GPU の使用率がもともと高い（ほかのアプリが使っている）ときは、同時に回しても速くならない。
  起動の前に、ほかに GPU を使うアプリ（Unity・ゲームなど）が開いていないかを表示する
各本のログは <log-dir>/part_<種の範囲>/ に分けて書く（まとめ.json の上書きを避ける）。
【再開】同じ --log-dir でもう一度起動すると、結果のある (覚え方, 種) をとばして残りだけ回す（途中で止まっても、やり直しにならない）。
最後に全部の <mode>_seed*.json から smith_yu.summarize で <log-dir>/まとめ.json を作る。

使い方:
  python G/scripts/run_parallel.py --seeds 0-19 --modes B,none --workers 2 --log-dir 段階6_並列 [smith_yu の他の引数]
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)


def gpu_state():
    """(使用中MB, 合計MB, 使用率%)。nvidia-smi が無ければ None。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        used, total, util = (int(x) for x in out.strip().splitlines()[0].split(","))
        return used, total, util
    except Exception:  # noqa: BLE001
        return None


def ram_free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:  # noqa: BLE001
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                                 capture_output=True, text=True, timeout=20).stdout
            return int(out.strip()) / 2 ** 20
        except Exception:  # noqa: BLE001
            return None


def other_gpu_apps():
    """GPU を使いそうな重いアプリが開いているか（Windows だけ。名前で見るだけ）。"""
    names = ("Unity", "steam", "EpicGamesLauncher", "obs64", "blender", "chrome")
    try:
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=20).stdout
        seen = sorted({ln.split(",")[0].strip('"') for ln in out.splitlines()
                       if any(ln.strip('"').lower().startswith(n.lower()) for n in names)})
        return seen
    except Exception:  # noqa: BLE001
        return []


def split_seeds(spec: str, n: int) -> list[str]:
    from core_v2.tools.smith_yu import parse_seeds
    seeds = list(parse_seeds(spec))
    parts = [seeds[i::n] for i in range(n)]
    return [",".join(str(s) for s in p) for p in parts if p]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    def take(name, default):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return type(default)(v)
        return default
    seeds = take("--seeds", "0-19")
    workers = take("--workers", 2)
    log_dir = take("--log-dir", "並列")
    min_gpu_free = take("--min-gpu-free-mb", 1500)
    min_ram_free = take("--min-ram-free-gb", 4.0)
    rest = argv                                  # smith_yu へそのまま渡す（--modes など）
    root = os.path.join(REPO, "G", "logs", log_dir)
    os.makedirs(root, exist_ok=True)

    g = gpu_state()
    apps = other_gpu_apps()
    print(f"[並列] GPU: {g}（使用中MB, 合計MB, 使用率%）  空きRAM: {ram_free_gb():.1f}GB", flush=True)
    if apps:
        print(f"[並列] 注意：GPU を使いそうなアプリが開いている → {apps}。閉じると速くなる", flush=True)
    if g and g[2] >= 90:
        print("[並列] 注意：まだ何も回していないのに GPU の使用率が90%以上。同時に回しても速くならない", flush=True)

    # 【再開できるように】もう結果のある (覚え方, 種) はとばす。覚え方ごとに残りの種を workers 本に分け、
    #   同時に回すのは最大 workers 本（終わった本の後に次を起動する）
    modes = "B,A"
    if "--modes" in rest:
        i = rest.index("--modes")
        modes = rest[i + 1]
        rest = rest[:i] + rest[i + 2:]
    from core_v2.tools.smith_yu import parse_seeds
    done = {os.path.basename(f) for f in glob.glob(os.path.join(root, "**", "*_seed*.json"), recursive=True)}
    jobs = []
    for mode in modes.split(","):
        left = [s for s in parse_seeds(seeds) if f"{mode}_seed{s}.json" not in done]
        skipped = len(parse_seeds(seeds)) - len(left)
        if skipped:
            print(f"[並列] {mode}：もう結果のある {skipped} 人はとばす", flush=True)
        for k in range(workers):
            part = left[k::workers]
            if part:
                jobs.append((mode, k, ",".join(str(s) for s in part)))
    procs, logs = [], []
    t0 = time.time()
    peak = [0, 0, 101.0]

    def watch():
        g = gpu_state()
        ram = ram_free_gb()
        if g:
            peak[0] = max(peak[0], g[0]); peak[1] = max(peak[1], g[2])
        if ram is not None:
            peak[2] = min(peak[2], ram)
        return g, ram
    for mode, k, part in jobs:
        while sum(p.poll() is None for p in procs) >= workers:
            watch()
            time.sleep(10)
        while True:
            g, ram = watch()
            ok_gpu = g is None or (g[1] - g[0]) >= min_gpu_free
            ok_ram = ram is None or ram >= min_ram_free
            if ok_gpu and ok_ram:
                break
            print(f"[並列] 起動を待つ（GPU空き {None if g is None else g[1] - g[0]}MB・RAM空き {ram}GB）", flush=True)
            time.sleep(15)
        sub = f"{log_dir}/part_{mode}_{k}"
        cmd = [sys.executable, "-m", "core_v2.tools.smith_yu", "--seeds", part, "--modes", mode,
               "--log-dir", sub] + rest
        lf = open(os.path.join(root, f"part_{mode}_{k}.log"), "a", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=env))
        logs.append(lf)
        print(f"[並列] 起動 {mode}_{k}: 種 {part}（pid {procs[-1].pid}）", flush=True)
        time.sleep(20)                           # 準備（DINOv2 の読み込み）が重なりすぎないように
    while any(p.poll() is None for p in procs):
        watch()
        time.sleep(10)
    for lf in logs:
        lf.close()
    codes = [p.returncode for p in procs]
    print(f"[並列] 終わり：{time.time() - t0:.0f}秒  終了コード {codes}  "
          f"GPUメモリ最大 {peak[0]}MB・使用率最大 {peak[1]}%・空きRAM最小 {peak[2]:.1f}GB", flush=True)

    from core_v2.tools.smith_yu import summarize
    results = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*_seed*.json"), recursive=True)):
        with open(f, encoding="utf-8") as fp:
            results.append(json.load(fp))
    summary = summarize(results)
    with open(os.path.join(root, "まとめ.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "workers": workers, "exit_codes": codes,
                   "runs": [{k: r[k] for k in ("seed", "mode", "target_share", "trials_target_longer",
                                               "trials_distractor_longer", "n_learn")} for r in
                            sorted(results, key=lambda r: (r["mode"], r["seed"]))]},
                  f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
