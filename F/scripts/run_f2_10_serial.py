# -*- coding: utf-8 -*-
"""F2-10（耳の長音展開）の再学習12本＋産出テスト4本を、1本ずつ順に回す。

同時起動はしない（2026-08-26に4本同時でメモリ不足＝Could not allocate memory
が起きたため。CLAUDE.md「同じミスが2回起きたら機械で防ぐ」）。
1本でも失敗したらそこで止める（モデルを受け渡す連鎖なので後続は無意味）。

  python F/scripts/run_f2_10_serial.py
"""
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ORDER = [
    "F2-10_ABAB_A1", "F2-10_ABAB_B1", "F2-10_ABAB_A2", "F2-10_ABAB_B2",
    "F2-10_ABAB2_A1", "F2-10_ABAB2_B1", "F2-10_ABAB2_A2", "F2-10_ABAB2_B2",
    "F2-10_ABAB3_A1", "F2-10_ABAB3_B1", "F2-10_ABAB3_A2", "F2-10_ABAB3_B2",
    "F2-10_言わせる_わんわん", "F2-10_言わせる_ぶーぶー",
    "F2-10_言わせる_りんご", "F2-10_言わせる_くつ",
]
PROG = os.path.join(ROOT, "F", "logs", "F2-10_進捗.log")


def note(msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(PROG, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def main():
    # 引数を渡すとその名前だけを、渡した順に回す（中断からの再開に使う）。
    #   例: python F/scripts/run_f2_10_serial.py F2-10_言わせる_ぶーぶー F2-10_言わせる_りんご
    global ORDER
    ORDER = sys.argv[1:] or ORDER
    os.makedirs(os.path.dirname(PROG), exist_ok=True)
    note("=== F2-10 開始（%d本を1本ずつ）===" % len(ORDER))
    t_all = time.time()
    for i, name in enumerate(ORDER, 1):
        spec = os.path.join(ROOT, "F", "experiments", "%s_2026-08-27.json" % name)
        if not os.path.exists(spec):
            note("[%d/%d] %s: 実験ファイルが無い → 中止" % (i, len(ORDER), name))
            return 1
        note("[%d/%d] %s 開始" % (i, len(ORDER), name))
        t0 = time.time()
        log = os.path.join(ROOT, "F", "logs", "F2-10_console_%s.log" % name)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        with open(log, "w", encoding="utf-8", errors="replace") as fp:
            r = subprocess.run([sys.executable, os.path.join(ROOT, "run", "main.py"), spec],
                               cwd=ROOT, stdout=fp, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0
        if r.returncode != 0:
            note("[%d/%d] %s 失敗(code=%d, %.1f分) → 中止。ログ: %s"
                 % (i, len(ORDER), name, r.returncode, dt / 60, log))
            return 1
        note("[%d/%d] %s 完了 (%.1f分)" % (i, len(ORDER), name, dt / 60))
    note("=== 全16本 完了 (合計 %.1f分) ===" % ((time.time() - t_all) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
