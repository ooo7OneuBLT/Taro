# -*- coding: utf-8 -*-
"""保持の制動を変えた走行を、基準と並べて判定する。

使い方: python F/scripts/f104_hold_damp_compare.py <基準の名前> <比較の名前> ...
例    : python F/scripts/f104_hold_damp_compare.py F2-116pre_保持を無条件にする F2-119pre_制動0.04

見る数字（advisor と決めた順）:
  1. 保持中の速さ（中央値）      … 下がれば制動が効いている
  2. 小さい命令(<2度)の符号一致  … 上がれば本題が改善
  3. 着地時の速度                … 変わらないはず（保持はサッケード中に効かない）
  4. 下流4指標                    … 見えていた率・照合回数・親の発話・撃った数
"""
import json, sys, io, os
import numpy as np

def load(name):
    d = "F/logs/%s" % name
    a = np.array(json.load(io.open(d+"/眼球の角度_毎ステップ.json", encoding="utf-8")), dtype=float)
    r = json.load(io.open(d+"/サッケード1発ごと.json", encoding="utf-8"))
    return d, a, r

def segments(mv):
    seg, s = [], None
    for i in range(len(mv)):
        if mv[i] == 0 and s is None: s = i
        elif mv[i] != 0 and s is not None: seg.append((s, i)); s = None
    return seg

def metrics(name):
    d, a, r = load(name)
    t, h, v, mv = a[:,0], a[:,1], a[:,2], a[:,3]
    dt = np.median(np.diff(t))
    vel = np.gradient(h, dt)
    seg = segments(mv)
    idx = np.concatenate([np.arange(s+2, e-2) for s, e in seg if e-s > 6])
    sp = np.abs(vel[idx])
    out = {"保持中の速さ": np.median(sp), "静止(<1.5度/秒)の割合": 100*(sp < 1.5).mean()}
    flips = np.concatenate([np.diff(np.sign(vel[s:e])) != 0 for s, e in seg if e-s > 4])
    out["速度の符号反転[回/秒]"] = flips.mean()/dt
    # 着地時の速度・符号一致
    land, agree_s, agree_all, amps = [], [], [], []
    for x in r:
        cx, cy = x["tgt_h"]-x["h0"], x["tgt_v"]-x["v0"]
        amp = np.hypot(cx, cy)
        if amp < 1e-6: continue
        i = int(round(x["t_end"]/dt))
        if 2 <= i < len(t)-1:
            land.append(np.hypot((h[i]-h[i-2])/(2*dt), (v[i]-v[i-2])/(2*dt)))
        ax, ay = x["h1"]-x["h0"], x["v1"]-x["v0"]
        ok = 1.0 if (ax*cx + ay*cy) > 0 else 0.0
        agree_all.append(ok); amps.append(amp)
        if amp < 2.0: agree_s.append(ok)
    out["着地時の速度"] = np.median(land) if land else float("nan")
    out["小命令(<2度)の符号一致[%]"] = 100*np.mean(agree_s) if agree_s else float("nan")
    out["全命令の符号一致[%]"] = 100*np.mean(agree_all)
    out["撃った数"] = len(r)
    out["命令の中央値[度]"] = np.median(amps)
    # 下流
    p = d + "/物体ファイル.csv"
    if os.path.exists(p):
        import csv
        rows = list(csv.DictReader(io.open(p, encoding="utf-8-sig")))
        if rows:
            mt = [x for x in rows if x.get("event") == "match"]
            out["照合できた回数"] = len(mt)
    p = d + "/発話イベント.csv"
    if os.path.exists(p):
        import csv
        out["親の発話"] = len(list(csv.DictReader(io.open(p, encoding="utf-8-sig"))))
    return out

def main():
    names = sys.argv[1:]
    if len(names) < 2:
        print(__doc__); return
    res = [(n, metrics(n)) for n in names]
    keys = list(res[0][1].keys())
    w = max(len(k) for k in keys) + 1
    print("\n" + " "*w + "".join("%22s" % n[:21] for n, _ in res))
    print("-"*(w + 22*len(res)))
    for k in keys:
        line = "%-*s" % (w, k)
        base = res[0][1].get(k)
        for i, (n, m) in enumerate(res):
            x = m.get(k, float("nan"))
            s = "%.1f" % x if isinstance(x, float) else str(x)
            if i > 0 and isinstance(x, float) and isinstance(base, float) and base:
                s += " (%+.0f%%)" % (100*(x/base - 1))
            line += "%22s" % s
        print(line)
    print()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
