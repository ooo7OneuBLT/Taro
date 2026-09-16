# -*- coding: utf-8 -*-
"""YCB Object Set（CC-BY 4.0）から選んだ物を取得する（F2-49・2026-09-02）。

    .venv/Scripts/python.exe F/scripts/f_ycb_fetch.py
保存先: F/assets/ycb/<id>/google_16k/（textured.obj, texture_map.png など）
"""
import os, sys, io, json, tarfile, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")
SELECT = {
    "ボール": ["053_mini_soccer_ball", "054_softball", "055_baseball", "056_tennis_ball", "057_racquetball", "058_golf_ball"],
}
BASE = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/%s_google_16k.tgz"


def fetch(oid, dst="F/assets/ycb"):
    out = os.path.join(dst, oid)
    if os.path.exists(os.path.join(out, "google_16k", "textured.obj")):
        return "済"
    os.makedirs(dst, exist_ok=True)
    tpath = os.path.join(dst, oid + ".tgz")
    try:
        urllib.request.urlretrieve(BASE % oid, tpath)
        with tarfile.open(tpath) as t:
            t.extractall(dst)
        os.remove(tpath)
        return "取得"
    except Exception as e:
        return "失敗: %s" % str(e)[:60]


if __name__ == "__main__":
    for jp, ids in SELECT.items():
        for oid in ids:
            print("%s %-24s %s" % (jp, oid, fetch(oid)), flush=True)
    io.open("F/assets/ycb/_選定.json", "w", encoding="utf-8").write(json.dumps(SELECT, ensure_ascii=False, indent=1))
    print("完了")
