"""★HTMLの中の操作スクリプト（JavaScript）が本当に動くかを確かめる。

【なぜ要るか、2026-07-30】配線図に「スクロールで拡大・ドラッグで移動」を実装したのに
ユーザーの手元で**動かなかった**。原因は JS の文字列に**実際の改行が入っていた**
構文エラーで、HTMLを見ただけでは分からない（ブラウザは黙って何もしない）。

⇒ 偽のブラウザ（document / window の最小限の代わり）を与えて node で走らせ、
  **最後まで走るか**と**どの操作が登録されたか**を確かめる。

⚠️node が無い環境では「確かめられなかった」と出す。★嘘の合格を出さない。
⚠️これで分かるのは「エラーで止まらない」ことだけ。**見た目や操作感は人が見る**。

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_js
    .venv/Scripts/python.exe -m run.tools.check_js E/logs/selfmodel_v2/配線図_条件C.html
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))

# 偽のブラウザ。★本物の DOM は使わない（画面が無くても走るようにする）
FAKE = """
const listened = [];
function el(id){
  return {id, dataset:{x:'0', y:'0'}, style:{}, textContent:'',
    setAttribute(){}, getAttribute(){ return ''; },
    addEventListener(t){ listened.push(t); },
    closest(){ return null; },
    querySelector(){ return el('cam'); },
    getBoundingClientRect(){ return {left:0, top:0, width:1052}; },
    setPointerCapture(){}, onclick:null};
}
const DATA = {nodes:{a:[0,0], b:[200,0]}, edges:[['a','b']],
              boxW:176, boxH:46, w:1052, h:926};
global.document = {
  querySelector: () => el('svg'),
  getElementById: (id) => id === 'wireData'
      ? {textContent: JSON.stringify(DATA)} : el(id)
};
global.window = {open: () => ({document:{write(){}}})};
global.navigator = {clipboard:{writeText: () => Promise.resolve()}};
global.__done = () => console.log('LISTENED:' + listened.join(','));
"""


def check(html_path):
    """(結果の文字列, 登録された操作のリスト) を返す。"""
    if shutil.which("node") is None:
        return "確かめられなかった（node が無い）", []
    if not os.path.exists(html_path):
        return f"確かめられなかった（{os.path.basename(html_path)} が無い）", []
    h = open(html_path, encoding="utf-8").read()
    blocks = re.findall(r"<script(?![^>]*type=)[^>]*>(.*?)</script>", h, re.S)
    if not blocks:
        return "⚠️スクリプトが見つからない", []
    for js in blocks:
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "check.js")
            open(f, "w", encoding="utf-8").write(FAKE + js + "\n__done();\n")
            r = subprocess.run(["node", f], capture_output=True, text=True,
                               errors="replace")
        if r.returncode != 0:
            lines = (r.stderr or "").strip().splitlines()
            msg = next((l for l in lines if "Error" in l), lines[0] if lines else "")
            return f"⚠️★エラーで止まる: {msg[:150]}", []
        got = [l[9:] for l in (r.stdout or "").splitlines()
               if l.startswith("LISTENED:")]
        ops = got[0].split(",") if got and got[0] else []
        return "★最後まで走った", [o for o in ops if o]
    return "⚠️確かめられなかった", []


def main():
    paths = sys.argv[1:] or [os.path.join(_ROOT, "E", "docs", "配線図.html")]
    print("=" * 78)
    print(" HTMLの操作スクリプトの検査")
    print("=" * 78)
    bad = 0
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        res, ops = check(p)
        print(f"\n  {os.path.relpath(p, _ROOT)}")
        print(f"    {res}")
        if ops:
            print(f"    登録された操作: {', '.join(ops)}")
            need = {"wheel", "pointerdown", "pointermove", "pointerup"}
            miss = need - set(ops)
            if miss:
                print(f"    ⚠️★足りない操作: {sorted(miss)}")
                bad += 1
        if res.startswith("⚠️"):
            bad += 1
    print("\n" + "-" * 78)
    print("  ★問題なし" if bad == 0 else f"  ⚠️{bad} 件の問題")
    return 0


if __name__ == "__main__":
    sys.exit(main())
