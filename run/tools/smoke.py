# -*- coding: utf-8 -*-
"""走行前点検（2段目）── 数十歩だけ捨て場所に走らせて、「出るはずのもの」が出たか確かめる。

【なぜ、2026-09-11・ユーザー指示】
「構造的にミスが起きてたら走行前に止まるようにできないの？走行後発覚するのだけ避けて」。

1段目（`preflight.py`）は走らせずに分かるものを止める。ここは**走らせないと分からない**
ものを、本番（9〜15分）でなく**1分**で捕まえる。

実例：F2-129 は 4000歩・9分を完走してから「`goal_id` が全行で空」＝判定不能と分かった。
原因は列を見出しにだけ足して行に足し忘れたこと。実験ファイルに

    "expect": {"物体ファイル.csv": ["goal_id", "g_at_goal"]}

と**1行**書いてあれば、120歩＝1分で捕まった。

【肝は「宣言」】点検する側は「何が正しいか」を知らない。**実験ファイルが自分で
「この走行はこれを出すはずだ」と宣言する**から確かめられる。

【書き方】実験ファイルの最上位に `expect` を足す（省略可）。

    "expect": {
      "物体ファイル.csv": ["goal_id", "g_at_goal"],   ← 1行以上で空でないこと
      "注意.csv": ["attended_id"]
    }

【使い方】
    python run/tools/smoke.py F/experiments/<実験>.json
    python run/tools/smoke.py <実験> --steps 200

本番の出力・モデルには一切触らない（全部 scratch へ向け直す）。
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))


# 【2026-09-11】**読み込む**ほうのパス。向け直すと「モデルが無い」で落ちる
#   （最初の試作で実際に落とした）。出力だけを捨て場所へ向ける。
_INPUT_KEYS = {"model", "load", "from", "init_from", "resume"}


def _redirect(node, scratch, where=""):
    """出力っぽい文字列を全部 scratch の下へ向け直す（本番を汚さないため）。"""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out[k] = v if k in _INPUT_KEYS else _redirect(v, scratch, "%s.%s" % (where, k))
        return out
    if isinstance(node, list):
        return [_redirect(v, scratch, where) for v in node]
    if isinstance(node, str):
        s = node.replace("\\", "/")
        if ("/logs/" in s or "/models/" in s) and not s.startswith("http"):
            return os.path.join(scratch, os.path.basename(s.rstrip("/")))
        return node
    return node


def _check_expect(expect, scratch):
    """宣言どおり出ているか。(ok, 行のリスト)。"""
    lines, ok = [], True
    for fname, cols in (expect or {}).items():
        path = os.path.join(scratch, os.path.basename(fname))
        if not os.path.exists(path):
            lines.append("  [出ていない] %s ができていない" % fname)
            ok = False
            continue
        # 【2026-09-12】JSONも見る。F2-135pre で「肝心の出力がJSONなので
        #   smokeが確かめられない」と分かった（CSVしか見ていなかった）。
        #   宣言の書き方：{"サッケード1発ごと.json": ["tgt_h", "h1"]}
        #   ＝中身が空でない配列で、最初の要素にその鍵があること。
        #   鍵を省いて {"...json": []} と書けば「空でないこと」だけ見る。
        if fname.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            if not obj:
                lines.append("  [出ていない] %s が空" % fname)
                ok = False
                continue
            head = obj[0] if isinstance(obj, list) else obj
            for c in (cols if isinstance(cols, (list, tuple)) else [cols]):
                if not isinstance(head, dict) or c not in head:
                    lines.append("  [鍵が無い] %s に %s が無い" % (fname, c))
                    ok = False
                else:
                    lines.append("  [ある] %s の %s（%d 件）"
                                 % (fname, c, len(obj) if isinstance(obj, list) else 1))
            if not cols:
                lines.append("  [ある] %s：%d 件" % (fname, len(obj) if isinstance(obj, list) else 1))
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            lines.append("  [出ていない] %s が空（行が1つも無い）" % fname)
            ok = False
            continue
        for c in (cols if isinstance(cols, (list, tuple)) else [cols]):
            if c not in rows[0]:
                lines.append("  [列が無い] %s に列 %s が無い" % (fname, c))
                ok = False
                continue
            n = sum(1 for r in rows if str(r.get(c, "")).strip() != "")
            if n == 0:
                lines.append("  [全行が空] %s の %s が %d 行すべて空"
                             % (fname, c, len(rows)))
                ok = False
            else:
                lines.append("  [ある] %s の %s：%d / %d 行が非空" % (fname, c, n, len(rows)))
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--steps", type=int, default=120, help="試し走行の歩数（既定120）")
    a = ap.parse_args()

    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    expect = spec.get("expect")
    if not expect:
        print("この実験ファイルには expect が無いので、出るはずのものを確かめられない。")
        print('最上位に足す： "expect": {"物体ファイル.csv": ["列名", ...]}')
        print("（それでも試し走行はする＝落ちずに動くかだけ見る）")

    scratch = tempfile.mkdtemp(prefix="smoke_")
    tmp = copy.deepcopy(spec)
    tmp.pop("expect", None)
    tmp["name"] = str(spec.get("name", "")) + "_試し走行"
    tmp["plugins"] = _redirect(tmp.get("plugins") or {}, scratch)
    tmp["taro"] = _redirect(tmp.get("taro") or {}, scratch)
    r = tmp.setdefault("run", {})
    if r.get("csv"):
        r["csv"] = os.path.join(scratch, "run.csv")
    r["steps"] = a.steps
    tmp_spec = os.path.join(scratch, "spec.json")
    with open(tmp_spec, "w", encoding="utf-8") as f:
        json.dump(tmp, f, ensure_ascii=False, indent=2)

    print("試し走行：%d 歩（捨て場所 %s）" % (a.steps, scratch))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "run.main", tmp_spec,
                        "--steps", str(a.steps), "--skip-preflight"],
                       cwd=_ROOT, env=env, capture_output=True, text=True,
                       errors="replace")
    # 【2026-09-11】終了コードだけで落ちたと決めない。短い試し走行では
    #   「親がまだ名前を教えていない（label_count=0）」で 1 が返るのが正常。
    #   本当に落ちたなら出力そのものができないので、そちらで見分ける。
    produced = [f for f in (expect or {})
                if os.path.exists(os.path.join(scratch, os.path.basename(f)))]
    if p.returncode != 0 and not produced:
        print("\n[落ちた] 試し走行が終了コード %d で止まり、出力もできていない。"
              "本番を回す前に直す：" % p.returncode)
        print("\n".join((p.stdout or "").splitlines()[-15:]))
        print("\n".join((p.stderr or "").splitlines()[-15:]))
        return 1
    if p.returncode != 0:
        print("（終了コード %d。短い走行では親がまだ名前を教えていないため。"
              "出力はできているので中身を見る）" % p.returncode)

    # 【2026-09-12・ユーザー指示「実行中はエラーログを絶対読むようにしよう」】
    #   試し走行が作った エラー.log を、本番を回す前にここで読む。
    #   落ちなくても警告が出ていれば本番でも同じ警告が出るので、先に知る。
    err_n = _show_error_log(scratch)

    ok, lines = _check_expect(expect, scratch)
    print("\n宣言（expect）との突き合わせ")
    for ln in lines:
        print(ln)
    if not expect:
        print("  （宣言が無いので、落ちなかったことだけ確認した）")
        return 0
    print("\n" + ("試し走行：合格。本番を回してよい" if ok
                  else "試し走行：**不合格**。本番を回す前に直す（1分で分かった＝本番なら9〜15分）"))
    if ok and err_n:
        print("  ただし警告が %d 行ある（上のエラー.log）。本番でも同じものが出る" % err_n)
    return 0 if ok else 1


def _show_error_log(scratch):
    """試し走行の エラー.log を探して中身を出す。戻り値は行数（0なら異常なし）。

    【2026-09-12】走行ごとの出力フォルダは実験ファイルの書き方によって変わるので、
      scratch の下を探しに行く（run/main.py の _out_dir_of がどこに作っても拾える）。
    """
    hits = []
    for root, _dirs, files in os.walk(scratch):
        if "エラー.log" in files:
            hits.append(os.path.join(root, "エラー.log"))
    if not hits:
        print("\nエラー.log：見つからない（古い作りの走行か、ログ初期化前に落ちた）")
        return 0
    total = 0
    for pth in hits:
        try:
            lines = [l for l in open(pth, encoding="utf-8").read().splitlines() if l.strip()]
        except Exception:
            continue
        total += len(lines)
        if lines:
            print("\nエラー.log：%d 行 ← 本番を回す前に読む" % len(lines))
            for l in lines[:25]:
                print("  | " + l)
            if len(lines) > 25:
                print("  | （他 %d 行。%s）" % (len(lines) - 25, pth))
    if total == 0:
        print("\nエラー.log：空（警告もエラーも無し）")
    return total


if __name__ == "__main__":
    sys.exit(main())
