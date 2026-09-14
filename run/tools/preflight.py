# -*- coding: utf-8 -*-
"""走行前点検（1段目）── 走らせなくても分かる構造のミスで、走る前に止める。

【なぜ、2026-09-11・ユーザー指示】
「構造的にミスが起きてたら走行前に止まるようにできないの？走行後発覚するのだけ避けて。
これは今回だけじゃなくてすべてに共通すること」。

実際に走行を捨てた事故（走行後に発覚したもの）：

| 起きたこと | 捨てた時間 |
|---|---|
| 出力先が別の実験を指していて、その実験のファイルを上書き（F2-129） | 9分＋基準データ1件 |
| 列を見出しにだけ足して行に足し忘れ（F2-129） | 9分 |
| `taro.save` を書き忘れてモデルが消えた（3回） | その都度 |
| 場面が違うものを比べていた | 複数本 |

このうち**走らせなくても分かるもの**をここで止める。走らせないと分からないものは
2段目（試し走行＋出るはずのものの宣言）が受け持つ。

【方針】
- 止める（ERROR）＝ほぼ確実に事故になるもの。`--skip-preflight` で無視できる
- 知らせる（WARN）＝意図的なこともあるもの。止めない
- **偽陽性で作業を止めない**こと。迷ったら WARN 側にする

【使い方】`run/main.py` が走行の前に自動で呼ぶ。単体でも動く：
    python run/tools/preflight.py F/experiments/<実験>.json
"""
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      os.pardir, os.pardir))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def _collect_out_paths(spec):
    """実験ファイルの中の「書き出す先」を全部集める。(どこで指定されたか, パス)。"""
    found = []
    r = spec.get("run") or {}
    if r.get("csv"):
        found.append(("run.csv", str(r["csv"])))
    t = spec.get("taro") or {}
    if t.get("save"):
        found.append(("taro.save", str(t["save"])))

    def walk(node, where):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, "%s.%s" % (where, k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, "%s[%d]" % (where, i))
        elif isinstance(node, str):
            s = node.replace("\\", "/")
            if ("/logs/" in s or "/models/" in s) and not s.startswith("http"):
                found.append((where, node))
    walk(spec.get("plugins") or {}, "plugins")
    return found


def _log_dir_of(path):
    """`F/logs/<実験名>/xxx.csv` → `F/logs/<実験名>`。logs 配下でなければ None。"""
    s = path.replace("\\", "/")
    i = s.find("/logs/")
    if i < 0:
        return None
    rest = s[i + len("/logs/"):]
    if "/" not in rest:
        return None            # F/logs/直下のファイル（_F2-129.log など）は対象外
    return s[:i + len("/logs/")] + rest.split("/", 1)[0]


def _場面の兄弟(name):
    """その場面の名前で始まる別の場面（＝あとから作られた派生）を返す。

    【なぜ名前で探すか、2026-09-14】場面は「親の名前＋しっぽ」の形で複製されてきた
    （実測26対）。ファイルの中には親子の関係が書かれていないので、名前が唯一の手がかり。
    **新しい＝良いとは限らない**（`_保持を切る` `_バネを切る` はわざと機能を切った版）ので、
    止めずに並べて見せるだけにする。選ぶのは人。
    """
    out = []
    for d in ("run/scenes", "run/scenes/_旧"):
        dd = _abs(d)
        if not os.path.isdir(dd):
            continue
        for f in os.listdir(dd):
            if not f.endswith(".json"):
                continue
            stem = f[:-5]
            if stem != name and stem.startswith(name + "_"):
                out.append(stem + ("" if d == "run/scenes" else "（_旧）"))
    return sorted(out)


def check(spec, spec_path=""):
    """(errors, warns) を返す。どちらも文字列のリスト。"""
    errors, warns = [], []
    name = str(spec.get("name") or "")
    outs = _collect_out_paths(spec)

    # --- 1. 出力先が2つ以上の実験フォルダに散らばっていないか -----------------
    #   F2-129：plugins のパスは書き換えたのに run.csv だけ F2-126 のままで、
    #   F2-126 の run.csv・ダッシュボードを上書きした。
    dirs = {}
    for where, p in outs:
        d = _log_dir_of(p)
        if d:
            dirs.setdefault(d, []).append(where)
    if len(dirs) > 1:
        detail = " / ".join("%s ← %s" % (d, "・".join(w)) for d, w in sorted(dirs.items()))
        errors.append(
            "書き出す先が2つ以上の実験フォルダに分かれている（別の実験を上書きする）：\n"
            "      %s" % detail)

    # --- 2. 他の実験のフォルダへ書こうとしていないか ---------------------------
    #   そのフォルダの run.meta.json に別の実験名が入っていれば、それは他人の場所。
    for d in dirs:
        meta = _abs(os.path.join(d, "run.meta.json"))
        if not os.path.exists(meta):
            continue
        try:
            with open(meta, encoding="utf-8") as f:
                other = str(json.load(f).get("name") or "")
        except Exception:
            continue
        if other and name and other != name:
            errors.append(
                "%s は別の実験『%s』のフォルダ（run.meta.json の名前が違う）。"
                "この走行『%s』の出力で上書きされる" % (d, other, name))

    # --- 3. 学習ありなのにモデルの保存先が無い -------------------------------
    r = spec.get("run") or {}
    if str(r.get("type", "measure")).lower() == "train" and not (spec.get("taro") or {}).get("save"):
        errors.append("学習ありの走行なのに taro.save が無い（走り終わってもモデルが残らない）")

    # --- 4. 読み込むものが実在するか ------------------------------------------
    model = (spec.get("taro") or {}).get("model")
    if model and not os.path.exists(_abs(str(model))):
        errors.append("出発モデルが無い: %s" % model)
    scene = spec.get("scene")
    if scene:
        cur = _abs(os.path.join("run/scenes", str(scene) + ".json"))
        old_p = _abs(os.path.join("run/scenes/_旧", str(scene) + ".json"))
        if not os.path.exists(cur) and not os.path.exists(old_p):
            warns.append("run/scenes に %s.json が見当たらない（別の探し方をしているなら無視してよい）"
                         % scene)
        else:
            # 【2026-09-14・ステップ2a】`_旧/` ＝ 使い終わった場面の置き場。
            #   読めるので止めないが、選び直したのかは分からないので知らせる。
            if not os.path.exists(cur):
                m = "場面 %s は run/scenes/_旧/ にある（使い終わった置き場）" % scene
                if not str(spec.get("note") or "").strip():
                    m += "。note が空＝わざと選んだ理由が残らない"
                warns.append(m)
            # 【なぜ現役の場面でも知らせるか】F2-129（本番）は、改善後の
            #   `_背景あり_2026-09-10` があるのに元の版で走った。元の版は
            #   直近で使われていて `_旧/` には無い＝上の判定では鳴らない。
            #   **止めない**：`_保持を切る` `_バネを切る` のように、わざと機能を
            #   切った派生もあるので、新しい＝良いとは限らない。並べて見せるだけ。
            sibs = _場面の兄弟(str(scene))
            if sibs:
                warns.append("場面 %s には派生がある（選び直したか確かめる）：\n      %s"
                             % (scene, "\n      ".join(sibs)))

    # --- 5. 既にあるものを上書きするか（同じ実験名なら意図的なことが多い）------
    #   1件1行にすると再走行のたびに10行出てうるさいので、まとめて1行にする。
    exist = [p for _w, p in outs if os.path.exists(_abs(p))]
    if exist:
        warns.append("既にあるものを %d 件上書きする（同じ実験の撮り直しなら想定どおり）: %s%s"
                     % (len(exist), exist[0],
                        " ほか%d件" % (len(exist) - 1) if len(exist) > 1 else ""))

    return errors, warns


def _索引の遅れを知らせる():
    """落とし穴チェックリストの索引が本体に追いついていなければ1行だけ知らせる。

    【2026-09-12・ユーザー指示「索引も随時更新するようにして」】
      索引は1か月（32件）放置されていた。「次からは更新する」では守られないので
      機械にする。ただし**実験は止めない**（文書の遅れで走行を止めるのは本末転倒）。
      中身は run/tools/check_index.py。
    """
    try:
        from run.tools import check_index as ci
        漏れ = [h for h in ci._見出し(ci.本体)
                if not any(k and k in h[1] for k in ci._索引の検索文字列(ci.索引))]
        if 漏れ:
            print("  [注意] 落とし穴チェックリストの索引に %d 件の漏れがあります"
                  "（索引を読んでも当たらない）。"
                  "`python -m run.tools.check_index` で一覧" % len(漏れ))
    except Exception:
        pass      # 点検の付け足しで走行を止めない


def run_check(spec, spec_path="", skip=False):
    """点検して表示する。ERROR があれば False を返す（走らせない）。"""
    if skip:
        print("走行前点検：--skip-preflight のため飛ばした")
        return True
    errors, warns = check(spec, spec_path)
    for w in warns:
        print("  [注意] %s" % w)
    _索引の遅れを知らせる()
    if not errors:
        print("走行前点検：問題なし（%d 件の注意）" % len(warns))
        return True
    print("\n" + "=" * 70)
    print(" 走行前点検で止めました（走らせる前に直せるものです）")
    print("=" * 70)
    for e in errors:
        print("  [止めた] %s" % e)
    print("\n  どうしても走らせたいなら --skip-preflight を付ける")
    print("=" * 70)
    return False


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    return 0 if run_check(spec, sys.argv[1]) else 1


if __name__ == "__main__":
    sys.exit(main())
