# -*- coding: utf-8 -*-
"""起動前チェック：一連の実験を回す前に、止まる原因を先に潰す。

【なぜ要るか・2026-08-27】2026-08-26に、実験の起動そのものが3回失敗した。
  ① ログの保存先フォルダが無くて起動直後に停止（ABAB1周目）
  ② 別のworld.xmlを使う古いシーンから複製したため姿勢データが不一致でエラー
  ③ 4本同時起動でメモリ不足（Could not allocate memory）
  どれも走らせる前に分かることだった。文書の注意書きではなく機械で防ぐ
  （CLAUDE.md「同じミスが2回起きたら機械で防ぐ」）。

【使い方】
  python run/tools/preflight.py F/experiments/F2-10_*.json
  → 実行順（モデルのチェーン）と、見つかった問題を並べて出す。
     問題があれば終了コード1。フォルダは作るが、それ以外は何も変えない。
"""
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _p(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def check(paths):
    errs, warns, made = [], [], []
    exps = []
    for p in paths:
        try:
            exps.append((p, json.load(open(p, encoding="utf-8"))))
        except Exception as e:
            errs.append("%s: 読めない (%s)" % (os.path.basename(p), e))
    # この一連で作られる予定のモデル
    will_make = set()
    for p, d in exps:
        s = d.get("taro", {}).get("save")
        if s:
            will_make.add(os.path.normpath(_p(s)))

    for p, d in exps:
        n = d.get("name") or os.path.basename(p)
        t, r = d.get("taro", {}), d.get("run", {})
        # ① シーン
        sc = d.get("scene")
        if sc:
            sp = _p(os.path.join("run", "scenes", sc + ".json"))
            if not os.path.exists(sp):
                errs.append("%s: シーンが無い → run/scenes/%s.json" % (n, sc))
            else:
                w = json.load(open(sp, encoding="utf-8")).get("world_xml")
                if w and not os.path.exists(_p(w)):
                    errs.append("%s: シーンが指すworld.xmlが無い → %s" % (n, w))
        # ② 起点モデル（この一連で作られるものはOK）
        m = t.get("model")
        if m:
            mp = os.path.normpath(_p(m))
            if not os.path.exists(mp) and mp not in will_make:
                errs.append("%s: 起点モデルが無い → %s" % (n, m))
        # ③ 保存先の衝突（既存モデルを上書きしない）
        s = t.get("save")
        if not s:
            errs.append("%s: taro.save が無い（学習結果が消える）" % n)
        elif os.path.exists(_p(s)) and os.path.normpath(_p(s)) not in will_make:
            warns.append("%s: 保存先が既にある（上書きになる） → %s" % (n, s))
        # ④ 出力フォルダを作る
        outs = [r.get("csv")] + [v.get("events_out") for v in
                                 (d.get("plugins") or {}).values() if isinstance(v, dict)]
        for o in outs + ([s] if s else []):
            if not o:
                continue
            dirn = os.path.dirname(_p(o))
            if dirn and not os.path.isdir(dirn):
                os.makedirs(dirn, exist_ok=True)
                made.append(os.path.relpath(dirn, ROOT))
    return exps, errs, warns, made


def chain_order(exps):
    """save→model の連鎖から実行順を決める。連鎖でないものは末尾へ。"""
    by_save = {os.path.normpath(_p(d["taro"]["save"])): (p, d)
               for p, d in exps if d.get("taro", {}).get("save")}
    remaining = list(exps)
    order, made = [], set()
    while remaining:
        ready = [(p, d) for p, d in remaining
                 if not d.get("taro", {}).get("model")
                 or os.path.normpath(_p(d["taro"]["model"])) not in by_save
                 or os.path.normpath(_p(d["taro"]["model"])) in made]
        if not ready:
            order.extend(remaining)
            break
        ready.sort(key=lambda x: x[0])
        p, d = ready[0]
        order.append((p, d)); remaining.remove((p, d))
        made.add(os.path.normpath(_p(d["taro"]["save"])))
    return order


def main(argv):
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)) or [a])
    if not paths:
        print("使い方: python run/tools/preflight.py <実験ファイル...>")
        return 2
    exps, errs, warns, made = check(paths)
    order = chain_order(exps)
    print("=== 実行順（モデルの受け渡しから決定）===")
    for i, (p, d) in enumerate(order, 1):
        print("  %2d. %s" % (i, os.path.basename(p)))
    if made:
        print("\n作ったフォルダ: %d個" % len(set(made)))
    for w in warns:
        print("\n[注意] " + w)
    if errs:
        print("\n=== 止まる原因 %d件 ===" % len(errs))
        for e in errs:
            print("  [NG] " + e)
        return 1
    print("\n✓ 起動前チェック OK（%d本）" % len(exps))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
