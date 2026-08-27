# -*- coding: utf-8 -*-
"""機構の稼働状況 ── 実装済みの機構が、実験で一度でも使われたかを一覧する。

【なぜ要るか、2026-08-01】色付きノイズ（cpg・Tier1・実測値あり）が、実装済みなのに
一度も実験でONになっていないまま何日も埋もれていたと分かった。太郎の機構は
run/wiring_map.py の NODES に37個あり、目で全部を追って「どれが一度も試されて
いないか」に気づくのは無理がある。⇒ 実験ファイル群を機械的に横断集計する。

【何をするか】
    E/experiments/*.json を全部読み、機構ごとに「過去に一度でもONで
    書かれたことがあるか」を出す。引数に実験ファイルを1つ渡すと、
    その実験の「今回のON/OFF」も並べて出す。

【何をしないか（段階1の範囲。設計 §11・仕様より）】
    ・「ONと書かれた」と「実際に完走した」は区別しない
      （E/logs/**/*.meta.json との突き合わせは段階2。設計 §6 の通り、
       meta.json は csv 指定かつ dashboard プラグインONのときしか作られず、
       無いことを「未使用」と誤読するリスクの方が大きいため見送った）
    ・「試されて棄却」と「単に埋もれている」は区別しない
      （区別するには run/wiring_map.py に VERDICTS 辞書が要る＝段階2。
       共通ファイルを触るので今回は作らない。仕様で明記済み）
    ・削除・改名された過去の実験ファイルは追わない（現存ファイルのみ）
    ・常時ONの機構（判定関数が無いもの。固有感覚・ドーパミンなど）は一覧に
      含めない（仕様 Q5採用。「使われたか」という問い自体が成立しないため）

【使い方】
    .venv/Scripts/python.exe run/tools/機構の稼働状況.py
        → E/experiments/ 配下の全 *.json を横断集計する

    .venv/Scripts/python.exe run/tools/機構の稼働状況.py E/experiments/<名前>.json
        → その実験の今回のON/OFFも並べて出す（横断集計と併記）

注意：機構ごとの判定は一切書き直さない。run/wiring_map.py の NODES と is_on()、
  run/config.py の Config.from_spec() をそのまま呼ぶだけ（設計 §5・仕様の指定）。
  このファイル自身は E/experiments/*.json を読むだけで、Taro の他のファイルは
  一切変更しない。
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from run.config import Config                          # noqa: E402
from run.wiring_map import NODES, is_on                 # noqa: E402

# 実験ファイルは目標フォルダごとに置かれる（E/experiments、F/experiments …）。
# 2026-08-23：目標Fの実験を E/experiments から F/experiments へ移したため、
# 1フォルダ固定をやめ、すべての目標フォルダを走査する。
EXPERIMENTS_DIRS = sorted(glob.glob(os.path.join(_ROOT, "*", "experiments")))


def targetable_nodes():
    """常時ONの機構（判定関数が無いもの）を除いた一覧を返す。

    NODES の7個目の要素（is_on の判定関数）が None のものは、実験ファイルの
    書き方によらず常にONになる（例：固有感覚・ドーパミン）。
    仕様 Q5 の採用回答：これらは「使われたか」という問い自体が成立しないので
    一覧から除く。混ぜると、本当に確かめたい機構が埋もれる。
    """
    return [n for n in NODES if n[6] is not None]


def load_experiments(dirpath):
    """dirpath 配下の *.json を全部読み、Config にする。

    Returns:
        (readable, unreadable)
        readable    … [(ファイル名, Config), ...]
        unreadable  … [(ファイル名, エラーの文面), ...]

    注意：1件が壊れていても他を止めない。`Config.__init__`（run/config.py）は
      taro/run 欄に知らないキーがあると ValueError で止める仕組みを持つ。
      ここではその仕組みを握りつぶさず、そのまま働かせたうえで、
      「このファイルだけ読めない」と記録して次のファイルに進む。
    """
    readable, unreadable = [], []
    if not os.path.isdir(dirpath):
        return readable, unreadable
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            cfg = Config.from_spec(spec)
        except Exception as e:                          # noqa: BLE001
            # 【なぜ広く受けるか】JSONの構文エラー・taro欄の知らないキー
            #   （ValueError）・組み合わせとして成り立たない設定（同じくValueError）
            #   のどれで壊れていても、このツールの役目は「一覧を止めないこと」。
            #   どんな理由で読めなかったかは unreadable にそのまま文面で残す。
            unreadable.append((fn, f"{type(e).__name__}: {e}"))
            continue
        readable.append((fn, cfg))
    return readable, unreadable


def ever_on_map(readable, nodes):
    """機構ID → E/experiments/ 全体を通して一度でもONで書かれたことがあるか。"""
    ever = {n[0]: False for n in nodes}
    for _fn, cfg in readable:
        for n in nodes:
            if not ever[n[0]]:          # 既に真と分かっているものは判定を省く
                ever[n[0]] = is_on(n, cfg)
    return ever


def _label(on):
    return "ON" if on else "OFF"


def _tier_label(tier):
    return tier if tier else "未記載"


def print_report(target_path=None):
    """一覧をテキスト表として標準出力へ書く。戻り値は終了コード。"""
    print("=" * 88)
    print(" 機構の稼働状況 ── 各目標フォルダの experiments/*.json を横断集計")
    print("=" * 88)

    if not EXPERIMENTS_DIRS:
        print(f"\n  対象なし: {_ROOT} の下に */experiments が1つもありません")
        return 0

    # 目標フォルダをまたぐのでファイル名だけだと衝突しうる。"E/名前.json" の形にする。
    readable, unreadable = [], []
    for d in EXPERIMENTS_DIRS:
        r, u = load_experiments(d)
        goal = os.path.basename(os.path.dirname(d))
        readable += [(f"{goal}/{fn}", cfg) for fn, cfg in r]
        unreadable += [(f"{goal}/{fn}", err) for fn, err in u]
    nodes = targetable_nodes()

    if not readable and not unreadable:
        dirs = "、".join(os.path.relpath(d, _ROOT) for d in EXPERIMENTS_DIRS)
        print(f"\n  対象なし: {dirs} に *.json が1件もありません")
        return 0

    print(f"\n  横断対象のファイル {len(readable)}件"
          f"（読めなかったもの {len(unreadable)}件）")
    print(f"  横断対象の機構 {len(nodes)}個"
          f"（常時ONの{len(NODES) - len(nodes)}個は除外。仕様 Q5採用）")

    ever = ever_on_map(readable, nodes)

    now = None
    if target_path is not None:
        # 注意：ここで例外を握りつぶさない。「今回」として明示的に指定された
        #   ファイルが壊れているなら、それは横断集計の1件が壊れているのとは違い、
        #   利用者が今まさに知りたい対象そのものなので、はっきり止めて伝える。
        with open(target_path, encoding="utf-8") as f:
            spec = json.load(f)
        cfg_now = Config.from_spec(spec)
        now = {n[0]: is_on(n, cfg_now) for n in nodes}
        print(f"  今回の実験ファイル: {os.path.basename(target_path)}")

    print()
    header = f"  {'機構ID':<10}{'日本語名':<24}{'Tier':<8}{'過去に一度でもON':<10}"
    if now is not None:
        header += "今回"
    print(header)
    print("  " + "-" * 84)
    for n in nodes:
        nid, jp, eng, col, tier, desc, _f = n
        row = (f"  {nid:<10}{jp:<24}{_tier_label(tier):<8}"
               f"{_label(ever[nid]):<10}")
        if now is not None:
            row += _label(now[nid])
        print(row)

    if unreadable:
        print(f"\n  読めなかったファイル（{len(unreadable)}件・集計から除外）")
        for fn, err in unreadable:
            print(f"    {fn}")
            print(f"      {err}")

    # ---- Tier1/Tier2で未使用のものを別枠で強調 ------------------------------
    #   これが設計のいちばんの狙い：実装済み・根拠ありなのに埋もれている機構を
    #   目立たせる（色付きノイズが何日も気づかれなかった事故を繰り返さない）。
    flagged = [n for n in nodes if n[4] in ("Tier1", "Tier2") and not ever[n[0]]]
    print()
    print("-" * 88)
    print(" 要確認：根拠がある(Tier1/Tier2)のに一度もONで書かれたことがない機構")
    print("-" * 88)
    if not flagged:
        print("  なし")
    else:
        for n in flagged:
            nid, jp, eng, col, tier, desc, _f = n
            print(f"  {nid:<10}{jp:<24}{tier:<8}{desc}")
        print()
        print("  注意：段階1の自動判定では「試されて棄却」と「単に埋もれている」を")
        print("  区別できない（設計 §8・§9のQ3）。ここに出た機構は、既に棄却済み")
        print("  かもしれない。doc/人間模倣からの逸脱リスト.md も合わせて見ること。")
    return 0


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target is not None and not os.path.isfile(target):
        print(f"エラー: 指定された実験ファイルが見つかりません: {target}",
              file=sys.stderr)
        return 1
    return print_report(target)


if __name__ == "__main__":
    sys.exit(main())
