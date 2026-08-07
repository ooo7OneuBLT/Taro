"""Viewer のランチャー — シーンを選んで起動するだけの入口。

【なぜ要るか、2026-07-29】ユーザーの要望「ダブルクリックで Viewer を開けるように」。
起動コマンドが長く、環境変数を1つ間違えると別の条件で立ち上がる問題があった
（シーン方式でシーン名1つになったが、それでもコマンドを打つ必要がある）。

使い方:
    プロジェクト直下の `Viewerを開く.bat` をダブルクリックする
    （コマンドから直接なら .venv/Scripts/python.exe run/viewer_tools/e_launch.py）

    番号を入れて Enter … そのシーンで開く
    そのまま Enter     … 前回と同じシーンで開く
"""
import os
import re
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SCENE_TOOLS = os.path.join(_ROOT, "run", "scene_tools")
if _SCENE_TOOLS not in sys.path:
    sys.path.insert(0, _SCENE_TOOLS)

import e_scene  # noqa: E402

LAST = os.path.join(e_scene.SCENE_DIR, "_前回開いたシーン.txt")


def read_last():
    try:
        with open(LAST, encoding="utf-8") as fp:
            return fp.read().strip()
    except Exception:
        return None


def write_last(name):
    try:
        os.makedirs(e_scene.SCENE_DIR, exist_ok=True)
        with open(LAST, "w", encoding="utf-8") as fp:
            fp.write(name)
    except Exception:
        pass


def describe(sc):
    """そのシーンが成立しているかを1行で。選ぶ前に分かるようにする。"""
    fp = sc.get("fingerprint") or {}
    se = fp.get("settle") or {}
    bits = [f"{sc['body']['age_months']:g}ヶ月",
            f"リクライニング{sc['world']['recline_deg']:g}度",
            "柵あり" if sc["world"]["fence"] else "柵なし",
            "頭を支える" if sc["setup"].get("head_hold") else "支えなし"]
    warn = []
    if se and not se.get("ok", True):
        warn.append("姿勢が崩れる")
    if fp.get("toy_visible_left") is False:
        warn.append("おもちゃが見えない")
    if fp.get("toy_reach_ratio") is not None and float(fp["toy_reach_ratio"]) > 1.0:
        warn.append(f"おもちゃに手が届かない（腕の{float(fp['toy_reach_ratio'])*100:.0f}%）")
    line = " / ".join(bits)
    if warn:
        line += "   注意" + "、".join(warn)
    return line


# 【なぜ、2026-08-07】以前は①（describe()の短い1行）と③（scene JSONのnote、
#   数百字規模の長い説明文）を選択肢ごとに常に両方表示していた。シーンが11個ある
#   と③だけで画面が埋まり、選びにくいというユーザー指摘があった。
#   ⇒ 既定では①だけを表示し、③は「?9」「d9」のように番号を指定したときだけ
#   その場で表示する形に変えた（表示するかどうかのon/offだけを変え、①の
#   テキスト内容そのもの＝describe()の返す文字列は一切変えていない）。
_DETAIL_RE = re.compile(r"^[?dD]\s*(\d+)$")


def print_scene_list(names, last):
    """①（describe()の1行、警告込み）だけを一覧表示する。③（note）はここでは出さない。"""
    print("=" * 74)
    print(" どのシーンで太郎を見ますか")
    print("=" * 74)
    for i, n in enumerate(names, 1):
        mark = "←前回" if n == last else ""
        try:
            info = describe(e_scene.load(n))
        except Exception as e:
            info = f"（読めない: {e}）"
        print(f"  {i}. {n}  {mark}")
        print(f"       {info}")


def print_footer(last):
    print("-" * 74)
    print(f"  番号を入れて Enter ／ そのまま Enter で「{last}」／ 詳しい説明は ?番号 か d番号")


def show_detail(names, idx):
    """番号idx（1始まり）のシーンのnote（③）を表示する。範囲外ならエラー文だけ出す。"""
    print("-" * 74)
    if not (1 <= idx <= len(names)):
        print(f"  {idx} は範囲外です（1〜{len(names)}で指定してください）")
        return
    n = names[idx - 1]
    try:
        note = e_scene.load(n).get("note")
    except Exception as e:
        print(f"  {idx}. {n}  （読めない: {e}）")
        return
    print(f"  {idx}. {n}")
    if note:
        print(f"       {note}")
    else:
        print("       （詳しい説明はありません）")


def select_scene(names, last):
    """一覧を表示し、選択を受け付ける。戻り値は選んだシーン名。
    EOF/中断なら None を返す（呼び出し側はそのまま終了する）。
    ・数字だけ入力     → その番号を選ぶ
    ・空Enter          → 前回選択（last）
    ・名前の一部だけ入力 → 部分一致で1件に絞れればそれを選ぶ
    ・?9 / d9 のような入力 → その番号のnote（③）を表示し、footerだけ出し直して再度入力を待つ
    """
    print_scene_list(names, last)
    print_footer(last)
    while True:
        try:
            s = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not s:
            return last

        m = _DETAIL_RE.match(s)
        if m:
            show_detail(names, int(m.group(1)))
            print_footer(last)
            continue

        try:
            i = int(s)
            if 1 <= i <= len(names):
                return names[i - 1]
            print(f"  {i} は範囲外なので「{last}」で開きます")
            return last
        except ValueError:
            # 名前の一部でも選べるようにする
            cand = [n for n in names if s in n]
            if len(cand) == 1:
                return cand[0]
            print(f"  「{s}」では決まらないので「{last}」で開きます")
            return last


def main():
    names = e_scene.list_scenes()
    if not names:
        print("シーンが1つもありません。")
        print("  .venv/Scripts/python.exe run/scene_tools/e_scene_make.py  で作れます")
        input("\nEnter で閉じます ")
        return 1

    last = read_last()
    if last not in names:
        last = names[0]

    pick = select_scene(names, last)
    if pick is None:
        return 0
    write_last(pick)

    print(f"\n  「{pick}」で開きます。体を作るのに数十秒かかります …\n")
    envv = dict(os.environ)
    envv["E_SCENE"] = pick
    envv["PYTHONIOENCODING"] = "utf-8"
    # 注意：シーンが全部を決めるので、古い個別指定は消す
    #   （残っていると「シーンの値と環境変数のどちらが効くのか」が曖昧になる）
    for k in ("E_AGE", "E_HEAD_HOLD", "E_FENCE", "E_TOY_RADIUS", "E_RECLINE",
              "E_EYE_REST_V", "E_HOLD_TILT", "E_TOY_POS", "E_SEAT_FRICTION",
              "E_TOY_MODE", "E_TOY_SHAPE", "E_TOY_DIST", "E_ORIENT_V"):
        envv.pop(k, None)
    return subprocess.call([sys.executable,
                            os.path.join(_HERE, "e_viewer.py")],
                           env=envv, cwd=_ROOT)


if __name__ == "__main__":
    sys.exit(main())
