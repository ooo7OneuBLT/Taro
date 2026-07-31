"""Viewer のランチャー — シーンを選んで起動するだけの入口。

【なぜ要るか、2026-07-29】ユーザーの要望「ダブルクリックで Viewer を開けるように」。
起動コマンドが長く、環境変数を1つ間違えると別の条件で立ち上がる問題があった
（シーン方式でシーン名1つになったが、それでもコマンドを打つ必要がある）。

使い方:
    プロジェクト直下の `Viewerを開く.bat` をダブルクリックする
    （コマンドから直接なら .venv/Scripts/python.exe E/scripts/e_launch.py）

    番号を入れて Enter … そのシーンで開く
    そのまま Enter     … 前回と同じシーンで開く
"""
import os
import sys
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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


def main():
    names = e_scene.list_scenes()
    if not names:
        print("シーンが1つもありません。")
        print("  .venv/Scripts/python.exe E/scripts/e_scene_make.py  で作れます")
        input("\nEnter で閉じます ")
        return 1

    last = read_last()
    if last not in names:
        last = names[0]

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
        try:
            note = e_scene.load(n).get("note")
            if note:
                print(f"       {note}")
        except Exception:
            pass
    print("-" * 74)
    print(f"  番号を入れて Enter ／ そのまま Enter で「{last}」")

    try:
        s = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0
    pick = last
    if s:
        try:
            i = int(s)
            if 1 <= i <= len(names):
                pick = names[i - 1]
            else:
                print(f"  {i} は範囲外なので「{last}」で開きます")
        except ValueError:
            # 名前の一部でも選べるようにする
            cand = [n for n in names if s in n]
            if len(cand) == 1:
                pick = cand[0]
            else:
                print(f"  「{s}」では決まらないので「{last}」で開きます")
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
