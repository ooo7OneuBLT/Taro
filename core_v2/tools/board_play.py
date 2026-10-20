"""6段目「親役で話す」の道具：人がママの役で書いた行動に、育った太郎が返す（core_v2/board/play.py）。

  python -m core_v2.tools.board_play --session 遊び.pt --new 育った太郎.pt      # 始める（5段目まで育った脳から）
  python -m core_v2.tools.board_play --session 遊び.pt "ボールを見せる"
  python -m core_v2.tools.board_play --session 遊び.pt "「ボールだよ」"
  python -m core_v2.tools.board_play --session 遊び.pt "待つ 30分"
  python -m core_v2.tools.board_play --interactive                                 # 続けて書いて遊ぶ（TaroPlay.bat）
1回ごとに、ママのしたこと・太郎のしたこと・その時の太郎の頭の中・今の様子を書き、遊びの続きを --session に保存する。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import torch

from core_v2.board.play import HEAD_JA, Play, load, save

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")
GROWN = os.path.join(DATA, "親役_太郎.pt")     # 当て推量の世界で5段目まで育てた太郎（種0、2026-10-20）
SAVES = os.path.join(DATA, os.pardir, os.pardir, "play_saves")
HELP = """ママの役で、することを書いて Enter を押してください。太郎が何かするまで世界が進みます。
  ミルクをあげる ／ 飲ませる ／ やめる ／ ボールを見せる ／ 片付ける ／ 笑う ／ 出ていく ／ 戻る ／ 待つ 10分
  話しかける時はかぎかっこで：「ぼーるだよ」（話す言葉はひらがな・カタカナで。漢字は太郎の耳に届きません）
  見せられる物：ボール コップ くつ わんわん くるま バナナ りんご ぼうし いす ほん にゃんにゃん でんしゃ など
  はじめから ＝ 太郎を育った時に戻す ／ おわり ＝ やめる（続きは保存されます）"""

HEAD_SHORT = {"food": "おなか", "know": "知りたい", "stop": "いや", "social": "相手"}


def describe(a: dict) -> str:
    if a["action"] == "cry":
        return f"{a['to']}に向かって泣いた（強さ {a['strength']:.1f}）"
    if a["action"] == "reach":
        return f"{a['to']}に手を伸ばした（強さ {a['strength']:.1f}）"
    if a["action"] == "say":
        about = f"。{a['about']}のことを、太郎はこの語で呼んでいる" if a.get("about") else ""
        return f"{a['to']}に「{a['say']}」と言った（強さ {a['strength']:.1f}{about}）"
    return "何もしなかった"


def report(r: dict) -> str:
    lines = []
    if r["notes"]:
        lines.append("ママ：" + " / ".join(r["notes"]))
    acted = [a for a in r["steps"] if a["action"] != "nothing"]
    if not acted and r["steps"]:
        lines.append(f"太郎：{len(r['steps'])}分のあいだ何もしなかった")
    for a in acted:
        lines.append(f"太郎（{a['t']}分め）：{describe(a)}")
        why = a["why"]
        d = why["drives"]
        lines.append("  欲求の強さ：" + "、".join(f"{HEAD_SHORT[h]} {d.get(h, 0):.2f}" for h in HEAD_SHORT))
        if why["head"]:
            lines.append(f"  選んだ理由：{HEAD_JA[why['head']]}のため")
            opts = sorted((o for o in why["options"] if o["head"] == why["head"]), key=lambda o: -o["value"])
            total = sum(o["value"] for o in opts) or 1.0
            lines.append("  思い浮かんだ選択肢（見込み → 選ぶ割合）：" + "、".join(
                f"{o['what']}→{o['to']} {o['value']:.3f}（{100 * o['value'] / total:.0f}%）" for o in opts[:5]))
    s = r["state"]
    obj = "なし" if not s["object"] else f"{s['object']}（名前を{'知っている' if s['knows_object'] else 'まだ知らない'}）"
    lines.append(f"様子：おなか {s['fullness']:.2f}（0.6より下だとミルクを飲める）、目の前 {obj}、"
                 f"部屋 {'・'.join(s['present']) or 'だれもいない'}{'、ミルクをすすめている最中' if s['offered'] else ''}")
    return "\n".join(lines)


def interactive(session: str, fullness: float, wait: int) -> None:
    """続けて書いて遊ぶ。1回ごとに保存するので、閉じても続きから始まる。"""
    os.makedirs(os.path.dirname(os.path.abspath(session)), exist_ok=True)
    fresh = lambda: setattr(p := Play.new(torch.load(GROWN, weights_only=False)), "fullness", fullness) or p
    play = load(session) if os.path.exists(session) else fresh()
    print(HELP)
    print(report({"notes": [], "steps": [], "state": play.state()}))
    while True:
        try:
            text = input("\nママ> ").strip()
        except EOFError:
            break
        if text in ("おわり", "終わり", "exit", "quit"):
            break
        if text in ("はじめから", "最初から"):
            play = fresh()
            save(play, session)
            with open(os.path.splitext(session)[0] + "_記録.txt", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M}] はじめから（太郎を育った時に戻した）\n")
            print("太郎を育った時に戻しました。")
            print(report({"notes": [], "steps": [], "state": play.state()}))
            continue
        if text in ("", "?", "？", "help", "ヘルプ"):
            print(HELP)
            continue
        out = report(play.turn(text, wait=wait))
        print(out)
        save(play, session)
        # やり取りを文字でも残す（あとで読み返す・Claude に渡すため）
        with open(os.path.splitext(session)[0] + "_記録.txt", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M}] ママ> {text}\n{out}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.path.join(SAVES, "遊び.pt"))
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--new", help="育った太郎（torch.save した Brain）から遊びを始める")
    ap.add_argument("--fullness", type=float, default=0.5)
    ap.add_argument("--wait", type=int, default=5, help="太郎が何かするまで待つ最長の分")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("text", nargs="?", default="")
    args = ap.parse_args()
    if args.interactive:
        return interactive(args.session, args.fullness, args.wait)
    if args.new:
        play = Play.new(torch.load(args.new, weights_only=False))
        play.fullness = args.fullness
    else:
        play = load(args.session)
    r = play.turn(args.text, wait=args.wait)
    save(play, args.session)
    print(json.dumps(r, ensure_ascii=False, default=str) if args.json else report(r))


if __name__ == "__main__":
    main()
