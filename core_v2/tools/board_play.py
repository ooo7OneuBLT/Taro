"""6段目「親役で話す」の道具：人がママの役で書いた行動に、育った太郎が返す（core_v2/board/play.py）。

  python -m core_v2.tools.board_play --session 遊び.pt --new 育った太郎.pt      # 始める（5段目まで育った脳から）
  python -m core_v2.tools.board_play --session 遊び.pt "ボールを見せる"
  python -m core_v2.tools.board_play --session 遊び.pt "「ボールだよ」"
  python -m core_v2.tools.board_play --session 遊び.pt "待つ 30分"
1回ごとに、ママのしたこと・太郎のしたこと・その時の太郎の頭の中・今の様子を書き、遊びの続きを --session に保存する。
"""

from __future__ import annotations

import argparse
import json

import torch

from core_v2.board.play import HEAD_JA, Play, load, save

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
    if not acted:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--new", help="育った太郎（torch.save した Brain）から遊びを始める")
    ap.add_argument("--fullness", type=float, default=0.5)
    ap.add_argument("--wait", type=int, default=5, help="太郎が何かするまで待つ最長の分")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("text", nargs="?", default="")
    args = ap.parse_args()
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
