"""図の「機械的に確かめられること」を検査する。見た目の主観は人が見る。

【なぜ要るか、2026-07-30】ユーザーの問い：

> 見た目(かぶってないかとかみやすいか)はいちいち僕が見たほうがいい？

私（AI）はブラウザの画面を撮れない環境なので、**絵を目で見られない**。
そこで役割を分ける：

    機械で確かめられる  箱の重なり／はみ出し／字が箱に収まるか／線の交差の数
                        線が箱を横切っていないか
    人が見るしかない  見やすいか・美しいか・AIっぽくないか

注意：この検査を通っても「見やすい」保証にはならない。**問題が無いことの確認**だけ。

【使い方】
    .venv/Scripts/python.exe -m run.tools.check_figure
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _text_w(text, size):
    """字の幅の見積もり。日本語は1文字≒size、英数字は≒size*0.55。"""
    return sum(size if ord(c) > 0x2000 else size * 0.55 for c in text)


def _seg_rect(p1, p2, rect, pad=4):
    """線分が箱を横切るか（粗い判定：箱の4辺との交差を見る）。"""
    (x1, y1), (x2, y2) = p1, p2
    rx, ry, rw, rh = rect
    rx -= pad; ry -= pad; rw += pad * 2; rh += pad * 2

    def cross(a, b, c, d):
        def s(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        d1, d2 = s(a, b, c), s(a, b, d)
        d3, d4 = s(c, d, a), s(c, d, b)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    corners = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
    for i in range(4):
        if cross((x1, y1), (x2, y2), corners[i], corners[(i + 1) % 4]):
            return True
    return False


def main():
    from run.wiring_map import NODES, EDGES, node_index
    from run.tools.wiring import layout, BOX_W, BOX_H
    pos, titles, W, H = layout()
    idx = node_index()
    ng = 0

    print("=" * 78)
    print(" 配線図の検査（機械で確かめられることだけ）")
    print("=" * 78)
    print(f"  絵の大きさ {W} x {H} ／ 機構 {len(NODES)} ／ 線 {len(EDGES)}")

    # ---- ① 箱の重なり -------------------------------------------------------
    ks = list(pos)
    over = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]
            if abs(pos[a][0] - pos[b][0]) < BOX_W and abs(pos[a][1] - pos[b][1]) < BOX_H]
    print(f"\n【①】箱の重なり : {'なし' if not over else f'{len(over)}組'}")
    for a, b in over[:5]:
        print(f"      {a} と {b}")
    ng += len(over)

    # ---- ② はみ出し ---------------------------------------------------------
    out = [k for k, (x, y) in pos.items()
           if x < 0 or y < 0 or x + BOX_W > W or y + BOX_H > H]
    print(f"【②】画面からのはみ出し : {'なし' if not out else f'{out}'}")
    ng += len(out)

    # ---- ③ 字が箱に収まるか --------------------------------------------------
    tight = []
    for nid, jp, eng, col, tier, desc, fn in NODES:
        # wiring.py の _fit と同じ計算で、縮小後の大きさが小さすぎないかを見る
        for text, base in ((jp, 12.5), (eng, 9.5)):
            w = _text_w(text, base)
            avail = BOX_W - 22
            if w > avail:
                shrunk = max(7.5, base * avail / w)
                if shrunk <= 8.0:      # これ以下は読めない
                    tight.append((nid, text, round(shrunk, 1)))
    print(f"【③】字が小さすぎる : {'なし' if not tight else f'{len(tight)}件'}")
    for nid, t, s in tight[:6]:
        print(f"      {nid}: 「{t}」→ {s}px")
    ng += len(tight)

    # ---- ④ 線が箱を横切るか（読みにくさの主因）------------------------------
    hit = []
    for a, b, label, ck in EDGES:
        if a not in pos or b not in pos:
            continue
        p1 = (pos[a][0] + BOX_W, pos[a][1] + BOX_H / 2)
        p2 = (pos[b][0], pos[b][1] + BOX_H / 2)
        for nid, (x, y) in pos.items():
            if nid in (a, b):
                continue
            if _seg_rect(p1, p2, (x, y, BOX_W, BOX_H)):
                hit.append((a, b, nid))
    print(f"【④】線が別の箱を横切る : {'なし' if not hit else f'{len(hit)}件'}")
    for a, b, n in hit[:8]:
        print(f"      {a}→{b} が {n} を横切る")

    # ---- ⑤ 線の交差の数（多いと読みにくい）-----------------------------------
    segs = []
    for a, b, label, ck in EDGES:
        if a in pos and b in pos:
            segs.append(((pos[a][0] + BOX_W, pos[a][1] + BOX_H / 2),
                         (pos[b][0], pos[b][1] + BOX_H / 2)))

    def cross(a, b, c, d):
        def s(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        d1, d2 = s(a, b, c), s(a, b, d)
        d3, d4 = s(c, d, a), s(c, d, b)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    n_cross = sum(1 for i in range(len(segs)) for j in range(i + 1, len(segs))
                  if cross(*segs[i], *segs[j]))
    print(f"【⑤】線どうしの交差 : {n_cross} 箇所"
          f"{'（多い＝読みにくい）' if n_cross > 40 else ''}")

    print("\n" + "-" * 78)
    if ng == 0 and not hit:
        print("  機械で分かる問題はありません（見やすさは人が見る必要あり）")
    else:
        print(f"  注意直すべき点 {ng + len(hit)} 件")
        print("     ④が多いときは、箱の並び順や列の割り当てを変えると減ります")
    return 0


if __name__ == "__main__":
    sys.exit(main())
