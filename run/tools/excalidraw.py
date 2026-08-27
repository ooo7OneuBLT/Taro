"""Excalidraw(.excalidraw)ファイルを組み立てる最小ヘルパー（2026-08-22新設）。

目的：構造や実験設計のイメージを、ユーザーと**双方向**で共有する。
  ①こちらが図を生成 → ②ユーザーがExcalidrawで直接編集・追記 → ③保存
  → ④こちらが読み戻して意図を汲む、という往復を成立させる。

Excalidrawで開く方法（どちらでもよい）：
  - https://excalidraw.com を開き、.excalidraw ファイルをドラッグ＆ドロップ
  - VS Code拡張「Excalidraw」をインストールしてファイルを直接開く（ローカル完結・推奨）

【なぜローカル完結を推奨するか】太郎は未公開の研究であり、商用化も視野に入っている
（方針2026-08-19）。図をクラウドの共同編集に載せると外部へ出る。VS Code拡張なら
ファイルはPC内に留まる。
"""
import json

_N = 0
def _nid():
    global _N; _N += 1; return f"el{_N}"

_COMMON = dict(
    isDeleted=False, fillStyle="solid", strokeWidth=2, strokeStyle="solid",
    roughness=1, opacity=100, angle=0, groupIds=[], frameId=None,
    boundElements=[], link=None, locked=False, version=1, versionNonce=1,
    updated=1, seed=1,
)

def box(x, y, w, h, bg="#ffffff", stroke="#1e1e1e"):
    return dict(_COMMON, id=_nid(), type="rectangle", x=x, y=y, width=w, height=h,
                backgroundColor=bg, strokeColor=stroke, roundness={"type": 3})

def text(x, y, s, size=16, color="#1e1e1e", w=None):
    lines = s.split("\n")
    w = w if w is not None else int(max(len(l) for l in lines) * size * 0.62)
    return dict(_COMMON, id=_nid(), type="text", x=x, y=y,
                width=w, height=int(size * 1.25 * len(lines)),
                text=s, originalText=s, fontSize=size,
                fontFamily=2,            # 2=Normal(Helvetica)。日本語が化けないのはこれ
                textAlign="left", verticalAlign="top",
                strokeColor=color, backgroundColor="transparent",
                containerId=None, lineHeight=1.25, baseline=int(size))

def arrow(x1, y1, x2, y2, color="#1e1e1e", label=None):
    return dict(_COMMON, id=_nid(), type="arrow", x=x1, y=y1,
                width=abs(x2 - x1), height=abs(y2 - y1),
                points=[[0, 0], [x2 - x1, y2 - y1]],
                strokeColor=color, backgroundColor="transparent",
                startBinding=None, endBinding=None,
                startArrowhead=None, endArrowhead="arrow", roundness={"type": 2})

def save(elements, path, bg="#fdfcfa"):
    doc = {"type": "excalidraw", "version": 2,
           "source": "taro/run/tools/excalidraw.py",
           "elements": elements,
           "appState": {"viewBackgroundColor": bg, "gridSize": None},
           "files": {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return path
