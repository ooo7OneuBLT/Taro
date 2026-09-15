"""視線誘導反射（Orienting Reflex）：周辺視野の動きへ、首・目を反射的に向ける。

【なぜ作るか】新生児は周辺の動きへ反射的に定位する。この反射は**上丘**（皮質下）が
媒介し、皮質（＝方策）を経由しない（`doc/やることリスト.md`2026-07-20の検討項目、
2026-07-21に文献確認して確定）。

【人間での根拠】
 ・新生児は視野端30度（横・斜め軸）にある対象へ、正しい方向への最初の跳躍運動を
   確実に行う。ただし**縦軸は10度までと弱い**＝横の反応の方がずっと強い。
   Aslin & Salapatek (1975) "Saccadic localization of visual targets by the very
   young human infant" https://link.springer.com/article/10.3758/BF03203214
 ・生後2〜5日の新生児でも"見かけの動き"がある場合にだけ注意が向く＝動き依存・
   対象非依存（対象が何かを知る必要はない）。
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4851695/
 ・新生児は7週児より反応が弱く、「受動的に捕まるのではなく本人がある程度サンプリング
   を内部制御している」。Harris & MacFarlane (1974)
   https://www.sciencedirect.com/science/article/abs/pii/0022096574901131
 ・首と目は一緒に動き、首の動きの方が先に始まることが多い。対象が視界の端に近い
   （周辺）ほど首を動かす傾向が強い（coordination of eye and head movements,
   PubMed 6357223）。

【守るべき原則（Baby Sophia 2025 の教訓）】
 「対象(手など)を検出して視線を向ける」＝対象が何かを知る仕組みは、創発の検証を
 壊す（先行研究 Baby Sophia (2025) は手を検出する方式で片手固着＝反対の手の可視率
 0.0%という失敗に終わった）。今回は「**動きの大きい方向**」という、対象が何かを
 知らない低次の反射のみを実装する。

【注意工学的対処・簡略化（ラベリング対象。詳細は doc/人間模倣からの逸脱リスト.md）】
 1. 視野を6マス（横3×縦2）に分けて動きを検出する方法自体は、上丘の実際の計算
    （視野全体の精密な空間地図）を大幅に簡略化した代用。
 2. 太郎自身が首を回すと、回転運動は遠近に関係なく画面全体をほぼ均等にずらすため、
    単純に「一番変化したマスを見る」だけでは自分の動きと他者の動きを区別できない
    （ユーザー指摘、2026-07-21）。対処として、6マスの**平均を引いた残差**を使う
    （自分の回転による均等な底上げを消し、他者の動きだけが残差として際立つ）。
    この計算方法自体に文献的根拠はなく、私が組んだ工学的解法。
 3. 縦方向の反応を横の`VERTICAL_DAMPING`倍に弱めるのは文献の比率(10/30)に基づくが、
    具体的な減衰率の値は近似。
 4. 目はVOR（`e_vor.py`）の指令に、この反射の分を**足し算**する。解剖学的に別回路が
    最終共通出力（外眼筋）でのみ合流するという構造を再現する意図（VORの指令自体は
    書き換えない）。首はbabble方策の出力に同様に加算する。
 5. 強さ（NECK_GAIN/EYE_GAIN）は文献に基準がなく試行錯誤で決める＝[ARBITRARY]。
"""
import numpy as np

# 関節への指令を、身体の駆動方式（筋肉2本／モーター1つ）に合った形で書き込む共通の写像。
import os as _os, sys as _sys
# 【2026-09-13・直し】ここは既に taro_core/src/brain/midbrain なので、
#   1つ上がれば brain。以前は「2つ上がって taro_core/src/brain を足す」と
#   書いてあり、**存在しないパス**（taro_core/src/taro_core/src/brain）を
#   足していた。呼び出し側（run/taro_setup.py）が正しい道を先に通していたので
#   気づかれなかった。単体で import されると落ちる。
_CORE_BRAIN = _os.path.abspath(_os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
if _CORE_BRAIN not in _sys.path:
    _sys.path.insert(0, _CORE_BRAIN)
from spinal_cord.cpg import write_joint_command as _write_joint_command

GRID_COLS = 3
GRID_ROWS = 2
VERTICAL_DAMPING = 10.0 / 30.0   # 文献の反応角度比（縦10度/横30度）。値は近似[ARBITRARY]
NECK_GAIN = 0.3                  # [ARBITRARY] 試行錯誤で決める暫定値
EYE_GAIN = 0.15                  # [ARBITRARY] VORの上に足す小さい成分


class OrientingReflex:
    """視野の残差（自分の回転による均等なズレを引いた残り）から、動きの方向へ首・目を向ける。"""

    def __init__(self, model):
        self.prev_eye = None
        self.n_actuator = int(model.nu)
        self.neck_idx = {}
        self.eye_idx = {"h": [], "v": []}
        for i in range(model.nu):
            name = model.actuator(i).name
            if name == "act:head_swivel":
                self.neck_idx["h"] = i
            elif name == "act:head_tilt":
                self.neck_idx["v"] = i
            elif "eye" in name and "horizontal" in name:
                self.eye_idx["h"].append(i)
            elif "eye" in name and "vertical" in name:
                self.eye_idx["v"].append(i)

    def reset(self):
        """引数・返り値は無い。内部状態を初期化する。前回の目の画像キャッシュ(prev_eye)をNoneに戻し、水平・垂直の反射方向(h_dir, v_dir)を0.0にする。
        """
        self.prev_eye = None
        self.h_dir = 0.0
        self.v_dir = 0.0

    def update(self, eye_image):
        """新しい画像が描画されたときに呼ぶ（毎物理stepではない、視覚のキャッシュ間隔なり）。
        方向を計算してキャッシュする。実際のactionへの反映はapply()で行う。"""
        self.h_dir, self.v_dir = self._direction(eye_image)

    def apply(self, action):
        """update()でキャッシュした方向を、actionに足して返す（首は加算、目はVOR適用後に加算）。"""
        h_dir, v_dir = getattr(self, "h_dir", 0.0), getattr(self, "v_dir", 0.0)
        if h_dir == 0.0 and v_dir == 0.0:
            return action
        out = np.array(action, dtype=float).copy()
        # 2026-07-26：直接 out[i] に符号つきで書いていたのが誤り。MuscleModel では
        #   1関節が2本の筋で駆動され各要素は [0,1] に切り捨てられるため、負の指令が消えて
        #   **片方向にしか動けなかった**。詳細は e_vor.py と cpg.write_joint_command 参照。
        n = self.n_actuator
        for key in ("h", "v"):
            if key in self.neck_idx:
                d = h_dir if key == "h" else v_dir
                _write_joint_command(out, self.neck_idx[key], NECK_GAIN * d, n,
                                     co_activation=0.0, additive=True)
        for i in self.eye_idx["h"]:
            _write_joint_command(out, i, EYE_GAIN * h_dir, n,
                                 co_activation=0.0, additive=True)
        for i in self.eye_idx["v"]:
            _write_joint_command(out, i, EYE_GAIN * v_dir, n,
                                 co_activation=0.0, additive=True)
        return out

    def _direction(self, eye_image):
        """画像から (横方向, 縦方向) の残差重み付き方向を計算する。おおむね[-1, 1]。"""
        cur = np.asarray(eye_image, dtype=np.float32)
        if self.prev_eye is None or self.prev_eye.shape != cur.shape:
            self.prev_eye = cur
            return 0.0, 0.0
        diff = np.abs(cur - self.prev_eye)
        self.prev_eye = cur
        h, w = diff.shape[0], diff.shape[1]
        rows = np.array_split(np.arange(h), GRID_ROWS)
        cols = np.array_split(np.arange(w), GRID_COLS)
        cell_vals = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
        for r, rid in enumerate(rows):
            for c, cid in enumerate(cols):
                cell_vals[r, c] = diff[rid[:, None], cid].mean()
        residual = np.clip(cell_vals - cell_vals.mean(), 0, None)  # 自分の動き分を引き、際立つ分だけ残す
        total = residual.sum()
        if total < 1e-6:
            return 0.0, 0.0
        col_centers = (np.arange(GRID_COLS) - (GRID_COLS - 1) / 2) / ((GRID_COLS - 1) / 2)
        row_centers = (np.arange(GRID_ROWS) - (GRID_ROWS - 1) / 2) / ((GRID_ROWS - 1) / 2)
        h_dir = float((residual.sum(axis=0) * col_centers).sum() / total)
        v_dir = float((residual.sum(axis=1) * row_centers).sum() / total) * VERTICAL_DAMPING
        return h_dir, v_dir

    def override(self, action, eye_image):
        """update()+apply()をまとめて行う便利関数（単体テスト・簡易利用向け）。
        実環境(e_toy_env.py)では描画のタイミングが違うため、update()とapply()を別々に呼ぶ。"""
        self.update(eye_image)
        return self.apply(action)
