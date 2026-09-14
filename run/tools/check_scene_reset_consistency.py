# -*- coding: utf-8 -*-
"""シーン保存時の姿勢と、実際に学習が始まるときの姿勢の整合性を確認する。

【なぜ要るか、2026-08-03】`E/scenes/新生児_仰向け_柵なし_伸展版.json` の検証
（作業記録（非公開））で、
**シーンファイルに保存された姿勢（`state.qpos`）と、実際に学習が始まるときに
Taroが使う姿勢が別物**であることが判明した。

    原因：`Taro.__init__`（`run/taro_setup.py:136`）が `scene.build()` の後に
         もう一度 `env.reset(seed=self.seed)` を呼んでおり、この2回目のresetは
         シーンの `state.qpos` ではなく、環境構築時に1度だけ計算される
         `self.init_position`（身体補正＋生理的屈曲/屈筋トーン＋settleの結果、
         `D/scripts/d_supine_env.py:152` の `reset_model`）へ戻る。

屈曲版ではこの2つがたまたま近い値だったため無害だったが、伸展版では
肘が113度も違う値になっていた。

【方針（新設当時）】`Taro.__init__` 側のリセットの仕組み自体は共通ファイル
（`run/taro_setup.py`）であり、目標C・D・Eの全実験に影響するため変更しない。
代わりに、この食い違いを**検証段階で必ず検出できるツール**としてこれを作った。

【追記・2026-08-03 同日】上の方針のあと、真因を遡って調べたところ、
`self.init_position`（`D/scripts/d_supine_env.py`）が「シーンの姿勢を流し込む
より前」に記録されていることが分かった。これは`run/taro_setup.py`ではなく
**目標E固有の`E/scripts/scene_io.py`の`build()`側の問題**だったため、
`build()`の最後で`init_position`を更新する1行を追加して**根本修正した**
（`run/taro_setup.py`・`run/trainer.py`は変更していない）。
詳細は`doc/検証の落とし穴チェックリスト.md`項95、`E/docs/研究日誌.md`同日の記録。
このツール自体は「修正後も不一致が起きていないか」を確認する回帰テストとして
今後も使う（実際、修正後の再実行で肩関節にのみ別の残存現象を発見した）。

【比較する2つの姿勢】
    状態A：scene.build() 直後の姿勢
           （シーンファイルの state.qpos が実際に反映された状態。
            `run/plugins/common/scene.py` の `build()` を呼んだ直後）
    状態B：Taro.__init__ 構築後の姿勢
           （2回目のresetを経た、実際に学習で使われる状態。
            `self.init_position` に相当する状態）

関節ごとに「状態A」「状態B」「差（度）」を表にして、差が
`THRESHOLD_DEG`（既定5度）以上あれば「不一致：学習ではこちらが使われます」と表示する。
最低限「肘・肩・首・体幹」（`FOCUS_GROUPS`）を含む全関節を比較する
（部位の分類は `E/scripts/e_head_hold.py` の `JOINT_GROUPS` をそのまま使う。
定数を別の場所に複製しない。落とし穴チェックリスト 項30）。

【この道具がやらないこと】
    ・修正はしない（読み取って比較するだけ。taro_core・run/trainer.py・
      run/taro_setup.py 等の本体コードは一切変更しない）
    ・学習は回さない（scene.build()とTaro構築だけ。数秒で終わる）

使い方::

    .venv/Scripts/python.exe run/tools/check_scene_reset_consistency.py
    .venv/Scripts/python.exe run/tools/check_scene_reset_consistency.py シーン名1 シーン名2
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
_R = r"C:\claude\AI\Taro"
os.chdir(_R)
if _R not in sys.path:
    sys.path.insert(0, _R)

from run.config import Config                      # noqa: E402
from run.plugins.common import scene as scene_mod   # noqa: E402
from run.taro_setup import Taro                     # noqa: E402

# scene_mod の import で E/scripts が sys.path に入るので、ここで初めて読める
import scene_io                                      # noqa: E402
from e_head_hold import joint_group                 # noqa: E402

# 仕様2節「最低限『肘・肩・首・体幹』を含む」に対応する部位
#   （分類そのものは e_head_hold.JOINT_GROUPS を流用。ここで定義を複製しない）
FOCUS_GROUPS = ("head", "trunk", "arm")

# 差の警告しきい値。仕様2節「差が一定以上（目安：5度以上）あれば警告」
THRESHOLD_DEG = 5.0


def _snapshot_joints(env):
    """joint名（"robot:"・左右接頭辞は残す）→角度(度) の辞書。

    注意：scene_io.snapshot_state() が既に持っているロジック
      （ヒンジ/スライド関節だけ・"robot:"接頭辞を外す・眼球を除く）をそのまま使う。
      同じ式をここで書き直すと、片方だけ直し忘れて食い違う（落とし穴チェックリスト 項30）。
    """
    return dict(scene_io.snapshot_state(env)["joints_readable"])


def check_scene(scene_name, seed=0, taro_kwargs=None, verbose=False):
    """1つのシーンについて、状態A・状態Bを実測して差を返す。

    Returns:
        (scene_name, rows, taro_obj)  rows は関節ごとの dict のリスト
    """
    taro_kwargs = dict(taro_kwargs or {})
    taro_kwargs.setdefault("vision", False)   # 検証を速くするためだけの指定。姿勢には無関係
    cfg = Config.from_spec({"scene": scene_name, "taro": taro_kwargs,
                            "run": {"steps": 10, "seed": seed}})

    # ---- scene.build()（run/plugins/common/scene.py 経由。組み立てはここだけ）------
    env, sc, hands = scene_mod.build(cfg.scene, taro=dict(cfg._taro), seed=seed,
                                     verbose=verbose, hybrid=True)

    # ---- 状態A：scene.build() 直後（シーンの state.qpos が反映された状態）----------
    state_a = _snapshot_joints(env)

    # ---- 状態B：Taro.__init__ 構築後（2回目のresetを経た、学習で使われる状態）-------
    #   注意：Taro.__init__ 内の env.reset(seed=self.seed)（run/taro_setup.py:136）が
    #     ここで走る。taro_setup.py は一切変更していない（読んで比較するだけ）。
    taro_obj = Taro(cfg, env, seed=seed, verbose=verbose)
    state_b = _snapshot_joints(env)

    env.close()

    names = sorted(set(state_a) | set(state_b))
    rows = []
    for nm in names:
        a = state_a.get(nm)
        b = state_b.get(nm)
        diff = None if (a is None or b is None) else (b - a)
        rows.append({"joint": nm, "group": joint_group(nm), "A": a, "B": b, "diff": diff})
    return scene_name, rows, taro_obj


def print_report(scene_name, rows):
    """関節ごとの比較表を表示し、不一致数（全体・focus部位）を返す。"""
    print("=" * 92)
    print(f" シーン「{scene_name}」： scene.build()直後(状態A) vs Taro.__init__後(状態B)")
    print("=" * 92)
    print(f"{'関節':30s} {'部位':6s} {'状態A(度)':>12s} {'状態B(度)':>12s} {'差(度)':>10s}  判定")
    print("-" * 92)
    n_warn = 0
    n_focus_warn = 0
    n_focus = 0
    for r in rows:
        a_s = "?" if r["A"] is None else f"{r['A']:.2f}"
        b_s = "?" if r["B"] is None else f"{r['B']:.2f}"
        is_focus = r["group"] in FOCUS_GROUPS
        n_focus += int(is_focus)
        if r["diff"] is None:
            d_s, verdict = "?", "不明（片方に無い関節）"
        else:
            d = r["diff"]
            d_s = f"{d:+.2f}"
            if abs(d) >= THRESHOLD_DEG:
                verdict = "不一致：学習ではこちらが使われます"
                n_warn += 1
                n_focus_warn += int(is_focus)
            else:
                verdict = "一致"
        mark = "*" if is_focus else " "
        print(f"{mark}{r['joint']:29s} {r['group']:6s} {a_s:>12s} {b_s:>12s} {d_s:>10s}  {verdict}")
    print("-" * 92)
    print(f"  （*＝肘・肩・首・体幹＝仕様2節の対象部位。全{n_focus}関節）")
    print(f"  しきい値={THRESHOLD_DEG:g}度  不一致の関節数={n_warn}/{len(rows)}"
          f"（うち肘・肩・首・体幹の不一致={n_focus_warn}）")
    if n_warn:
        print("  → このシーンは Taro.__init__ の2回目の reset(seed) で、"
              "保存された姿勢と異なる姿勢に切り替わっています。")
    else:
        print("  → 保存された姿勢と、学習で使われる姿勢はほぼ一致しています。")
    print()
    return n_warn, n_focus_warn


def main():
    scenes = sys.argv[1:] or [
        "新生児_仰向け_柵なし_伸展版",
        "新生児_仰向け_柵なし",
        "リーチング_リクライニング60度",
    ]
    summary = []
    for nm in scenes:
        try:
            scene_name, rows, _ = check_scene(nm)
        except Exception as e:      # noqa: BLE001
            print(f"[エラー] シーン「{nm}」の確認中に例外: {type(e).__name__}: {e}")
            summary.append((nm, None, None))
            continue
        n_warn, n_focus_warn = print_report(scene_name, rows)
        summary.append((scene_name, n_warn, n_focus_warn))

    print("=" * 92)
    print(" まとめ")
    print("=" * 92)
    for nm, n_warn, n_focus_warn in summary:
        if n_warn is None:
            print(f"  {nm}: 確認できず（上記エラー参照）")
        else:
            print(f"  {nm}: 不一致関節数={n_warn}（うち肘・肩・首・体幹={n_focus_warn}）")


if __name__ == "__main__":
    main()
