"""既存のプリセットからシーンファイルを作る（シーン方式への引っ越し）。

【なぜ要るか、2026-07-29】これまでの環境の条件は
コードの定数・環境変数・Viewerの `PRESETS`・`viewer_saved.json` の4か所に
散らばっていた。それを1つのシーンファイルへ集める。

姿勢（qpos）は「作ってリセットし、少し落ち着かせた状態」を保存する。
そこから先はViewerで人が調整して保存し直す（それが本来の使い方）。

使い方:
    .venv/Scripts/python.exe E/scripts/e_scene_make.py            # 全部作る
    .venv/Scripts/python.exe E/scripts/e_scene_make.py 名前       # 1つだけ
"""

# ⚠️★古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   ⚠️このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import sys
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np      # noqa: E402
import e_scene          # noqa: E402


def _scene(name, note, body=None, world=None, setup=None):
    sc = e_scene.default_scene(name)
    sc["note"] = note
    for key, over in (("body", body), ("world", world), ("setup", setup)):
        if over:
            sc[key] = e_scene._merge(sc[key], over)
    return sc


# ============================================================================
# 引っ越し元＝`e_viewer.py` の PRESETS と、2026-07-28 までの起動コマンド
# ============================================================================
SCENES = [
    _scene(
        "視線誘導反射_仰向け_頭を支える",
        "4ヶ月・仰向け・実験者が頭を支える。3シードで反射の機能を確認済み"
        "（反射ON 3.2度 vs OFF 10.2度）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": 0.0},
        world={"recline_deg": 0.0, "fence": False,
               "toy": {"shape": "sphere", "radius": 0.0056, "mode": "hold",
                       "dist": 0.086, "delay_sec": 0.0}},
        setup={"head_hold": {"stiffness": 200.0, "target_deg": None}},
    ),
    _scene(
        "リーチング_リクライニング60度",
        "4ヶ月・リクライニング60度・実験者が頭を60度で支える（顎を引く）・眼球-15度。"
        "⚠️おもちゃの位置は未確定（遮蔽で候補が体に隠れる）",
        body={"age_months": 4.0, "eye_rest_vertical_deg": -15.0},
        world={"recline_deg": 60.0, "seat_friction": 2.0, "fence": False,
               "toy": {"shape": "sphere", "radius": 0.0056, "mode": "hold",
                       "dist": 0.18, "delay_sec": 0.0}},
        setup={"head_hold": {"stiffness": 200.0,
                             "target_deg": {"head_tilt": 60.0}},
               # ★肩から手先と指と眼球だけを自由にし、それ以外は椅子とベルトが支える
               #   （ユーザーの提案 2026-07-29）。リーチングは腕と目でやる動作なので、
               #   体幹や脚が動いて姿勢が崩れるのを実験条件として止める。
               #   ⚠️脚の固定は人間からの逸脱（現実の乳児は椅子でも脚は自由）。
               "body_support": {"stiffness": 200.0, "free": ["arm", "finger"]}},
    ),
    _scene(
        "新生児_仰向け_柵あり",
        "0ヶ月・仰向け・柵あり。運動性喃語（自発運動）の学習で使ってきた環境",
        body={"age_months": 0.0, "flexion": True},
        world={"recline_deg": 0.0, "fence": True,
               "toy": {"shape": "box", "radius": 0.020, "mode": "tether",
                       "dist": 0.086, "delay_sec": 1.0}},
        setup={"neck_tone": {"target_deg": None, "stiffness": None}},
    ),
    _scene(
        "新生児_仰向け_柵なし",
        "0ヶ月・仰向け・柵なし・おもちゃなし。2026-07-31 新設。"
        "柵を外した理由：柵の寸法（±31cm×±16cm）は新生児に合わせたもので、"
        "体を育てる実験で見直されていなかった。実測すると4ヶ月の体は柵の左右幅の97%を占め"
        "（余裕8mm）、頭と目が柱に当たったまま学習していた。しかも0ヶ月でも"
        "腕を広げた幅34.0cm > 柵の内寸29.6cm。"
        "柵の目的はコードのコメントによれば『長時間の学習で遠くへ行かない保険』であって"
        "学習を助けるものではない。半径スイープの実測でも柵なしが最もズレが小さかった"
        "（なし0.088m / 0.23m柵0.163m）。"
        "おもちゃを外した理由：tetherで吊るすので揺れて動く＝自分の動き以外の変化が入る。"
        "自己モデル（自分の行動の結果を予測する）の学習には不要。"
        "⚠️柵なしで長時間動かすと遠くへ行く可能性がある。Viewerで目視して確かめること。",
        body={"age_months": 0.0, "flexion": True},
        world={"recline_deg": 0.0, "fence": False,
               "toy": {"enabled": False}},
        setup={"neck_tone": {"target_deg": None, "stiffness": None}},
    ),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    made = []
    for sc in SCENES:
        if only and sc["name"] != only:
            continue
        print("=" * 74)
        print(f" {sc['name']}")
        print("=" * 74)
        print(f"  {sc['note']}")
        env, hands = e_scene.build(sc, orient=False, vor=True, seed=0, verbose=False)
        # 物理を少し進めて落ち着かせてから保存する
        #   ＝作った直後は力が釣り合っておらず、そのまま保存すると
        #     測定を始めた瞬間に動いてしまう
        u = env.unwrapped
        dt = float(u.model.opt.timestep) * int(u.frame_skip)
        a = np.zeros(env.action_space.shape[0], dtype=np.float32)
        for _ in range(int(2.0 / dt)):
            env.step(a)
        # ★★【2026-07-29】落ち着いた**後**に、おもちゃを視線の正面へ置き直す。
        #
        # 【なぜ要るか】これを入れないと、おもちゃは「リセット直後の視線の正面」に
        # 置かれ、そのあと2秒ぶん頭が動いた分だけ視野の端へ寄る。
        # 実測：視線誘導反射の測定で初期ずれが **+0.44 ぶん右へ偏り**、
        #   視野の右端（+4.0cm）が見えなくなった（移行前は見えていた）。
        # 移行前の環境は「1秒待ってから親が運んでくる」ので、
        # **運び終わった時点＝落ち着いた姿勢での正面**に置かれていた。それに揃える。
        # ⚠️★体を留める設定があるシーンでは、ここで留め直してはいけない。
        #   `build` が**落ち着かせる前の姿勢**で既に留めている。それが正しい。
        #   落ち着かせてから留め直すと「転がり落ちた位置」で固定されてしまう
        #   （2026-07-29 の実測：リクライニング60度で頭が中心から25cm 横へずれ、
        #     視線が真後ろを向いた状態のまま保存された）。
        if u._toy and hasattr(u, "_set_anchor"):
            u._set_anchor()
            e_scene.place_toy(env, u._rest_pos)
            for _ in range(int(0.3 / dt)):
                env.step(a)
        saved, drift = e_scene.save(sc, env=env, hands=hands,
                                    settle_seconds=3.0, verbose=True)
        fp = saved["fingerprint"]
        print(f"  首 {fp['head_angles_deg']}")
        print(f"  体幹の傾き {fp['trunk_tilt_deg']}度")
        if fp.get("toy_pos"):
            print(f"  おもちゃ {fp['toy_pos']}  "
                  f"目から{fp['toy_dist_cm']}cm  視線から{fp['toy_angle_deg']}度  "
                  f"{'見える' if fp.get('toy_visible_left') else '見えない'}  "
                  f"腕の{(fp.get('toy_reach_ratio') or 0)*100:.0f}%")
        env.close()
        made.append(sc["name"])
        print()
    print("=" * 74)
    print(f" {len(made)}件のシーンを作った → E/scenes/")
    print("=" * 74)
    for n in made:
        print(f"  {n}")


if __name__ == "__main__":
    main()
