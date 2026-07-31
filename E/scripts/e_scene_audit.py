"""シーンに「入れ忘れ」が無いかを機械的に照合する。

【なぜ要るか、2026-07-29】シーン方式に移した初日に、**反射のバージョン**
（`E_ORIENT_V`）をシーンに入れ忘れていたことが分かった。既定は旧版 v1 なのに
16個の測定スクリプトがそれぞれ `E_ORIENT_V=2` を指定していて、
指定を忘れたコードは**黙って旧版で走る**。実際に私（Claude）が踏んだ。

注意：入れ忘れを当てずっぽうで探さない。**環境側のコードから機械的に列挙**して、
  シーンがカバーしているかを突き合わせる。
  ＝落とし穴チェックリストの方針「バグは直して終わりにせず、
    原因を特定してチェックに変える」（[[feedback-bug-to-checklist]]）。

【判定の3分類】
    カバー済み   シーンが値を決めている（環境変数を設定する／引数で直接渡す）
    意図的に外す 実験の条件（反射のON/OFF・乱数の種）や道具の設定（Viewerの速度）
    未カバー   どちらでもない＝入れ忘れの候補

使い方:
    .venv/Scripts/python.exe E/scripts/e_scene_audit.py
"""

# 注意：古い方式（2026-07-30 に整理）。新しい実験は `run/main.py` を通す。
#   【経緯】目標Eの実験スクリプトが118本あり、うち66本が**独立に環境を組み立てていた**。
#     そのため「学習は関節モード（90関節を独立に駆動＝逸脱リスト 逸脱5）、
#     測定とViewerは筋肉モード（拮抗筋2本/関節）」という**別の体で動く**事故が起きた
#     （ユーザーの目視「視線誘導反射の実験の時とは動きが全然違う」で発覚。
#      実測で動きが人間の新生児の約3.3倍速かった）。
#   【設計と移行計画】`E/docs/実行基盤_設計.md`
#   注意：このファイルは**記録として残す**（削除しない方針）。
#     中の測り方は再利用できるので、プラグインへ移すときの元にする。
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 環境（＝太郎の体と世界）を組み立てるときに読まれる設定があるファイル
SOURCES = [
    "E/scripts/e_toy_env.py",
    "D/scripts/d_supine_env.py",
    "E/scripts/e_body_config.py",
    "taro_core/src/body/infant_body.py",
    "taro_core/src/body/infant_neck.py",
    "taro_core/src/body/infant_limbs.py",
    "E/scripts/e_orienting_v2.py",
    "E/scripts/e_vor.py",
]

# 実験の条件・道具の設定＝シーンに入れない（入れると比較ができなくなる）
INTENTIONAL = {
    "E_ORIENT": "反射のON/OFF＝同じシーンで比べるのが実験",
    "E_VOR": "前庭動眼反射のON/OFF＝同上",
    "E_ORIENT_SEED": "反射の神経ノイズの種",
    "E_SEED": "物理の初期ゆらぎの種",
    "E_SCENE": "シーンの指定そのもの",
    "E_FREEZE": "Viewerで物理を止めるか",
    "E_SPEED": "Viewerの再生速度",
    "E_EDIT": "Viewerの編集モード",
    "E_SHAKE": "Viewerでおもちゃを揺らす",
    "E_SHAKE_AMP": "同上",
    "E_SHAKE_HZ": "同上",
}
# 反射の内部のつまみ＝実験ごとに振るためのもの。シーンではなく実験条件。
TUNING_PREFIX = ("E_SACC", "E_OMS", "E_EYE_S", "E_NECK_SHARE", "E_LI_",
                 "E_TIME_", "E_EYE_STILL", "E_ORIENT_BIAS", "E_SC_")


def env_vars_read():
    """環境の組み立てで読まれる環境変数を、ファイルごとに集める。"""
    out = {}
    for rel in SOURCES:
        path = os.path.join(_ROOT, rel)
        if not os.path.exists(path):
            continue
        src = open(path, encoding="utf-8").read()
        for mm in re.finditer(
                r'environ\.(?:get|setdefault)\(\s*["\'](E_[A-Z0-9_]+)["\']', src):
            out.setdefault(mm.group(1), set()).add(os.path.basename(rel))
    return out


def scene_coverage():
    """シーンがカバーしている設定を2通りで集める。

    注意：**環境変数を設定するものだけ**を数えてはいけない。体型・四肢・屈曲は
      環境変数を経由せず `_body_kwargs` が引数で直接渡している
      （呼び出し側が上書きできないようにするため）。
      これを見落とすと「入れ忘れ」を7件も誤検出する（2026-07-29 に実際にやった）。
    """
    src = open(os.path.join(_HERE, "e_scene.py"), encoding="utf-8").read()
    by_env = set(re.findall(r'os\.environ\["(E_[A-Z0-9_]+)"\]', src))
    # 引数で直接渡しているもの＝シーンの項目名 → 対応する環境変数
    by_arg = {
        "E_SHAPE": "body.shape",
        "E_HEAD_ELONG": "body.head_elongation",
        "E_FLEXION": "body.flexion",
        "E_FLEX_STIFF": "body.flexion_stiffness",
        "E_LIMB_SCALE": "body.limb_scale",
        "E_LIMB_FIX": "body.limb_fix",
        "E_DISTAL_MASS": "body.distal_mass",
        "E_AGE": "body.age_months",
    }
    return by_env, by_arg


def main():
    read = env_vars_read()
    by_env, by_arg = scene_coverage()

    covered, skipped, missing = [], [], []
    for v, files in sorted(read.items()):
        if v in by_env:
            covered.append((v, "環境変数で渡す"))
        elif v in by_arg:
            covered.append((v, f"引数で直接渡す（{by_arg[v]}）"))
        elif v in INTENTIONAL:
            skipped.append((v, INTENTIONAL[v]))
        elif v.startswith(TUNING_PREFIX):
            skipped.append((v, "反射のつまみ＝実験の条件"))
        else:
            missing.append((v, "、".join(sorted(files))))

    print("=" * 78)
    print(" シーンの入れ忘れ照合")
    print("=" * 78)
    print(f"  環境が読む設定    {len(read)} 件")
    print(f"  シーンがカバー    {len(covered)} 件")
    print(f"  意図的に外す      {len(skipped)} 件")
    print(f"  未カバー        {len(missing)} 件")

    print("\n" + "-" * 78)
    print(" 意図的に外しているもの（実験の条件・道具の設定）")
    print("-" * 78)
    for v, why in skipped:
        print(f"  {v:<22} {why}")

    if missing:
        print("\n" + "=" * 78)
        print(" シーンに入っていない設定＝入れ忘れの候補")
        print("=" * 78)
        for v, f in missing:
            print(f"  {v:<22} {f}")
        print("\n  ⇒ `e_scene.default_scene()` に項目を足し、`build()` で値を渡すこと。")
        print("     実験の条件なら INTENTIONAL に理由つきで登録すること。")
        return 1

    print("\n" + "=" * 78)
    print(" 入れ忘れは無い — 環境の設定はすべてシーンが決めている")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
