"""環境を作る。環境を組み立てるのはここだけ。

【なぜここだけにするか、2026-07-30】環境を組み立てる場所が2つあったため、
「学習は関節モード（90関節を独立に駆動）、測定・Viewerは筋肉モード（拮抗筋2本/関節）」
という**別の体で動いていた**事故が起きた。ユーザーの目視で発覚：

> そもそもなんで視線誘導反射の実験の時とは動きが全然違うの？速さが

実測で動きが人間の新生児の約3.3倍速かった。
⇒ 組み立てるのは1箇所。他は呼ぶだけ。

注意：駆動モデルは**渡さない**。`e_scene.build` の既定（筋肉モード）に従う。
  関節モードは逸脱リストの「逸脱5」に登録された逸脱なので、
  使うときは実験ファイルに明示的に書かせる（`taro.actuation: "joint"`）。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
for _p in (os.path.join(_ROOT, "E", "scripts"),
           os.path.join(_ROOT, "D", "scripts"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "taro_core", "src", "brain"),
           os.path.join(_ROOT, "taro_core", "src", "senses"),
           os.path.join(_ROOT, "taro_core", "src", "wrapper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build(scene_name, *, taro=None, seed=0, verbose=False, hybrid=False):
    """シーンの名前から環境を作る。

    Args:
        scene_name: E/scenes/<名前>.json の名前
        taro: 実験ファイルの `taro` 欄（月齢の上書き・駆動モードの指定に使う）
        seed: リセットの乱数の種
        hybrid: 内臓（内受容感覚）を足すか。**学習では必須**。
            【なぜ、2026-07-30】太郎の融合層（MinimalFusion）は観測の
            `interoception`（空腹・眠気・不快・覚醒）を島皮質(insula)経由で使う。
            HybridEnv で包まないとこのキーが観測に無く、脳の入力次元が合わない。
            注意：measure（脳を通さず環境だけ進める）では不要なので既定 False。
            包むと観測が変わる＝**包む/包まないで別の実験**になる。
    Returns:
        (env, scene, hands)
    """
    import e_scene
    taro = taro or {}
    sc = e_scene.load(scene_name)

    # 月齢は実験ファイルで上書きできる（体を育てる実験のため）。
    #   注意：上書きしたらシーンの指紋とは一致しないので照合を外す。
    if taro.get("age_months") is not None:
        sc["body"]["age_months"] = float(taro["age_months"])
        sc["fingerprint"] = None

    # 駆動モデル。既定（None）＝ e_scene の既定＝筋肉モード。
    act = None
    mode = str(taro.get("actuation", "muscle")).lower()
    if mode in ("joint", "spring", "springdamper", "torque"):
        from mimoActuation.actuation import SpringDamperModel
        act = SpringDamperModel
        print("注意[actuation] 関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の"
              "逸脱を選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]", flush=True)
    elif mode not in ("muscle", "muscles", ""):
        raise ValueError(f"taro.actuation が不明: {mode}（muscle か joint）")

    env, hands = e_scene.build(sc, orient=bool(taro.get("orienting_reflex", False)),
                               vor=bool(taro.get("vor", True)),
                               seed=seed, verbose=verbose, actuation_model=act)
    if hybrid:
        # 注意：内臓を足す（内受容感覚が観測に入る）。学習ではこれが要る。
        from hybrid_env import HybridEnv
        env = HybridEnv(env)
    return env, sc, hands
