"""環境を作る。環境を組み立てるのはここだけ。

【なぜここだけにするか、2026-07-30】環境を組み立てる場所が2つあったため、
「学習は関節モード（90関節を独立に駆動）、測定・Viewerは筋肉モード（拮抗筋2本/関節）」
という**別の体で動いていた**事故が起きた。ユーザーの目視で発覚：

> そもそもなんで視線誘導反射の実験の時とは動きが全然違うの？速さが

実測で動きが人間の新生児の約3.3倍速かった。
⇒ 組み立てるのは1箇所。他は呼ぶだけ。

注意：駆動モデルは**渡さない**。`scene_io.build` の既定（筋肉モード）に従う。
  関節モードは逸脱リストの「逸脱5」に登録された逸脱なので、
  使うときは実験ファイルに明示的に書かせる（`taro.actuation: "joint"`）。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir))
for _p in (os.path.join(_ROOT, "run", "scene_tools"),
           os.path.join(_ROOT, "D", "scripts"),
           os.path.join(_ROOT, "taro_core", "src", "body"),
           os.path.join(_ROOT, "taro_core", "src", "brain"),
           os.path.join(_ROOT, "taro_core", "src", "senses"),
           os.path.join(_ROOT, "taro_core", "src", "wrapper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def resolve(scene_name, taro=None):
    """シーンを読んで、実験ファイルの `taro` 欄による上書きまで済ませた辞書を返す。

    【なぜ切り出したか、2026-09-14】走行の記録（`run.meta.json`）に
    「実際に使った場面」を丸ごと残したいが、その中身は `build()` の中でしか
    作られていなかった。同じ処理を記録側でもう一度書くと**二重の真実**になるので、
    ここを唯一の場所にして `build()` と記録の両方から呼ぶ。
    物理は組み立てない（JSONを読むだけ）ので軽い。
    """
    import scene_io
    taro = taro or {}
    sc = scene_io.load(scene_name)

    # 月齢は実験ファイルで上書きできる（体を育てる実験のため）。
    #   注意：上書きしたらシーンの指紋とは一致しないので照合を外す。
    if taro.get("age_months") is not None:
        sc["body"]["age_months"] = float(taro["age_months"])
        sc["fingerprint"] = None
    return sc


def build(scene_name, *, taro=None, seed=0, verbose=False, hybrid=False):
    """シーンの名前から環境を作る。

    Args:
        scene_name: run/scenes/<名前>.json の名前
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
    import scene_io
    taro = taro or {}
    sc = resolve(scene_name, taro)

    # 駆動モデル。既定（None）＝ scene_io の既定＝筋肉モード。
    act = None
    mode = str(taro.get("actuation", "muscle")).lower()
    if mode in ("joint", "spring", "springdamper", "torque"):
        from mimoActuation.actuation import SpringDamperModel
        act = SpringDamperModel
        print("注意[actuation] 関節モード（90関節を独立に駆動）＝逸脱リスト 逸脱5 の"
              "逸脱を選んでいます。人間の新生児は拮抗筋を同時に力ませる[Tier1]", flush=True)
    elif mode not in ("muscle", "muscles", ""):
        raise ValueError(f"taro.actuation が不明: {mode}（muscle か joint）")

    orient = bool(taro.get("orienting_reflex", False))
    vision_on = bool(taro.get("vision", True))
    # 【2026-09-13新設・設計_視覚を脳へ戻す 第2段】視覚と注意（VisualAttention）を
    #   環境（太郎の側）に持たせる設定。`taro.orienting_reflex` と同じ道を通す：
    #   ここで実験ファイルの `taro` 欄から読み、scene_io.build() → ToySupineEnv へ渡す。
    #   値は辞書（run/plugins/common/object_files.py の `plugins.object_files` に
    #   書いていた設定と同じ形）。未設定（None）なら scene_io.build() 側で何も作らない
    #   ＝この設定を書かない既存の実験は1ビットも変わらない。
    visual_attention = taro.get("visual_attention")
    if visual_attention is not None and not vision_on:
        raise ValueError(
            "taro.visual_attention には taro.vision=True が要る。\n"
            "  VisualAttention は環境のカメラ描画(get_vision_obs)の画像(eye_left)を\n"
            "  入力にしており、vision=False は環境の視覚センサ自体を無効化する\n"
            "  アブレーションになるため（taro.orienting_reflexと同じ理由）。")

    # ---- 視覚（2026-08-13、目標EでLeanMimoEnv.strip_texturesを発動させる修正）------
    #   【なぜ】taro.vision=False でも、これまでは環境が常にフルテクスチャ(約977MB)を
    #   持ったままだった。既存の仕組み(strip_textures)を発動させるため、
    #   taro.vision をそのまま scene_io.build() の vision= へ渡す。
    #
    #   【注意・視線誘導反射との相性】OrientingReflex（視線誘導反射）は環境側の
    #   カメラ描画(get_vision_obs)で毎ステップ更新される。vision=False では
    #   self.vision が None になり get_vision_obs が一度も呼ばれないため、
    #   orienting_reflex=True と vision=False を同時に指定すると反射が
    #   静かに動かなくなる（エラーは出ない）。「エラーが出ずに動いた」を
    #   信じない、という方針（doc/検証の落とし穴チェックリスト.md 項17）に
    #   従い、この組み合わせは明示的に止める。
    if orient and not vision_on:
        raise ValueError(
            "taro.orienting_reflex=True には taro.vision=True が要る。\n"
            "  視線誘導反射(OrientingReflex)は環境側のカメラ描画(get_vision_obs)に\n"
            "  依存しており、vision=False は環境の視覚センサ自体を無効化する\n"
            "  アブレーションになるため（strip_textures適用時、self.visionがNoneになり\n"
            "  get_vision_obsが一度も呼ばれない）。")
    if not vision_on:
        print("注意[vision] OFF: 環境のテクスチャ(顔・服)を単色化しメモリを節約します"
              "（LeanMimoEnv.strip_textures）。run.type=view/edit でこの設定のまま"
              "開くと、絵が単色になります（物理・学習には影響しません）。", flush=True)

    # 【段A・2026-09-13】visual_attention は**環境へ渡さない**。持ち主は太郎
    #   （run/taro_setup.py: build_visual_attention）。ここでは上の妥当性検査
    #   （vision=True が要る）だけを行う。
    env, hands = scene_io.build(sc, orient=orient,
                               vor=bool(taro.get("vor", True)),
                               seed=seed, verbose=verbose, actuation_model=act,
                               vision=vision_on)

    # ---- 感覚運動の伝達遅延（候補5、2026-08-02）----------------------------
    #   【なぜここで配線するか】TE.ToySupineEnv(...) を実際に呼んでいるのは
    #   run/scene_tools/scene_io.py の build()（このファイルは「組み立てるのはここだけ」と
    #   自分の docstring に書いているが、それは「scene_io.build() を呼ぶ場所を1つに
    #   絞る」という意味で、TE.ToySupineEnv(...) 自体の呼び出し箇所ではない）。
    #   scene_io.py は今回の実装範囲外（触ってよいファイルの一覧に無い）なので、
    #   コンストラクタに直接 sensory_delay=/motor_delay= を渡すのではなく、
    #   組み立て終わった env に対して同じ属性を書き換える形で配線する。
    #
    #   これで元の実装と等価かどうか：MIMoEnv.__init__（mimo_env.py 299-339行）は
    #   sensory_delay/motor_delay を**属性に代入するだけ**で、他の初期化処理は
    #   この値に依存しない（分岐は無い）。実際に使われるのは
    #   _delayed_action/_delayed_observation（同ファイル593-813行）で、これらは
    #   env.step() が呼ばれる**たびに** self.sensory_delay/self.motor_delay を読む。
    #   ⇒ コンストラクタ引数で渡しても、構築直後に属性を書き換えても、
    #     以後の env.step() から見える挙動は同一。
    #
    #   既定 0ms のときは何もしない（既存の属性 0 のままなので1行も実行されず、
    #   コード的に「触っていない」のと同じ状態になる）。
    sensory_ms = float(taro.get("sensory_delay_ms", 0) or 0)
    motor_ms = float(taro.get("motor_delay_ms", 0) or 0)
    if sensory_ms > 0 or motor_ms > 0:
        u = env.unwrapped
        dt = float(u.model.opt.timestep) * float(u.frame_skip)
        sensory_steps = int(round((sensory_ms / 1000.0) / dt)) if sensory_ms > 0 else 0
        motor_steps = int(round((motor_ms / 1000.0) / dt)) if motor_ms > 0 else 0
        if sensory_steps > 0:
            u.sensory_delay = sensory_steps
            u._obs_history = []
        if motor_steps > 0:
            u.motor_delay = motor_steps
            u._action_history = []
        print(f"注意[delay] 感覚運動遅延を配線: sensory_delay_ms={sensory_ms}"
              f"({sensory_steps}step, dt={dt*1000:.1f}ms) "
              f"motor_delay_ms={motor_ms}({motor_steps}step)。"
              f"[Tier2/Tier3、根拠は run/config.py のコメントを参照]", flush=True)

    # ---- 筋肉の活性化の時定数（震え問題への感度確認、2026-08-07）------------
    #   【なぜここで配線するか】上の感覚運動遅延と同じ理由。MuscleModel.tau
    #   （MIMo/mimoActuation/muscle.py 53行目）は self.tau = 0.01 とハードコード
    #   されており、実測較正されたものではないと既に注記されている。
    #   MuscleModel.__init__ はコンストラクタ引数で tau を受け取らないため、
    #   scene_io.py（触ってよいファイルの一覧に無い）を変更せずに済むよう、
    #   組み立て終わった env の actuation_model に対して属性を直接書き換える形で配線する。
    #
    #   _update_activity（muscle.py 291行）は env.step() が呼ばれるたびに
    #   self.tau を読むので、コンストラクタ引数で渡しても構築直後に属性を
    #   書き換えても、以後の env.step() から見える挙動は同一。
    #
    #   既定 None のときは何もしない（既存の属性 0.01 のままなので1行も実行されず、
    #   コード的に「触っていない」のと同じ状態になる）。
    #   関節モード（SpringDamperModel）には tau 属性が無いので hasattr で確認して
    #   から代入する（無ければ何もしない）。
    muscle_tau = taro.get("muscle_tau", None)
    if muscle_tau is not None:
        u = env.unwrapped
        if hasattr(u.actuation_model, "tau"):
            old_tau = u.actuation_model.tau
            u.actuation_model.tau = float(muscle_tau)
            print(f"注意[muscle_tau] 筋活性化の時定数を配線: tau={old_tau}"
                  f"→{float(muscle_tau)}秒。既定0.01秒は実測較正されていない値"
                  f"[Tier3、根拠は run/config.py のコメントを参照]", flush=True)
        else:
            print(f"注意[muscle_tau] taro.muscle_tau={muscle_tau} が指定されましたが、"
                  f"actuation_model に tau 属性がありません（関節モード等）。無視します。",
                  flush=True)

    if hybrid:
        # 注意：内臓を足す（内受容感覚が観測に入る）。学習ではこれが要る。
        from hybrid_env import HybridEnv
        env = HybridEnv(env)
    return env, sc, hands
