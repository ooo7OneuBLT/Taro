"""関節可動域の限界へ近づく抵抗を滑らかにする — MuJoCoのjnt_solimpを実行時に書き換える。

【なぜ作ったか、2026-08-13】太郎の全関節は、MIMoのXML（`MIMo_modelv2.xml`）に
jointlimit用のsolimp指定が一切無く、MuJoCoの組み込み既定値
`solimp = (0.9, 0.95, 0.001, 0.5, 2)` をそのまま使っている。width=0.001ラジアン
（約0.057度）しか「軟化ゾーン」が無く、限界の直前まで完全に無抵抗で、
そこから一気に強い抵抗が立ち上がる「ほぼ垂直な壁」になっている。この壁に
右肩が183度で547tick張り付く事象が観測され、「最も長く自己接触した優秀な
ラン」として指標上誤評価された（実際は可動域の限界に固着していただけ）。

人間の関節（靭帯・関節包）は限界のかなり手前から徐々に抵抗が強まる。これを
模して、可動域の端に近づくにつれ滑らかに抵抗が強まるように `model.jnt_solimp`
を実行時（Python側）から上書きする。MuJoCoのXML本体は書き換えない（他の
`infant_body.py`・`infant_limbs.py`の補正と同じ方針＝Git管理外のMIMoを触ると
再現性が失われるため）。

【重要・2026-08-13の検証（check_joint_compliance.py）で判明した2点】
実装は仕様・統合版設計・案B作業記録の数式のとおりだが、実機で検証したところ
設計時の物理モデルの想定と食い違う点が2つ見つかった。実装担当への報告で
「想定外」として上げてあるので、詳細はそちらを参照。ここにも要点を残す。

  1. MIMoのhinge関節93本は、実測すると **raw MuJoCoの組み込み既定
     (0.9, 0.95, 0.001, 0.5, 2) ではなく、全hinge関節が共通して
     (0.98, 0.99, 0.001, 0.5, 1.0) を持っていた**（free/ball関節3本のみ
     raw既定のまま）。統合版設計・案B作業記録は「XMLにsolimp指定が一切無い」
     としていたが、これはMuJoCoの`<default class="...">`機構による
     クラス既定が、raw既定とは別に存在することを見落としていたと考えられる。
     width=0.001ラジアン（約0.057度、垂直な壁の正体）自体は一致しており、
     「壁が急峻」という主結論は変わらない。d0(0.9→0.98)・d_width(0.95→0.99)・
     power(2→1)の差は本実装のTier3数値そのもの（仕様5節指定の
     width_frac=0.08・d0=0.5・d_width=0.95・midpoint=0.8・power=2）を
     変更する理由にはならないと判断し、仕様どおりの数値のまま実装した。

  2. **`model.jnt_margin` が全関節で0.0だった。** MuJoCoの関節可動域制約は
     `pos = margin - distance`が正になった時点（＝実際に可動域を超えた
     瞬間）で初めて有効になり、solimpの`width`はその**有効化した後**の
     侵入深さに対する滑らかさを決める（margin自体が「限界の手前、
     どれだけの距離から検知を始めるか」を決める別のパラメータ）。
     margin=0のままだと、本実装（jnt_solimpのみを書き換える）は
     「限界に到達する前」には一切効かず、可動域を実際に超えた瞬間の
     衝突そのものは既定と同じ急激さで起きる（実測：衝突tickの角速度は
     ON/OFFでほぼ同一）。差が出るのは**衝突後、どれだけ深く侵入し続けるか**
     （d_width=0.95がbaselineの0.99より弱いため、ONの方がむしろ侵入が
     深くなる）という、設計が意図した「手前からの滑らかな減速」とは
     異なる効果になっている。この機能で「限界のかなり手前から徐々に
     抵抗が強まる」を物理的に実現するには、`jnt_margin`もあわせて
     引き上げる必要があると考えられるが、仕様5節はsolimpの5数値のみを
     指定しており、jnt_marginには触れていない。今回はスコープ外と判断し、
     仕様どおりjnt_solimpのみを書き換える実装のまま留めた（実装担当判断・
     ユーザーへの報告で上げてある）。

【追記・2026-08-13、marginを追加】
上記2点目の指摘を受け、続きの仕様（作業記録（非公開）
2026-08-13_関節可動域の壁を滑らかにする実装_margin追加.md`）で
`model.jnt_margin`も設定できるように拡張した（`margin_frac`、
`_margin_from_params()`）。`run/tools/check_joint_compliance.py`の
`verify_f_margin_sweep`で、右肩（実測した可動域幅267.0度）に対し margin_frac を
0（margin無し=前回のバグ状態）・約0.37%（1度相当）・約1.9%（5度相当）・
約3.7%（10度相当）・8%（widthと同じ、約21.4度相当）・16%（widthの2倍、
約42.7度相当）で振り、一定トルク(2.0N・m)で押し当て続けたときの角速度
（限界までの残り角度30/20/10/5/2/1/0.057度の各時点）・限界±1度以内に留まった
連続tick数・400tick終了時点の最終角度（183度からの超過量）を実測した
（実行ログ・作業記録作業記録（非公開）
2026-08-13_関節可動域の壁を滑らかにする実装_margin追加.md`参照）。

実測結果の要旨：
  margin無し(0度)・1度相当は、188.6度前後（超過4.6〜5.6度）に静止し、
  衝突tick直前まで速度がほぼ一定（単調な減速になっていない）。
  5度相当（margin_frac≈0.0187）では超過が0.96度まで縮み、10度相当の残り角度から
  減速が始まって最も改善が大きかった。10度相当以上（margin_frac≈0.0375以上）は
  400tick以内に限界へ到達すらせず途中で止まる（10度相当は178.95度、
  widthと同じ8%は167.44度、widthの2倍の16%は147.05度で停止）。
  つまり margin を大きくしすぎると、一定トルクを与え続けても本来の可動域の
  終端（183度）に到達しなくなる＝可動域が実用上狭くなりすぎる、という
  トレードオフが実測で確認できた。

既定値は margin_frac=0.02（この関節ではおよそ5.3度相当）を採用した。
[Tier3・工学的判断] 根拠：実測した候補の中で「限界に近づく速度の減速が
明確に見られる」かつ「本来の可動域の終端付近まで到達できる」を両立していたのが
5度相当の候補だったため、それに最も近い切りの良い値として0.02を選んだ。
width_frac=0.08と同じ値にはしなかった（実測でwidthと同じ割合のmarginは
強すぎ、本来の可動域を大きく狭めてしまうことが分かったため）。

「188.6度で新たに静止する現象」は、margin_frac=0.02〜0.0375相当（5〜10度）の
範囲ではほぼ消える（超過0.96度、あるいは限界に到達すらしない）ものの、
1度相当以下では前回とほぼ同じ角度に静止し**解消していない**。したがって
「marginを足せば必ず解消する」という単純な話ではなく、**値の選び方次第**という
のが正直な結論。詳細・生の実測表は作業記録を参照。

【根拠レベル（Tier1〜3）】
    Tier1（事実そのもの）
      MuJoCoの既定solimpがwidth=0.001ラジアン（約0.057度）しか軟化ゾーンを
      持たないという事実。これは人間の話ではなく、太郎が使うシミュレータ
      自身の設定に関する事実（MuJoCo公式ドキュメントの数式・調査確認済み）。
    Tier2（間接的な支持・定性的な移植）
      「関節可動域の限界に近づくほど抵抗が滑らかに強まる」という関節包・靭帯の
      受動的な性質そのものは、Silder, Whittington, Heiderscheit & Thelen (2007)
      J Biomechanics 40(12):2628-2635（健常成人20名、股関節・膝関節・足関節の
      他動運動から測定した受動トルクM=e^(β(θ−α))）から人間に存在すると言える。
      ただし対象部位（肩）・対象年齢（乳児）には直接の文献が無いため、
      「肩の乳児にもこの性質が定性的にあるはず」という一般化そのものは
      部位と年齢をまたいだ外挿。DEFAULT_JOINT_COMPLIANCE のmidpointを0.5より
      限界寄りに設定することで、Silderのカーブの"形"（限界のごく近くだけ
      急激に強まる凸型）を質的に模倣している。
    Tier3（工学的な近似・恣意的な数値決定）
      width_frac・d0・midpointという具体的な数値そのもの。MuJoCoのsolimpが
      表す量（無次元の拘束インピーダンス、0〜1）と、Silderの式が表す量
      （実際のトルク、N・m）は単位・定義域が異なり、数値そのものを移植する
      ことはできない（3点の理由：(1)肩ではない (2)成人であり乳児ではない
      (3)単位・定義域が異なりMuJoCoのsolimpへ数値を直接移植できない）。
      借りているのは「限界近くで急激に強まる凸型カーブ」という定性的な形のみ。
      詳細は `doc/人間模倣からの逸脱リスト.md` 項㉑。

【将来の拡張候補（判断1、ユーザー＝実装担当からの指示、2026-08-13）】
人間の受動トルク（Silderの指数関数モデル）をmjcb_passiveコールバックで独自に
計算する案（設計・案C）も検討された。数式的にはより忠実だが、未解決の
技術的論点（座標変換の未較正・トルクのクリップ値未決定・配線箇所未特定等）が
多く実装リスクが高いため、今回はsolimpベース（本ファイル）を採用した。将来
これに乗り換えられるよう、`_solimp_from_params()`（関節ごとのパラメータ辞書＋
現在のjnt_rangeから、solimpの5つの数値を計算する処理）を、関節の選択・
model配列への書き込みロジック（`apply_joint_compliance()`本体）とは別の1つの
関数に分離してある。この関数を差し替えれば、solimpベース以外の計算方式
（例えばmjcb_passiveを使う案C方式）にも展開しやすくなる、という程度の分離で
あり、過剰な抽象化（プラグイン機構・コールバック登録の仕組み）はしていない。

【中立姿勢への復元力（apply_limb_tone）との関係（判断3、2026-08-13）】
既存の`infant_limbs.apply_limb_tone`（jnt_stiffness・qpos_springで「今の姿勢/
目標姿勢へ戻る弱いバネ」を作る機構）とは別の仕組みのまま実装した。
Maekawa & Ochiai (1975, Dev Med Child Neurol 17(4):440-446、抄録で確認、
Thelen, Fisher & Ridley-Johnson 1984 で独立に裏付け)によれば、新生児の
屈曲姿勢は少なくとも生後まもない時期には能動的な筋力ではなく受動的な拘縮
（passive contracture）による可能性がある。これは「可動域の端の抵抗」
（本ファイル）と「中立姿勢への復元力」（apply_limb_tone）が、人間の生理では
同じ現象（受動的な組織の拘縮）に由来し、将来この2機構が統合されうることを
示唆する。今回は新生児の生理的屈曲位の具体的な角度が未調査のため統合を
見送り、分離したまま実装した。この将来の統合に備え、関節選択ロジック
（groups/joints解決部分）は `infant_limbs.py` の `_expand_groups` /
`limb_tone_joints_by_group` と同じ考え方（グループ名→関節名の展開、
`LIMB_TONE_ALIASES` の再利用）に揃えてある。

【依存関係の向き】このファイルは `infant_limbs.py` の `LIMB_TONE_GROUPS` /
`LIMB_TONE_ALIASES`（関節のグループ分けの表）だけを読み取り専用で参照する。
`apply_limb_tone` 等の関数は呼ばない。逆方向（`infant_limbs.py` がこのファイルを
import する）は作らない。

【使い方】
    from joint_compliance import apply_joint_compliance
    apply_joint_compliance(model, groups=["shoulder"])          # 左右の肩だけ
    apply_joint_compliance(model, groups=["all"])                # 限界を持つ全関節
    apply_joint_compliance(model, joints=["right_shoulder_ad_ab"])  # 個別指定
"""

# 既定パラメータ。DEFAULT_JOINT_COMPLIANCE と JOINT_COMPLIANCE_OVERRIDES から
# 「関節ごとの最終的な値」を組み立てる（_resolve_params）。
#
# 太郎の全関節の既定値（MuJoCo組み込み既定）は (0.9, 0.95, 0.001, 0.5, 2)。
# width=0.001ラジアン（約0.057度）が壁の正体であり、d0=0.9も「軟化ゾーンに
# 入った瞬間から既に9割の強さ」という意味で、そもそも軟化ゾーン自体が
# 実質的に存在しない設定になっている。
DEFAULT_JOINT_COMPLIANCE = dict(
    width_frac=0.08,   # [Tier3] 可動域全体の8%を軟化ゾーンにする
    d0=0.5,            # [Tier3] 軟化ゾーンに入った直後は「まだ弱い」
    d_width=0.95,      # MuJoCo既定のまま（限界そのものでの強さは変えない）
    midpoint=0.8,      # [Tier3] 変化の中心を限界寄りに置く（Silderのカーブの
                       #   "形"＝限界近くで急激に強まる凸型を質的に模す）
    power=2,           # MuJoCo既定のまま（変える根拠も変えない根拠も無い）
    # [Tier3・2026-08-13追加] margin=0のままだとjnt_solimpは「限界に到達する
    #   前」には一切効かない（モジュールdocstring【追記・2026-08-13】節参照）。
    #   width_fracと同じ考え方＝可動域全体に対する割合でmarginを決める。
    #   width_fracと同じ0.08にはしていない——実測（verify_f_margin_sweep）で
    #   0.08はこの関節では強すぎ、一定トルクを与え続けても限界(183度)の16度
    #   手前(167.44度)で止まってしまい、可動域を大きく狭めてしまうことが
    #   分かった。0.02（この関節では約5度相当）は、限界に近づく速度の
    #   減速が明確に見られつつ、本来の可動域の終端付近まで到達できる
    #   バランス点として選んだ（モジュールdocstring【追記・2026-08-13】節、
    #   作業記録2026-08-13_関節可動域の壁を滑らかにする実装_margin追加.md参照）。
    margin_frac=0.02,
)

# 関節ごとの上書き（初期状態は空）。将来、他関節の文献値・調整値が
# 見つかったら {"関節のbase名": {"width_frac": ...}} の形で1行足す。
# キーは "right_"/"left_" を除いた base 名（LIMB_TONE_GROUPS と同じ命名）。
JOINT_COMPLIANCE_OVERRIDES = {}

# パラメータとして受け付けるキー（フラットなparams辞書か、関節名キー付き
# 辞書かを見分けるのにも使う。_is_flat_params 参照）。
_PARAM_KEYS = frozenset(DEFAULT_JOINT_COMPLIANCE.keys())


def _solimp_from_params(range_lo, range_hi, params):
    """関節ごとのパラメータ辞書＋現在のjnt_rangeから、solimpの5つの数値を計算する。

    【なぜ apply_joint_compliance() 本体と分離してあるか（判断1、2026-08-13）】
    将来「人間の受動トルクモデル（案C、Silderの指数関数）」など、solimp以外の
    計算方式へ移行する可能性を見越し、「パラメータ→solimpの5数値」という計算
    そのものを、関節の選択・model配列への書き込みロジック（呼び出し側）から
    切り離してある。この関数を差し替えるだけで、計算方式を入れ替えられる、
    という程度の分離であり、プラグイン機構やコールバック登録のような
    過剰な抽象化はしていない。

    Args:
        range_lo, range_hi: その関節の可動域 [ラジアン]（model.jnt_range の行）。
        params: DEFAULT_JOINT_COMPLIANCE と同じキーを持つ辞書
            （width_frac, d0, d_width, midpoint, power）。

    Returns:
        (d0, d_width, width, midpoint, power) の5要素タプル。MuJoCoの
        `model.jnt_solimp[jid, :]` にそのまま代入できる並び順。
    """
    width_frac = float(params["width_frac"])
    d0 = float(params["d0"])
    d_width = float(params["d_width"])
    midpoint = float(params["midpoint"])
    power = float(params["power"])

    range_span = float(range_hi) - float(range_lo)
    # 可動域に対する割合で軟化ゾーンの幅を決める（絶対角度ではない、design参照）。
    # 【なぜ、2026-08-13】可動域が狭い関節（生理的屈曲で狭められた肘等）でも、
    #   同じ割合を使えば軟化ゾーンの「効き方の感覚」を揃えられる。
    # width=0はMuJoCo側で無効値になりうるため、下限を設ける。
    width = max(width_frac * range_span, 1e-6)
    return (d0, d_width, width, midpoint, power)


def _margin_from_params(range_lo, range_hi, params):
    """関節ごとのパラメータ辞書＋現在のjnt_rangeから、margin（ラジアン）を計算する。

    【なぜ _solimp_from_params とは別関数か（2026-08-13、margin追加時の判断）】
    _solimp_from_params は既存の呼び出し元（run/tools/check_joint_compliance.pyの
    検証0）が5要素タプルの戻り値をそのまま検算しているため、戻り値の形を変える
    と既存の検証が壊れる。margin は model.jnt_margin という別の配列に書く、
    solimpとは独立した量なので、計算そのものを別関数に分離した。

    width_frac と同じ考え方（可動域全体に対する割合）で計算する。marginが
    widthより小さいと、物理的な限界に到達する時点でもまだ最大強度(d_width)に
    達しておらず、限界を超えて侵入してから初めて最大強度に達する
    （MuJoCo公式仕様、調査報告2026-08-13_関節可動域の壁を滑らかにする仕組みの
    調査.md 1-1節）。

    Args:
        range_lo, range_hi: その関節の可動域 [ラジアン]（model.jnt_range の行）。
        params: DEFAULT_JOINT_COMPLIANCE と同じキーを持つ辞書（margin_fracを含む）。

    Returns:
        float: margin [ラジアン]（0以上）。
    """
    margin_frac = float(params["margin_frac"])
    range_span = float(range_hi) - float(range_lo)
    return max(margin_frac * range_span, 0.0)


def _is_flat_params(params):
    """params が「共通上書き（フラットな辞書）」か「関節名キー付き辞書」かを判定する。

    値の型で判定する（キー名で判定しない）：フラット辞書の値は数値
    （width_frac・d0等）、関節名キー付き辞書の値はさらに辞書（その関節専用の
    上書き辞書）。キー名で判定すると、フラット辞書に打ち間違いのキー
    （例："d_min"）が来たとき「未知のパラメータキー」ではなく「未知の関節名」
    と誤解釈され、_validate_param_keys が一度も呼ばれず**黙って無視される**
    （2026-08-13、check_joint_compliance.py の検証で実際に踏んだ）。
    """
    if not params:
        return False
    return not any(isinstance(v, dict) for v in params.values())


def _validate_param_keys(params):
    """params の各キーが既知のパラメータ名かを確認する。未知なら例外で止める。

    黙って無視せず明示的に例外にする（前提.md・検証の落とし穴チェックリスト項111
    と同じ考え方＝「通ってはいけない条件」で実際に止まることを保証する）。
    """
    unknown = [k for k in params.keys() if k not in _PARAM_KEYS]
    if unknown:
        raise ValueError(
            f"apply_joint_compliance: paramsに未知のキーがあります: {unknown}"
            f"（有効なキー: {sorted(_PARAM_KEYS)}）")


def _resolve_params(base_name, params):
    """ある関節（base名）に最終的に使うパラメータ辞書を組み立てる。

    優先順位（後のものが前を上書きする）：
      1. DEFAULT_JOINT_COMPLIANCE
      2. JOINT_COMPLIANCE_OVERRIDES[base_name]（今は空。将来、関節ごとの
         文献値が見つかったらここに乗る）
      3. apply_joint_compliance() に渡された params
         （フラット辞書＝全対象共通の上書き、または関節名キー付き辞書＝
         その関節だけへの個別上書き。_is_flat_params で判定）
    """
    out = dict(DEFAULT_JOINT_COMPLIANCE)
    out.update(JOINT_COMPLIANCE_OVERRIDES.get(base_name, {}))

    if params:
        if _is_flat_params(params):
            _validate_param_keys(params)
            out.update(params)
        else:
            per_joint = params.get(base_name)
            if per_joint:
                _validate_param_keys(per_joint)
                out.update(per_joint)
    return out


def _resolve_target_joints(model, groups, joints):
    """groups・joints から、対象にする関節の一覧 [(jid, full_name, base_name), ...] を返す。

    関節選択のロジックは `infant_limbs.py` の `_expand_groups` /
    `limb_tone_joints_by_group` と同じ考え方（グループ名→関節名の展開、
    LIMB_TONE_ALIASES の再利用）に揃えてある（判断3、将来の統合に備える）。

    groups の特別な値 "all"：モデル中の全ての「限界を持つ関節
    （model.jnt_limited[j]==1 のhinge関節）」を対象にする。既存の
    LIMB_TONE_GROUPS のグループ分けに依存しないので、既存の部位分けに
    無い関節（首・体幹等）も対象になりうる（判断2、実装担当の判断）。

    groups=None かつ joints=None は不正な状態として明示的に ValueError で
    止める（黙って何もしない、を避ける。落とし穴チェックリスト項111）。
    """
    if not groups and not joints:
        raise ValueError(
            "apply_joint_compliance: groups と joints のどちらも指定されていません。"
            " 意図しない『対象が0件のまま黙って何もしない』状態を避けるため、"
            "明示的な例外にしています。全関節を対象にしたい場合は"
            " groups=['all'] を明示してください。"
        )

    import mujoco as _mj
    from infant_limbs import LIMB_TONE_GROUPS, LIMB_TONE_ALIASES

    groups = list(groups or [])
    joints = list(joints or [])

    want_all = "all" in groups
    named_groups = [g for g in groups if g != "all"]

    # "arm"/"leg" のような別名を細かい部位へ展開する（infant_limbs._expand_groups
    # と同じ考え方）。
    expanded = []
    for g in named_groups:
        expanded.extend(LIMB_TONE_ALIASES.get(g, (g,)))
    expanded = list(dict.fromkeys(expanded))

    unknown_groups = [g for g in expanded if g not in LIMB_TONE_GROUPS]
    if unknown_groups:
        raise ValueError(
            f"apply_joint_compliance: 未知のgroup名です: {unknown_groups}"
            f"（有効なgroup名: {sorted(LIMB_TONE_GROUPS)} + 'all'）")

    want_bases = set()
    for g in expanded:
        want_bases.update(LIMB_TONE_GROUPS[g])

    out = []
    seen_ids = set()

    def _base_name(short):
        base = short
        for pre in ("right_", "left_"):
            if base.startswith(pre):
                return base[len(pre):]
        return base

    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(_mj.mjtJoint.mjJNT_HINGE):
            continue
        full = (model.joint(j).name or "").split(":")[-1]
        base = _base_name(full)
        matched = False
        if want_all:
            if int(model.jnt_limited[j]) == 1:
                matched = True
        elif base in want_bases:
            matched = True
        if matched and j not in seen_ids:
            out.append((j, full, base))
            seen_ids.add(j)

    # joints（個別指定）を追加。存在しなければ黙って無視せず例外で止める。
    for name in joints:
        short = name.split(":")[-1]
        jid = None
        for j in range(model.njnt):
            full = (model.joint(j).name or "").split(":")[-1]
            if full == short:
                jid = j
                break
        if jid is None:
            raise ValueError(f"apply_joint_compliance: 関節が見つかりません: {name}")
        if jid not in seen_ids:
            out.append((jid, short, _base_name(short)))
            seen_ids.add(jid)

    return out


def apply_joint_compliance(model, groups=None, joints=None, params=None, verbose=True):
    """関節可動域の限界へ近づく抵抗を滑らかにする（既定OFF・呼び出し側でON/OFFを制御）。

    `model.jnt_solimp` を実行時に書き換える。対象関節の `model.jnt_limited` も
    1へ設定する（solimpは jnt_limited=1 でなければ効かないため）。

    Args:
        model: MuJoCo の MjModel。
        groups: 対象の部位グループ名のリスト。
            "all"          限界を持つ全ての hinge 関節（infant_limbsのグループ
                           分けに依存しない、判断2）
            "shoulder" 等   infant_limbs.LIMB_TONE_GROUPS のグループ名
            "arm"/"leg" 等  infant_limbs.LIMB_TONE_ALIASES の別名（自動展開）
        joints: 個別の関節名（":" より後ろ、例 "right_shoulder_ad_ab"）のリスト。
            groups と併用可（和集合として扱う）。
        params: パラメータの上書き。
            フラットな辞書（キーが width_frac/d0/d_width/midpoint/power のいずれか
              のみ）なら、対象関節すべてへの共通上書き。
            関節のbase名をキーにした辞書（例 {"shoulder_ad_ab": {"d0": 0.3}}）なら、
              その関節だけへの個別上書き。
            None なら DEFAULT_JOINT_COMPLIANCE（＋JOINT_COMPLIANCE_OVERRIDES）
              をそのまま使う。
        verbose: 適用結果を1行表示するか。

    Returns:
        int: solimpを書き換えた関節の数。

    Raises:
        ValueError: groups・joints のどちらも未指定、未知のgroup名、
            存在しない関節名、未知のparamsキー、のいずれか。
    """
    targets = _resolve_target_joints(model, groups, joints)

    n = 0
    applied = []
    for jid, full_name, base_name in targets:
        lo = float(model.jnt_range[jid, 0])
        hi = float(model.jnt_range[jid, 1])
        if not (hi > lo):
            # 可動域が定義されていない・壊れている関節は安全側でスキップする
            # （このタスクの対象関節では起きない想定だが、groups=["all"]で
            #   将来対象が広がったときの防御）。
            continue
        p = _resolve_params(base_name, params)
        solimp = _solimp_from_params(lo, hi, p)
        margin = _margin_from_params(lo, hi, p)
        model.jnt_solimp[jid, :] = solimp
        model.jnt_margin[jid] = margin
        model.jnt_limited[jid] = 1
        n += 1
        applied.append(f"{full_name}(width={solimp[2]:.4f}rad={_rad2deg(solimp[2]):.2f}deg,"
                       f"margin={margin:.4f}rad={_rad2deg(margin):.2f}deg)")

    if verbose and n:
        print(f"[joint_compliance] {n}関節にsolimpを適用: " + " ".join(applied) +
              " [Tier3: 定性的な形のみSilder et al. 2007から借用、数値は工学的近似。"
              "詳細は doc/人間模倣からの逸脱リスト.md 項㉑]")
    return n


def _rad2deg(x):
    import math
    return math.degrees(x)
