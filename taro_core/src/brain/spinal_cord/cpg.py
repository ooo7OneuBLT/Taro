"""脊髄・中枢パターン発生器（CPG）相当のモジュール。

【解剖学的位置づけ、2026-07-23】新生児のwrithing GM（運動性喃語）は、皮質からの精密制御を
ほぼ受けず、脳幹・脊髄の自律的なリズム生成回路（central pattern generator, CPG）が主体と
考えられている（Hadders-Algra 2018ほか）。太郎のB-min（`d_c5_motor_quality.py`のE_WMEAN=0
経路）はこれに対応し、以下2つの本能を実装する：
  ①色付きノイズ（1/f^β）＝López et al. 2026 の実測βに基づく（旧colored_noise.py、ここに統合）
  ②粗いシナジー（脚・腕）＝Dominici 2011（脚, Tier1）・Physiopedia他（腕, Tier2）
運動野（motor_head/pc_latent/motor_gru、taro_brain_motor.py）や皮質脊髄路（w_mean混合、
`corticospinal.py`）とは別の、より下位の自律回路として位置づける。
"""
import os as _os
import numpy as np


class ColoredNoiseGenerator:
    """1/f^β ノイズを次元ごとに独立生成する。sample(beta) を毎tick呼ぶと (n_dim,) を返す。

    注意：【2026-07-26・位置づけの明示】これは**人間の自発運動の発生機構の実装ではない**
    [Tier不能・人間側が未特定]。
        人間（文献の記述） ＝ 神経細胞の確率的バースト発火
                            → 脊髄・脳幹の回路で統合
                            → 身体力学と感覚を通じて秩序が生まれる（自己組織化）
        ここの実装         ＝ **関節トルクに直接ノイズを流す（中間の段階がない）**
    ＝逸脱の本体は「ノイズか振動子か」ではなく **ノイズを入れる場所**。

    それでもこの方式を使う理由（3つ）：
      ① 実測と矛盾しない。Kanazawa & Kuniyoshi (PNAS 2022) は新生児の感覚運動状態の遷移が
         **統計的にランダムと有意差なし（P=0.6776）**と報告（構造が出るのは発達後 P=0.0210）。
      ② 代替も近似。國吉研の BVP 振動子も生理学的実測から出た値ではない。
      ③ 元論文（López et al. 2026）の著者自身が「RL探索のための工学的手法」と位置づけており、
         **生成機構の主張はしていない**。太郎はその位置づけのまま使っている。
    注意：「人間の機構を実装している」とは書かないこと（シナジーで実際にやった誤り＝項42）。
      書いてよいのは「**人間の出力統計（β）を借りた近似**」まで。
    注意：人間の機構そのものが未特定。Hadders-Algra 2018 (DMCN 60:39-46) 自身が
      複雑性・変動性の由来を "assumed"（仮定される）と明記している。
    → doc/人間模倣からの逸脱リスト.md「2026-07-26：自発運動の生成方式そのものの位置づけ」

    【使い方】次元ごとに独立な過程をブロック単位（既定2000tick分）で作り、尽きたら次の
    ブロックをその時点のβで作り直す＝βを毎tick変えても（発達に応じたスケジュールでも）動く。
    Timmer & König (1995) のスペクトル整形法：周波数領域で振幅を f^(-β/2) に整形したガウス
    雑音を作り、逆FFTで時間領域に戻す。βが大きいほど低周波成分が強い＝滑らかで持続的な動き。
    """

    def __init__(self, n_dim, block_len=2000, seed=None):
        self.n_dim = n_dim
        self.block_len = block_len
        self.rng = np.random.default_rng(seed)
        self._buf = None
        self._pos = 0

    def _make_block(self, beta):
        L = self.block_len
        freqs = np.fft.rfftfreq(L).copy()
        freqs[0] = freqs[1] if L > 1 else 1.0  # DC成分の0除算を避ける
        amp = freqs ** (-beta / 2.0)
        block = np.empty((L, self.n_dim), dtype=np.float32)
        for d in range(self.n_dim):
            re = self.rng.normal(size=amp.shape) * amp
            im = self.rng.normal(size=amp.shape) * amp
            im[0] = 0.0
            if L % 2 == 0:
                im[-1] = 0.0
            x = np.fft.irfft(re + 1j * im, n=L)
            x = x - x.mean()
            x = x / (x.std() + 1e-9)
            block[:, d] = x
        self._buf = block
        self._pos = 0

    def sample(self, beta):
        """1/f^βノイズを1tickぶん取り出して返す。引数betaを受け取り、内部バッファが未生成か使い切っていれば、その時点のbetaで新しいブロックを作り直してからshape=(n_dim,)の1点を返す。
        """
        if self._buf is None or self._pos >= self.block_len:
            self._make_block(beta)
        v = self._buf[self._pos]
        self._pos += 1
        return v


class CPG:
    """複数関節をまとめて駆動するCPG本体。独立ノイズ＋粗い相関（脚・腕）を合成して
    n_act次元の babbling 出力を1回のsample()で返す（呼び出し側は低頻度化キャッシュだけ
    考えればよい構成、2026-07-23の整理でd_c5_motor_quality.pyから抽出）。

    注意：【2026-07-25 名前についての重要な訂正】引数名は歴史的経緯で `synergy` だが、
    **これは人間の筋シナジーの実装ではない**。
        人間のシナジー ＝ 脊髄回路が**運動出力そのもの**を制約する（Tier1）
        ここの実装     ＝ **探索ノイズを相関させるだけ**
                          学習後の方策は全次元を自由に出せる＝決定的な行動には一切効かない
    ＝「人間にある機構を実装済み」と誤認して寝返り問題の対策に使おうとし、
      効かない理由を別の場所に探し続けた（→ 検証の落とし穴チェックリスト 項42）。
    正しい実装は「方策の出力を低次元のシナジー空間に通す」（工学の先例 SAR の形）で、
      これは未実装。やることリストの課題。
    注意：既定は OFF。理由は「効かなかったから」ではなく**人間の機構を再現していないから**
      （「効かないからOFF」は項38が禁じている工学的判断）。

    synergy=False（既定）なら各関節は完全独立＝従来のColoredNoiseGeneratorと1バイト差なし。

    【pair_offset・2026-07-25】筋肉モード（MuscleModel）対応。
    MuscleModel の行動は 2*n_joint 次元で、**前半が負方向筋（曲げる側）・後半が正方向筋
    （伸ばす側）**。leg_r 等のindexは関節番号（0..n_joint-1）なので、そのままでは
    「曲げる側の筋」にしかシナジーが掛からない。pair_offset=n_joint を渡すと、
    対になる筋 i+offset にも **符号を反転して** 同じシナジーを混ぜる。

    符号を反転するのが要点。同符号で入れると「曲げる筋と伸ばす筋を同時に強める」＝
      **共収縮（関節が固まる）**になり、「脚がまとまって曲がる／伸びる」にならない。
      新生児の kicking は股・膝・足首がまとまって屈曲し、まとまって伸展する
      （＝ユーザーが自発運動の動画で観察した「伸ばす・縮めるの繰り返し」）。

    注意：これを入れるまで、筋肉モードでは呼び出し側がシナジーを**強制OFF**にしていた
      （e_growth_train.py の `_use_syn = _E_SYNERGY and not _MUSCLE`）。
      ＝文献が一致して言う新生児の特徴「まとめてしか動かせない」が、
      学習に使う設定では一度も効いていなかった。

    【2026-08-11・符号バグの修正】上の「曲げる側」というラベルは、行動配列の前半
      （index i、pair_offset実装上の"neg方向筋"）を指すだけの**配列上の呼び名**であり、
      実際にその筋がMuJoCoの関節のどちら向き（生の qpos の +/-）を屈曲とするかとは
      無関係だった。ところがMuJoCoの関節は正方向の定義が関節ごとにバラバラ
      （実測：肩水平は `+`=屈曲・肘は `+`=伸展、股関節1・膝は `-`=屈曲・足首は `+`=屈曲）。
      旧実装は「index i を常に-1（曲げる方向）とみなして+s寄せ」ていたため、
      肘・股関節1・膝のように「生の+方向が屈曲でない」関節では、シナジーで作った
      「まとまった屈伸」が符号レベルで逆になり、肩が曲がると肘が伸びる＝
      関節同士が打ち消し合っていた（肩-肘の屈曲量相関 実測 −0.253±0.098）。

      修正：leg_r/leg_l/arm_r/arm_l の各要素に、`(index, sign)` のタプルで
      「生のqpos正方向が屈曲なら sign=+1、伸展なら sign=-1」を持たせられるようにした
      （後方互換のため、単なる int も引き続き受け付ける＝sign不明・無補正の旧来どおりの式）。
      sign を持つ要素は、`_blend()` 内で以下のように補正する：
        pair_offset==0（関節空間・符号つき直接指令モード）：
          gen_np[i] を sign*s に近づける
        pair_offset!=0（拮抗筋2チャンネル・活性化[0,1]モード。既定の使用モード）：
          gen_np[i]（"曲げる側"ラベルの筋）を -sign*s に、
          対になる筋 gen_np[i+pair_offset] を +sign*s に近づける
      （pair_offset!=0モードで符号が反転する理由：index iは配列上「常にneg方向筋
       （曲げる側ラベル）」だが、sign=+1の関節（生の+方向が屈曲）ではラベルと実際の
       屈曲方向が逆になるため、シナジー信号 s の符号を反転してからneg筋に与える必要が
       ある。sign=-1の関節ではラベルと実際が一致するので式は変わらない＝旧来どおり）。

      符号の根拠（実測。詳細は作業記録（非公開）
      2026-08-11_自発運動3日間横断監査.md`・作業記録（非公開）
      2026-08-11_CPGシナジー符号バグ修正.md`）：
        shoulder_horizontal（腕index14/43）  sign=+1（+方向=屈曲。動力学つき・複数シードで確定）
        elbow（腕index17/46）                sign=-1（+方向=伸展。静的mj_kinematicsスイープで単調・確定）
        hip1（脚index72/81）                 sign=-1（+方向=伸展。静的スイープで単調・確定）
        knee（脚index75/84）                 sign=-1（+方向=伸展。静的スイープで単調・確定）
        foot1/ankle（脚index76/85）          sign=+1（+方向=屈曲。静的スイープで単調・確定）
      次の関節は符号を推測せず、シナジーグループから**除外**した（詳細は仕様参照）：
        shoulder_ad_ab（腕index15/44）：静的スイープで非単調・確定した符号の記載なし
        hand2/wrist_flexion（腕index19/48）：neutral付近が最伸展という構造で
          単純な二値の符号判定が原理的に成り立たない
        hip2/hip_abduction（脚index73/82）：静的スイープでは単調だが、Dominici 2011の
          kicking synergy（股矢状面・膝・足首の協調）に外転内転は含まれない
    """

    def __init__(self, n_act, leg_r=(), leg_l=(), arm_r=(), arm_l=(), seed=None,
                 pair_offset=0):
        self.gen = ColoredNoiseGenerator(n_act, seed=seed)
        s = (lambda k: seed + k) if seed is not None else (lambda k: None)
        self.syn_leg = ColoredNoiseGenerator(1, seed=s(1))
        self.syn_arm_r = ColoredNoiseGenerator(1, seed=s(2))
        self.syn_arm_l = ColoredNoiseGenerator(1, seed=s(3))
        self._leg_r, self._leg_l = list(leg_r), list(leg_l)
        self._arm_r, self._arm_l = list(arm_r), list(arm_l)
        self._pair_offset = int(pair_offset)

    def _blend(self, gen_np, idx, s, syn_w):
        """関節indexの集合 idx にシナジー信号 s を syn_w の重みで混ぜる。

        idx の各要素は `(index, sign)` のタプル、または（後方互換）単なる int。
        sign は「生のqpos正方向が屈曲なら+1、伸展なら-1」（Noneなら符号不明＝無補正の
        旧来どおりの式を使う。上のクラスdocstring「2026-08-11・符号バグの修正」参照）。
        pair_offset があれば対になる筋（伸ばす側ラベル）にも混ぜる。"""
        n = len(gen_np)
        for entry in idx:
            if isinstance(entry, tuple):
                i, sign = entry
            else:
                i, sign = entry, None
            if i >= n:
                continue
            if self._pair_offset:
                # 拮抗筋2チャンネル（活性化[0,1]）モード。index iは配列上「常にneg方向筋
                # （曲げる側ラベル）」なので、signがラベルと逆（sign=+1）のときだけ
                # シナジー信号の符号を反転してから混ぜる（sign=Noneは旧来どおり無補正）。
                target_i = s if sign is None else -sign * s
                target_j = -s if sign is None else sign * s
                gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * target_i
                j = i + self._pair_offset
                if j < n:
                    gen_np[j] = (1 - syn_w) * gen_np[j] + syn_w * target_j
            else:
                # 関節空間・符号つき直接指令モード。
                target_i = s if sign is None else sign * s
                gen_np[i] = (1 - syn_w) * gen_np[i] + syn_w * target_i

    def sample(self, beta, synergy=False, syn_w=0.6):
        """引数betaで色付きノイズをshape=(n_act,)ぶん生成して返す。synergy=Trueの場合は、脚(左右逆位相)・右腕・左腕それぞれの共通シナジー信号を重みsyn_wで各関節の値へ混ぜてから返す。synergy=Falseなら混ぜずにそのまま返す。
        """
        gen_np = self.gen.sample(beta)
        if not synergy:
            return gen_np
        gen_np = gen_np.copy()
        leg_s = float(self.syn_leg.sample(beta)[0])
        self._blend(gen_np, self._leg_r, leg_s, syn_w)
        # 脚は左右逆位相（粗い交互パターン、Dominici 2011）
        self._blend(gen_np, self._leg_l, -leg_s, syn_w)
        self._blend(gen_np, self._arm_r, float(self.syn_arm_r.sample(beta)[0]), syn_w)
        self._blend(gen_np, self._arm_l, float(self.syn_arm_l.sample(beta)[0]), syn_w)
        return gen_np


def antagonist_map(motor_cmd, co_activation=0.3):
    """関節あたりの運動指令 motor_cmd(n_joint次元, [-1, 1]) を、拮抗筋2本ペアの
    活性化(2*n_joint次元, [0, 1])に写像する。MIMoの MuscleModel は
    先頭 n_joint 次元が負方向筋(neg=曲げる側)、後半 n_joint 次元が正方向筋(pos=伸ばす側)。

    【なぜ】人間の1関節は「曲げる筋」と「伸ばす筋」の2本の別々の筋肉で動く。両方を同時に
    力ませることを共収縮(co-activation)といい、新生児期に強く見られる主要パターン
    [Tier1、Hadders-Algra et al. 1992 "Developmental course of general movements in early
    infancy. II. EMG correlates"、健常乳児22名EMG+ビデオ]。文献要点：
      - co-activationの存在は確定（新生児期に主要パターン）
      - writhing→fidgety移行後も**残り続ける**主要パターン（＝消える設計は文献に反する）
      - 発達で①burst持続が短くなる ②振幅減衰 ③tonic背景活動が下がる（数値は非公開）

    【写像の意味】
      motor_cmd[i] = 0（動かない指令）＆ co_activation=0.3 → neg=0.3, pos=0.3
        ＝両方軽く力ませて関節を穏やかに固める（スティフネス上昇）
      motor_cmd[i] = +0.5（伸ばす指令）＆ co_activation=0.3 → neg=0.3, pos=0.8
        ＝曲げる筋も残しつつ伸ばす（硬さを保ちながら動く）
      motor_cmd[i] = +0.5 ＆ co_activation=0 → neg=0, pos=0.5（従来の独立駆動と等価）
      motor_cmd[i] = 0 ＆ co_activation=0.9 → neg=0.9, pos=0.9（関節ロック）

    【co_activation の値】新生児のCIの実測値は文献に見つからず [Tier3・ARBITRARY]。
    暫定 0.3 でスタート、目視・数値を見ながら調整する（把握反射TARGET=0.9等と同じ扱い）。
    月齢連動は将来やることリスト（消さない設計、Hadders-Algra文献に忠実）。
    """
    n = motor_cmd.shape[0]
    neg = np.clip(co_activation + np.maximum(-motor_cmd, 0.0), 0.0, 1.0)
    pos = np.clip(co_activation + np.maximum(+motor_cmd, 0.0), 0.0, 1.0)
    out = np.empty(2 * n, dtype=np.float32)
    out[:n] = neg
    out[n:] = pos
    return out


def is_antagonist_action(action, n_actuator):
    """行動配列が拮抗筋2本ペア形式か（＝長さが 関節数×2 か）を判定する。

    MIMo は身体の駆動方式が2種類ある：
      MuscleModel        … 1関節に2本の筋。行動は 関節数×2 次元・各要素 [0, 1]
      SpringDamperModel  … 1関節に1つのモーター。行動は 関節数 次元・各要素 [-1, 1]
    どちらで走っているかは行動配列の長さで判別できる。
    """
    return int(len(action)) >= 2 * int(n_actuator)


# 【2026-09-11】additive のとき、筋の表現のまま足すか、符号つきの指令に戻して
#   から足すか。**既定は前者＝筋の表現のまま足す**（E_SUM_BEFORE_MUSCLE の既定は "0"）。
#   1 にすると人間の最終共通路に近い形（符号つきに戻してから足す）になるが、
#   実測で悪化したため既定を 0 に戻した（下の2026-09-11の節）。
#
# 【2026-09-11・既定を 1 → 0 に戻した。実測で悪化したため】
#   F2-123pre（基準 F2-122pre と同設定・種91・600歩）で 1 にして走らせた結果：
#     保持中の速さ        16.7 → 18.5 度/秒（+11%）
#     着地時の速度       111.3 → 127.8 度/秒（+15%）
#     小命令(<2度)の符号一致 59.5 → 53.3%（-10%）
#     見えていた率        46.2 → 43.6%（-6%）
#   ほぼ全項目で悪化した。**予測は「ほとんど変わらない」だったので外れ。**
#   （机上でも実データでも正味の指令は新旧で完全に一致しており＝差が出た点0.0%、
#     変わるのは共収縮の有無だけ。それでもこれだけ動いた）
#
#   推測（実測ではない）：共収縮は関節を硬くするので、減衰が足りない系
#   （眼球のプラントは臨界減衰の2.7〜4%しかない）で**偶然ブレーキの役目**を
#   していた。人間に忠実な形にすると、そのブレーキが消える。
#
#   人間では共収縮は起きない【原文確認】（Sparks 2002 Fig.8：サッケード中、
#   拮抗筋側の運動ニューロンは一時的に発火がゼロになる）。つまり**この修正は
#   人間に近づける方向として正しいが、単独では成立しない**。
#   減衰の不足という別の問題を先に直す必要がある。
#   調査：doc/文献調査/二語文/2026-09-11_VORとサッケードの合成_人間側.md
#   記録：F/logs/_机上/保持の残り_原因の切り分け_2026-09-11.md
_SUM_BEFORE_MUSCLE = _os.environ.get("E_SUM_BEFORE_MUSCLE", "0") == "1"


def write_joint_command(action, joint_index, cmd, n_actuator, co_activation=0.0,
                        additive=False):
    """1つの関節への指令 cmd([-1, 1]) を、行動配列に**駆動方式に合った形で**書き込む。

    【2026-07-26・なぜ作ったか】反射（前庭動眼反射・視線誘導反射・口の探索）が
    そろって `action[joint_index] = cmd`（符号つき）と直接書いており、**筋肉モデルでは
    半分の指令が消えていた**。MuscleModel は行動を [0, 1] に切り捨てるので、
      正の値 → 「負方向筋」が収縮する（意図と無関係にいつも同じ向きへ動く）
      負の値 → 0 に切り捨て（＝何も起きない。正方向筋には触れていないので戻せない）
    となり、**眼も首も片方向にしか動けなかった**。実測：VOR を入れると左目の上下角が
    1秒で可動域の下限 −47度に張り付き、二度と戻らなかった（重力を切っても同じ）。
    ユーザーの目視「眼球だけずっと下に引っ張られてる」と一致。

    自発運動（脳側 explore → to_env_action）は `antagonist_map` を通しており正しかった。
    **反射だけがこの写像を飛ばしていた**ので、共通の入り口をここに用意する。

    Args:
        action: 書き込み先の行動配列（その場で書き換える）。
        joint_index: 関節の番号（＝アクチュエータの番号。0〜n_actuator-1）。
        cmd: その関節への指令。[-1, 1]。符号が向き。
        n_actuator: 関節（アクチュエータ）の総数。
        co_activation: 共収縮の度合い。拮抗筋形式のときだけ効く。
            眼球運動は既定の 0 でよい。人間の外眼筋は**相反神経支配**
            （Sherrington の相反神経支配の法則）で、一方が収縮するとき他方は
            積極的に弛緩する＝共収縮しない。四肢の共収縮とは別の話。
        additive: True なら既にある値に足す（複数の反射を重ねる場合）。

    Returns:
        書き換えた action（引数と同じ配列）。
    """
    c = float(np.clip(cmd, -1.0, 1.0))
    n = int(n_actuator)
    # 【2026-09-11・最終共通路】additive のとき、**筋の表現のまま足さない**。
    #   いったん符号つきの指令に戻して足し、書き込みは1回にする。
    #
    # 【なぜ・実測】VOR（空間に対する視線を留める）と定位反射の保持（頭に対する
    #   目の角度を留める）は、頭が動くと**定義上 逆を向く**。太郎では頭が
    #   中央値31.4度/秒で動いており（止まっているのは4%だけ・F2-122pre）、
    #   両者は87%の時間 逆符号だった（相関 −0.883・F2-121pre）。
    #   筋の表現のまま足すと、逆向きのとき**両方の筋が同時に縮む**（共収縮）。
    #   実測で指令の68%がここで消え、正味の向きは VOR に71%・保持に35%しか
    #   従っていなかった。
    #
    # 【人間はどうか】外眼筋の運動ニューロンは、前庭核(VOR)・バースト細胞
    #   (サッケード)・神経積分器(保持)・滑動性追跡から収束入力を受け、
    #   **合算されてから**1つの発火率になる（最終共通路）。相反神経支配で
    #   拮抗筋は積極的に抑制される＝共収縮しない。このファイルの
    #   co_activation の注釈（Sherrington の相反神経支配の法則）が既に
    #   そう書いているのに、additive の経路がそれを破っていた。
    #   ※ この生理の裏取りは調査中（doc/文献調査/二語文/
    #     2026-09-11_VORとサッケードの合成_人間側.md）。
    #
    # 【何が変わるか】向きが同じときは結果が変わらない（足し算は足し算）。
    #   **逆向きのときだけ変わる**＝そこが直したい所。
    #   E_SUM_BEFORE_MUSCLE=0 で従来の挙動に戻せる（比較用）。
    if additive and _SUM_BEFORE_MUSCLE:
        cur = read_joint_command(action, joint_index, n, co_activation)
        c = float(np.clip(cur + c, -1.0, 1.0))
        additive = False          # 合算済み。書くのは1回だけ
    if is_antagonist_action(action, n):
        neg = float(np.clip(co_activation + max(-c, 0.0), 0.0, 1.0))
        pos = float(np.clip(co_activation + max(+c, 0.0), 0.0, 1.0))
        if additive:
            action[joint_index] = float(np.clip(action[joint_index] + neg, 0.0, 1.0))
            action[joint_index + n] = float(np.clip(action[joint_index + n] + pos, 0.0, 1.0))
        else:
            action[joint_index] = neg
            action[joint_index + n] = pos
    else:
        if additive:
            action[joint_index] = float(np.clip(action[joint_index] + c, -1.0, 1.0))
        else:
            action[joint_index] = c
    return action


def read_joint_command(action, joint_index, n_actuator, co_activation=0.0):
    """行動配列から、1つの関節への指令（[-1, 1]・符号が向き）を取り出す。

    `write_joint_command` の逆。反射が方策の出力に上書きするとき、
    「今その関節に何の指令が来ているか」を知るために使う。
    """
    n = int(n_actuator)
    if is_antagonist_action(action, n):
        neg = float(action[joint_index]) - co_activation
        pos = float(action[joint_index + n]) - co_activation
        return float(np.clip(pos - neg, -1.0, 1.0))
    return float(action[joint_index])
