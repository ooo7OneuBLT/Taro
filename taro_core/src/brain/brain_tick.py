# -*- coding: utf-8 -*-
"""太郎の脳の「1周期」──毎tickで脳が行う処理そのもの（2026-09-13・段B-1）。

【なぜこのファイルがあるか】ここに書いてある処理は**太郎の能力**そのもの
（親の声を聞く・見た物の名前を言う・世界を予測する・寝ている間に復習する）。
それが長いあいだ `run/trainer.py`（＝実験を走らせる仕組み）の中にあった。

  ・`run/` は「実験する人の道具」で、太郎でも世界でもない
  ・そこに脳があると、core を読んでも本当の太郎が分からない
  ・「この機能を切ったらどうなるか」を測ろうとしても、切る場所が道具の中にある

2026-09-13 に洗い出したところ、`run/trainer.py` 2913行のうち**約1,863行が脳の働き**
だった。設計：`doc/設計_太郎をCoreで完結させる_2026-09-13.md`。

【いまの姿（段B完了・2026-09-13）】このクラスは **`Taro` が継承している**
（`run/taro_setup.py: class Taro(TaroBrainTick)`）。つまりここに書いてある処理は
**太郎のもの**で、`taro.名前()` で呼べる。走らせる側（`run/trainer.py`）は
`self.taro.名前()` と頼むだけになった。

  ・掲示板（測る道具の ctx）にはもう**一度も触らない**（動くコードで0箇所）
  ・自分の値は自分の作業台（`self.tick` ＝ `BrainTickState`）に置く
  ・時刻は自分の時計（`self.now` ＝ `BrainClock`。走らせる側が毎tick進める）
  ・世界は `self.env`、目に映った絵は**引数で受け取る**
  ・処理時間の計測（`_prof_t0`/`_prof_add`）は測る側の仕事なので、ここには
    「何もしない版」だけ置き、走らせる側が本物を差し込む

【どうやって移したか】`run/trainer.py` から**1文字も変えずに** ast で切り貼りし
（段B-1）、そのあと依存を1種類ずつ外して md5 で確かめた（段B-2）。
各段で基準の実験3本（従来の書き方／太郎が持つ／道具を外す）を走らせ、
`注意.csv`・`物体ファイル.csv`・`太郎の発話.csv`・`発話イベント.csv`・
`世界の予測器.csv` が**すべて一致**することを門にした。

【古い実験について】`taro.visual_attention` を書かない実験では、視覚と注意を
測る道具が動かして掲示板へ置く。太郎はもう掲示板を読まないので、
`run/trainer.py` の `_mirror_legacy_ctx` が道具の直後に作業台へ写している。
後方互換を消すときに、その橋も一緒に消せる。

【依存（移設時に実測した。これだけ）】
  モジュール直下 : torch / numpy / random / _cosine_sim（下に一緒に移した）
"""
import random

import numpy as np
import torch


def _cosine_sim(a, b):
    """コサイン類似度。F1-4b（語から注意への読み出し回路）の一致度計算に使う。

    a: np.ndarray、b: tuple/list（Lexicon.assoc()の戻り値）。どちらかがゼロベクトル
    なら0.0（run/plugins/common/word_learning.py の_cos()と同じ規約だが、Noneでなく
    0.0を返す＝呼び出し側set_recognition()がそのままmax(0,・)に渡せる形にする）。
    """
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class BrainClock:
    """太郎の時計（段B-2c・2026-09-13）。

    【なぜ要るか】太郎は「親が最後に話してから何秒たったか」「前に言ってから
    冷却時間が過ぎたか」を知る必要がある。これまではその時刻を**測る道具の掲示板**
    （`self.now.sim_sec` / `self.now.step`）から読んでいた。道具が時刻を持っているのは逆で、
    時刻は世界から来て太郎が受け取るもの。

    【誰が進めるか】走らせる側（`run/trainer.py`）が毎tickの最初に `set()` する。
    値の出どころも計算式も、掲示板が返していたものと同じ（`self.now.sim_sec` は
    `step × dt`、`self.now.step` はそのステップ番号）＝振る舞いは変わらない。
    """

    __slots__ = ("sim_sec", "step")

    def __init__(self):
        self.sim_sec = 0.0
        self.step = 0

    def set(self, *, step, dt):
        self.step = step
        self.sim_sec = step * dt


class BrainTickState:
    """太郎が1周期の中で自分に渡す値の置き場＝**脳の作業台**（段B-2a・2026-09-13）。

    【なぜ要るか】これまで太郎は、自分の中で受け渡すだけの値
    （いま何に注意しているか・さっき何と言ったか・世界の予測はどうだったか）を
    **測る道具の掲示板（ctx）**に置いていた。そのため
      ・道具を外すと太郎が自分の値を失う
      ・「道具は読むだけ」という約束が守れない
    という2つが同時に起きていた。

    【掲示板との関係】道具は今までどおり `ctx.<名前>` で読める。
    `run/context.py` の `Ctx` が、ここに無い名前を引かれたら**この作業台を見に行く**
    （写さない＝参照する）。写すと、道具が読む位置と太郎が書く位置がずれて
    1周期古い値を読むことになる。

    【属性は「書かれたときだけ」生える】あらかじめ None を入れない。
    `getattr(ctx, "X", None)` が今までと同じ答えを返すようにするため。
    """

    #: 掲示板から移した値（段B-2a）＋ Trainer が持っていた脳の状態（段B-2b）
    __slots__ = (
        "_active_assoc", "_last_parent_events", "_wp_last_parent_speak_sec",
        "_wp_parent_chunks", "_wp_prev_action", "_wp_surprise_trace",
        "_wp_z_max", "attended_object", "attention_point",
        "attention_switch_t", "efference", "goal_point",
        "last_babble", "last_parent_utterance", "last_produce", "last_vision_vec",
        "last_world_pred", "object_files", "priority_map_result",
        "priority_map_result_t", "salience_map", "surprise_trace",
        "vanish_misses", "world_pred_by_file", "world_pred_inputs",
        "world_predictor_grad_report",
    )

    def __repr__(self):
        got = [k for k in self.__slots__ if hasattr(self, k)]
        return "<脳の作業台 %d/%d>" % (len(got), len(self.__slots__))


class TaroBrainTick:
    """毎tickで太郎の脳が行う処理。`run/trainer.py` の Trainer が継承する。

    このクラスは単体では使えない（`self.cfg` / `self` / `self.ctx` など
    Trainer が持つものを参照している）。切り離すのは段B-2。
    """

    # ------------------------------------------------ 処理時間の計測（道具）
    #   【段B-2c3・2026-09-13】どこに何ミリ秒かかったかの計測は**測る側**の仕事で、
    #   太郎の能力ではない。太郎は「何もしない版」を持ち、走らせる側
    #   （run/trainer.py の build）が本物を差し込む。差し込まれなければ何も起きない。
    def _prof_t0(self):
        return None

    def _prof_add(self, key, t0):
        return None

    def _vision_backend_encode(self, o):
        """`t.vision_backend.encode()` に渡す画像を選ぶ（F1-7新設）。

        中心窩カメラ（`eye_left_fovea`/`eye_right_fovea`）が観測に含まれていれば
        そちらを、無ければ従来どおり `eye_left`/`eye_right` を渡す。
        `body.fovea_camera` が既定False（未指定シーン含む）のときは、観測に
        該当キーが無いため常に従来経路＝**1ビットも挙動が変わらない**。

        中心窩画像は撮影時点で既に視野15度に切り出し済み（MuJoCo側のカメラfovy）
        なので、下流バックエンド（例：dinov2_vits14）が持つ `fovea_crop` を
        二重にかけると情報を失う。バックエンドが `fovea_px` 属性を持つ場合のみ、
        encode()呼び出しの間だけ「crop が実質no-opになる値」へ一時的に差し替え、
        呼び出し後に元へ戻す（vision_backends.py は触ってよいファイルの外なので、
        ここでの一時上書きで対応する）。customバックエンドはfovea_crop自体を
        呼ばないので、この上書きは何もしない。
        """
        t = self
        # 【案A・2026-08-31・設計やり直し（ユーザー指示）】lexicon_vision.source=
        #   "wide" なら、認識には**周辺60度の画像まるごと1枚**を使う（中心窩カメラは
        #   使わない・切り出しもしない）。人間の「1枚の網膜像・周辺込み・1本の処理」
        #   に構造を合わせ、DINOの実行も1回で済む。既定（キー無し）は従来どおり。
        _lv = getattr(self.cfg, "lexicon_vision", None) or {}
        _wide = (_lv.get("source") == "wide") if isinstance(_lv, dict) else False
        has_fovea = (not _wide) and "eye_left_fovea" in o and "eye_right_fovea" in o
        if has_fovea:
            img_l, img_r = o["eye_left_fovea"], o["eye_right_fovea"]
        else:
            img_l, img_r = o["eye_left"], o["eye_right"]
        backend = t.vision_backend
        old_fovea_px = None
        has_fovea_px_attr = (has_fovea or _wide) and hasattr(backend, "fovea_px")
        if has_fovea_px_attr:
            old_fovea_px = backend.fovea_px
            backend.fovea_px = 10 ** 9   # fovea_crop側のmin(fovea_px,h)でno-op化
        # 【作業C・2026-08-24】この時間は呼び出し元（_hear_parent_utterance＝
        #   step_k内＝"env_step"に含まれる／_apply_word_attention・
        #   _apply_word_production＝"produce"に含まれる）の**内訳**として重ねて足す
        #   （他区分との合計が二重計上になる。CSV列名 t_vision_backend_sec は
        #   「他区分の一部」であって独立区分ではないと作業記録に明記）。
        _prof_t = self._prof_t0()
        try:
            return backend.encode(img_l, img_r)
        finally:
            if has_fovea_px_attr:
                backend.fovea_px = old_fovea_px
            self._prof_add("vision_backend", _prof_t)

    def _vision_channels(self, o):
        """語の意味に渡す視覚を、感覚チャンネルの辞書で返す（2026-08-28・F2-13）。

        `cfg.lexicon_peripheral` が False（既定）なら**リストを1本返すだけ**で、
        従来と1ビットも変わらない（周辺視の推論も行わない＝コストも同じ）。

        True のときは2チャンネルを返す：
          "vision"     … 中心窩カメラ（視野15度）。細部。従来と同じ値・同じ名前
                         （既存モデルの channels キーが "vision" なので変えない）
          "peripheral" … 周辺カメラ（視野60度）。全体の配置・輪郭

        【なぜ要るか】人間は中心窩（細部）と周辺視（全体）の両方で物を見分ける。
        太郎は中心窩1枚だけで判断しており、板の見かけ49.8度に対し中心窩は15度＝
        犬の首輪のあたりしか写らず、**形ではなく色で語を選んでいた**
        （2026-08-28実測。図＝F/logs/F2-12_視界確認/図_視界全体と中心窩.png）。

        周辺カメラには `fovea_crop` をかけない（かけると中央32pxだけになり
        中心窩とほぼ同じ絵になってしまう）。本体 `_vision_backend_encode` が
        中心窩カメラに対して行っているのと同じ一時上書きで no-op 化する。
        """
        base = self._vision_backend_encode(o)
        if not getattr(self.cfg, "lexicon_peripheral", False):
            return base
        fovea = base.tolist() if hasattr(base, "tolist") else list(base)
        if "eye_left" not in o or "eye_right" not in o:
            return {"vision": fovea}
        backend = self.vision_backend
        old = getattr(backend, "fovea_px", None)
        if old is not None:
            backend.fovea_px = 10 ** 9      # 切り出さず視野60度の全体を使う
        _prof_t = self._prof_t0()
        try:
            per = backend.encode(o["eye_left"], o["eye_right"])
        finally:
            if old is not None:
                backend.fovea_px = old
            self._prof_add("vision_backend", _prof_t)
        return {"vision": fovea,
                "peripheral": per.tolist() if hasattr(per, "tolist") else list(per)}

    def _context_feed(self, speaker, token_ids, vision_vec=None, gone=False, here=False):
        """文脈（GRUの隠れ状態）へ発話を1本流す（2026-08-30・設計_文脈（コンテキスト）.md）。

        speaker: "parent" か "self"。話者の印（決定3）を列の頭に付けて
        forward_hidden に通し、持続する文脈 taro._context_hidden を更新する。
        読み出し専用（no_grad）＝学習は1ビットも変えない。
        produce.context=false（既定）では何もしない。

        gone: 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う】
          Trueなら話者トークンの直後に<GONE>を挟む（＝「[親][消えた]バスないね」の
          形にする）。既定False（呼び出し元が渡さない）では従来と1ビットも
          変わらない。taro._gone_idがNone（produce.vanish_input無し）のときも
          何もしない（安全側）。

        here: 【M4d・2026-09-06・仕様_M4d_あるの印】Trueなら話者トークンの直後に
          <HERE>を挟む（＝「[親][ある]バスだね」の形にする。gone/hereは排他＝
          仕様上どちらか一方しか渡されない）。既定False・taro._here_idがNone
          （produce.here_input無し）のときは何もしない（安全側）。
        """
        t = self
        if not getattr(t, "_context_enabled", False):
            return
        # 【範囲ガード・2026-08-30】produce_vocab.encode() は実行中に未知の文字
        #   （「ゃ」等の拗音）へ新IDを動的に割り振るが、embedding は起動時の
        #   サイズのまま伸びない。従来はそのIDが脳に入る経路が無く無害だったが、
        #   文脈経路は脳へ直接食わせるので、口より大きいIDは門前で捨てる
        #   （実測：「にゃんにゃん」の「ゃ」= id78 が embedding 78個で範囲外）。
        _cap = t.brain.embedding.num_embeddings
        ids = [t._context_speaker_ids[speaker]] + [int(i) for i in token_ids
                                                   if int(i) < _cap]
        if gone and getattr(t, "_gone_id", None) is not None:
            ids = [ids[0], t._gone_id] + ids[1:]
        elif here and getattr(t, "_here_id", None) is not None:
            ids = [ids[0], t._here_id] + ids[1:]
        # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        #   「EOSを足す」節。既定listen_eos=False では通らない＝1ビットも変わらない。
        #   自己発話（speaker=="self"）は既に別経路でEOS付きのため対象外。
        if speaker != "self" and getattr(t, "_listen_eos", False):
            ids = ids + [2]
        if ids[0] >= _cap:
            return
        # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】状態の線。
        #   gone/hereは排他（呼び出し元の決定）。produce.state_channel（既定False）
        #   が偽なら常にNone＝forward_hiddenの挙動不変（1ビットも変わらない）。
        _state_channel = bool((self.cfg.produce or {}).get("state_channel", False))
        _state_id = None
        if _state_channel:
            _state_id = 2 if gone else (1 if here else None)
        x = torch.tensor([ids], dtype=torch.long, device=t.brain._device())
        # 【V1・2026-08-31】視覚トークン：そのとき目に映っているものの要約を
        #   変換層で64次元に翻訳し、列の頭に1トークンぶん添える
        #   （設計_視覚とGRUの統合.md。vision_context無効なら常にNone＝従来どおり）。
        _vin = None
        if vision_vec is not None and getattr(t, "_visual_projection", None) is not None:
            _vin = torch.tensor(list(vision_vec), dtype=torch.float32,
                                device=t.brain._device())
        # 【聞く学習・2026-08-31】発話を文脈に足す前に、その発話で1回学習する
        #   （次トークン予測・誤差逆伝搬。形は目標Bの learn_perception と同じ、
        #   違いは hidden=文脈 から始める点だけ＝発話をまたぐ依存を学べる）。
        #   既定OFF（listen_learn無し）ではこのifを通らない＝挙動不変。
        # 【2026-09-02・F2-46】listen_self=false なら太郎自身の発話では聞く学習を回さない
        #   （F2-45で自己発話1,183回の半分が誤りで本体を汚染した疑い）。文脈への投入は従来どおり。
        if (getattr(t, "_listen_learn", False) and len(ids) >= 2
                and (speaker != "self" or getattr(t, "_listen_self", True))):
            import torch.nn.functional as F
            xin = torch.tensor([ids[:-1]], dtype=torch.long, device=t.brain._device())
            tgt = torch.tensor([ids[1:]], dtype=torch.long, device=t.brain._device())
            # 視覚トークンは勾配つきで作る（変換層も一緒に育つ）。
            _pfx = t._visual_projection(_vin) if _vin is not None else None
            out, _ = t.brain.forward_hidden(xin, hidden=t._context_hidden,
                                            prefix_vec=_pfx, state_id=_state_id)
            if _pfx is not None:
                out = out[:, 1:, :]      # 先頭＝視覚トークン位置の出力は損失に使わない
            loss = F.cross_entropy(t.brain.perception_head(out)[0], tgt[0])
            t._listen_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p_ for g in t._listen_optimizer.param_groups for p_ in g["params"]),
                t._listen_grad_clip)
            t._listen_optimizer.step()
            # 【言語海馬・段階1・2026-09-01・設計_言語海馬と睡眠リプレイ.md】
            #   聞いた瞬間、海馬に1回で焼き付ける（人間の海馬の性質）。
            #   key_visは聞く学習ブロックがprefix_vecに使っている入力そのもの
            #   （_pfx＝visual_projection後の64次元）を再利用する。新たな描画・
            #   DINO呼び出しは追加しない。vision_context無効（_pfxがNone）の
            #   ときは視覚キーが無いのでzerosで代用する（書き込み自体は行う。
            #   speakerで語を分けたいだけの用途もあるため）。
            #   既定OFF（language_hippocampus無し）ではこのifを通らない＝挙動不変。
            #   【段階1の追加制約・2026-09-01】書き込むのは**親の発話だけ**。
            #   実測で太郎自身の喃語・言い間違い（「にいんにうん」等）が44件中30件
            #   書き込まれており、睡眠で再生すると誤りの自己増幅になるため。
            #   自分の声のリハーサルは段階2以降で別途設計する（設計に追記済み）。
            _is_parent = (getattr(t, "produce_vocab", None) is not None and
                          ids[0] == t.produce_vocab.char2idx.get("<PARENT>", -1))
            # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §3】chunk_level真の
            #   ときは海馬を塊の列で書く（_chunk_context_feedが別に書く）ので、
            #   ここ（モーラの列）では書かない。既定False（chunk_level無し）では
            #   このガードは常にFalse＝従来と1ビットも変わらない。
            if (getattr(t, "language_hippocampus", None) is not None and _is_parent
                    and not getattr(t, "chunk_level", False)):
                _kv = (_pfx.detach().cpu().numpy() if _pfx is not None
                       else np.zeros(t.brain.embedding.embedding_dim, dtype=np.float32))
                # written_atは診断専用（recall等の測定用）。学習ループ本体の
                # step変数はここまで届かないので、trainer側で持つ単純な連番でよい。
                _wa = getattr(self, "_lang_hippo_step", 0)
                # 【M4e・2026-09-07】驚きの書き込み：消えたの印が立っている間に
                #   聞いた発話は書く強さをgone_strength（既定1.0）にする。
                #   hpはproduce.hippocampusの辞書（無ければ{}＝既定1.0のまま）。
                _hp = (self.cfg.produce or {}).get("hippocampus") or {}
                _st = float(_hp.get("gone_strength", 1.0)) if gone else 1.0
                t.language_hippocampus.write(_kv, ids[0], ids[1:], _wa, strength=_st)
                self._lang_hippo_step = _wa + 1
        with torch.no_grad():
            _pfx2 = t._visual_projection(_vin) if _vin is not None else None
            _, h = t.brain.forward_hidden(x, hidden=t._context_hidden, state_id=_state_id,
                                          prefix_vec=_pfx2)
        t._context_hidden = h.detach()

    def _chunk_context_feed(self, speaker, chunk_ids, vision_vec=None, gone=False, here=False):
        """文脈（塊レベルGRUの隠れ状態）へ塊の列を1本流す（新設・2026-09-07）。

        仕様_M5_塊レベル層_2026-09-07.md 後半§4。_context_feed（音レベル）と同型で、
        入口が「塊のid列」である点だけが違う。produce.chunk_level=false（既定）では
        呼ばれない＝既存の音レベル経路は1ビットも変わらない。

        speaker: "parent" か "self"（t.chunk_vocab.specialsのキー）。
        gone/hereはM4/M4dと同じ規約（排他・話者トークンの直後に挟む）。
        """
        t = self
        cv = t.chunk_vocab
        # 【M7a・2026-09-08・仕様_M7a_世界の予測器_測るだけ】親が喋った塊の列を
        #   世界の予測器へ渡す置き場に控える（_world_predictor_stepが消費してNoneに
        #   戻す）。cfg.world_predictorがNone（既定）でも書き込み自体は行うが、
        #   読む側が無ければ使われないだけ＝既存挙動には影響しない。
        if speaker == "parent":
            self.tick._wp_parent_chunks = [int(i) for i in chunk_ids]
        # 【仕様書§4】音レベル側と同じ投射を使うが、勾配は流さない
        #   （taro._chunk_optimizerはchunk_brain.parameters()だけを持つため、
        #   visual_projectionへ逆伝播しても学習器に登録された対象が無く無駄）。
        # 【M6b・2026-09-07・仕様_M6b】message_layer.observe()が塊ごとの見た目
        #   （key_vis）を要るようになったため、_pfxの計算をobserve呼び出しより
        #   前に持ってきた（以前は下のブロックで計算していた）。
        _pfx = None
        if vision_vec is not None and getattr(t, "_visual_projection", None) is not None:
            _vin = torch.tensor(list(vision_vec), dtype=torch.float32,
                                device=t.chunk_brain._device())
            _pfx = t._visual_projection(_vin).detach()
        # 【言いたいことの層・2026-09-07・仕様_M6_言いたいことの層.md §2】
        #   親の発話を聞くたびに4つの表（塊の役割の元になる見た目・述語・順番）を
        #   数える。自己発話（speaker=="self"）では更新しない。既定False
        #   （produce.message_level無し）ではmessage_layerがNoneのまま＝
        #   このifは一度も通らず、既存の全経路は1ビットも変わらない。
        # 【M6b・2026-09-07】key_vis=親の発話時の見た目（_pfx）を渡す。役割判定が
        #   「状態と一緒に変わるか」から「見た目のばらつきが小さいか」に変わった
        #   ため（仕様_M6b_役割は見た目との結び付きで_2026-09-07.md）。
        if speaker == "parent" and getattr(t, "message_layer", None) is not None:
            _msg_state = "gone" if gone else ("here" if here else None)
            if _msg_state is not None:
                _key_vis = _pfx.detach().cpu().numpy() if _pfx is not None else None
                t.message_layer.observe([int(i) for i in chunk_ids], _msg_state,
                                        key_vis=_key_vis)
        ids = [cv.specials[speaker]]
        if gone and cv.specials.get("gone") is not None:
            ids.append(cv.specials["gone"])
        elif here and cv.specials.get("here") is not None:
            ids.append(cv.specials["here"])
        # 【仕様書§4】EOSを常に付ける（塊の列は短いので終わりを学ばせる）。
        ids = ids + [int(i) for i in chunk_ids] + [2]
        _state_channel = bool((self.cfg.produce or {}).get("state_channel", False))
        _state_id = None
        if _state_channel:
            _state_id = 2 if gone else (1 if here else None)
        if (getattr(t, "_listen_learn", False) and len(ids) >= 2
                and (speaker != "self" or getattr(t, "_listen_self", True))):
            import torch.nn.functional as F
            xin = torch.tensor([ids[:-1]], dtype=torch.long, device=t.chunk_brain._device())
            tgt = torch.tensor([ids[1:]], dtype=torch.long, device=t.chunk_brain._device())
            out, _ = t.chunk_brain.forward_hidden(xin, hidden=t._chunk_context_hidden,
                                                  prefix_vec=_pfx, state_id=_state_id)
            out = out[:, 1:, :]      # 先頭＝視覚トークン位置の出力は損失に使わない
            loss = F.cross_entropy(t.chunk_brain.perception_head(out)[0], tgt[0])
            t._chunk_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p_ for g in t._chunk_optimizer.param_groups for p_ in g["params"]),
                getattr(t, "_listen_grad_clip", 1.0))
            t._chunk_optimizer.step()
            # 【仕様書§4】海馬は塊の列で書く（親の発話だけ。_context_feedと同じ制約）。
            if (speaker == "parent"
                    and getattr(t, "language_hippocampus", None) is not None):
                _kv = (_pfx.detach().cpu().numpy() if _pfx is not None
                       else np.zeros(t.chunk_brain.embedding.embedding_dim, dtype=np.float32))
                _wa = getattr(self, "_lang_hippo_step", 0)
                _hp = (self.cfg.produce or {}).get("hippocampus") or {}
                _st = float(_hp.get("gone_strength", 1.0)) if gone else 1.0
                t.language_hippocampus.write(_kv, ids[0], ids[1:-1], _wa, strength=_st)
                self._lang_hippo_step = _wa + 1
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long, device=t.chunk_brain._device())
            _, h = t.chunk_brain.forward_hidden(x, hidden=t._chunk_context_hidden,
                                                prefix_vec=_pfx, state_id=_state_id)
        t._chunk_context_hidden = h.detach()

    def _world_predictor_step(self):
        """世界の予測器（M7a／M7b-1）を1tick進める。測るだけ（M7b-1で驚き→NEの1本だけ
        別の部品＝青斑核へ書き込む。他の部品には一切書き込まない）。

        仕様：F/docs/二語文/仕様_M7a_世界の予測器_測るだけ_2026-09-08.md
        「後半：実装担当向け技術付録」3節(c)。M7b-1追記：F/docs/二語文/
        仕様_M7b-1_物ごとの予測器と驚きの配線_2026-09-09.md「後半」2節。
        cfg.world_predictorがNone（既定）なら即returnし、追加計算は一切走らない
        ＝既存実験の挙動・コストは1ビットも変わらない。
        wp_cfg.multi_objectが偽（既定）なら従来（M7a・単一状態）どおり＝
        既定不変。self.tick.last_world_pred・self.ctx.world_pred_by_fileに
        置くだけ（読むのはプラグイン）。青斑核への書き込みは
        _world_predictor_step自身ではなく呼び出し側（on_step、NE更新の直後）が行う
        （self._wp_z_maxをここに置くだけ）。
        """
        t = self
        wp = getattr(t, "world_predictor", None)
        if wp is None:
            self.tick.last_world_pred = None
            self.tick._wp_z_max = None
            return
        import math

        wp_cfg = self.cfg.world_predictor if isinstance(self.cfg.world_predictor, dict) else {}
        multi_object = bool(wp_cfg.get("multi_object", False))

        # 【なぜ224.0か】object_files.py:438 CENTER_X, CENTER_Y = 112.0, 112.0
        #   （画像は224x224前提で中央=112,112）から逆算した画像1辺の長さ。
        IMG_SIZE = 224.0

        # ---- 親（このtickに喋ったか・どの塊か。全物に共通）----------------------
        parent_events = getattr(self.tick, "last_parent_utterance", None) or []
        parent_spoke = 1.0 if parent_events else 0.0
        if parent_spoke:
            self.tick._wp_last_parent_speak_sec = self.now.sim_sec
        chunk_ids = self.tick._wp_parent_chunks
        self.tick._wp_parent_chunks = None      # 消費（次tickへ持ち越さない）
        chunk_id_plus1 = (int(chunk_ids[0]) + 1) if chunk_ids else 0
        if self.tick._wp_last_parent_speak_sec is None:
            time_since = 1.0                # まだ一度も喋っていない＝上限扱い
        else:
            time_since = min(self.now.sim_sec - self.tick._wp_last_parent_speak_sec, 10.0) / 10.0

        # ---- 直前の運動（全物に共通）------------------------------------------
        act = self.tick._wp_prev_action
        act_list = (act.detach().cpu().numpy().reshape(-1).tolist()
                    if act is not None else None)

        # 【M7b-1改・2026-09-09】ポート型（PortWorldPredictor）分岐。既存の
        #   単一・多物分岐（下）は無改修。wp_cfg.ports（既定False）が真のときだけ
        #   ここへ来る（taro_setup.pyがports/multi_objectを排他で構築するため、
        #   同時に真になることは無い）。
        if bool(wp_cfg.get("ports", False)):
            self._world_predictor_step_ports(wp, wp_cfg, parent_events, parent_spoke,
                                              chunk_id_plus1, time_since, act_list)
            return

        if not multi_object:
            # ---- 従来どおり（M7a・単一状態・既定不変）--------------------------
            att = getattr(self.tick, "attended_object", None)
            if att is not None:
                present = 1.0
                visible = 1.0 if att.get("visible") else 0.0
                vanished = 1.0 if att.get("vanished") else 0.0
                pos = att.get("pos") or (0.0, 0.0)
                pos_x = float(pos[0]) / IMG_SIZE
                pos_y = float(pos[1]) / IMG_SIZE
                area = float(att.get("area") or 0.0)
                area_norm = math.log1p(max(area, 0.0)) / 10.0
                obj_state = [present, visible, vanished, pos_x, pos_y, area_norm]
                obj_vec = att.get("last_seen_vec")
            else:
                obj_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                obj_vec = None

            # 【なぜbuild_inputをここで呼ばないか、2026-09-08】WorldPredictor.step()の
            #   "中"（tbptt窓のflush=backward/optimizer.stepが終わった"後"）で入力を
            #   組み立てないと、学習対象パラメータ（obj_vec_norm・chunk_embedding）の
            #   計算グラフが更新前の重みのまま次の窓に紛れ込み、in-placeエラーになる
            #   （world_predictor.pyの_build_input docstring参照）。そのためstep()には
            #   生の値のまま渡す。
            input_now = {
                "obj_state": obj_state, "obj_vec": obj_vec,
                "parent_spoke": parent_spoke, "chunk_id_plus1": chunk_id_plus1,
                "time_since_parent": time_since, "act": act_list,
            }
            target = {
                "obj_state": obj_state,
                "obj_vec": obj_vec,
                "parent_spoke": parent_spoke,
                "parent_chunk_id_plus1": chunk_id_plus1 if parent_spoke else None,
            }
            res = wp.step(input_now, target)

            self.tick.last_world_pred = {
                "step": self.now.step,
                "t_sec": round(self.now.sim_sec, 3),
                "present": obj_state[0],
                "visible": obj_state[1],
                "vanished": obj_state[2],
                "parent_spoke": parent_spoke,
                "parent_text": (parent_events[0].get("text", "") if parent_events else ""),
                "err_state": res["err_state"],
                "err_vec": res["err_vec"],
                "err_parent": res["err_parent"],
                "err_slow": res["err_slow"],  # 追記2026-09-08（err_totalには含めない）
                "err_total": res["err_total"],
                "baseline": res["baseline"],
                "z": res["z"],
            }
            self.tick._wp_z_max = None   # 単一状態モードでは驚き→NEの配線自体を使わない
            return

        # ---- 多物モード（M7b-1・2026-09-09）------------------------------------
        # 仕様「後半」2節：files=ctx.object_files.files、since_seenの小さい順に
        # max_files個。注意中の物は必ず含める。wp.active_ids()にあって今回の
        # 一覧に無いidはwp.drop(id)（記憶を捨てる＝物が視界から長く消えたら忘れる）。
        ofs = getattr(self.tick, "object_files", None)
        files = list(getattr(ofs, "files", None) or [])
        max_files = int(wp_cfg.get("max_files", 4))
        att = getattr(self.tick, "attended_object", None)
        attended_id = att.get("file_id") if att is not None else None

        # 【F2-92 の実測を受けた追加・2026-09-09】4000歩で物体ファイルが62個でき、寿命10tick未満が7個、
        #   注意外の物の驚き（z>2）が702回（注意中は514回）。一瞬しか続かない検出（机の縁・親の手など）にも
        #   予測器の記憶を作り、その驚きが z_max に混ざる。`min_hits`（既定0＝従来どおり）以上の検出回数が
        #   ある物だけを予測の対象にする。注意中の物は常に含める
        min_hits = int(wp_cfg.get("min_hits", 0))
        if min_hits > 0:
            files = [f for f in files if int(getattr(f, "hits", 0)) >= min_hits or f.id == attended_id]

        files_sorted = sorted(files, key=lambda f: f.since_seen)
        selected = files_sorted[:max_files]
        if attended_id is not None and not any(f.id == attended_id for f in selected):
            att_f = next((f for f in files if f.id == attended_id), None)
            if att_f is not None:
                # 【実装判断・2026-09-09、仕様に明記が無いため理由を残す】容量が
                #   いっぱいなら「since_seenが一番大きい（一番長く見ていない）」
                #   物を追い出して注意中の物を入れる（since_seen昇順ソート済みの
                #   末尾を1つ削る）。注意中の物を必ず含める、という指示を
                #   満たす最小の変更。
                if len(selected) >= max_files:
                    selected = selected[:max_files - 1]
                selected = selected + [att_f]

        selected_ids = set(f.id for f in selected)
        if not hasattr(self, "_wp_app0"):
            self._wp_app0 = {}          # file_id -> 最初に見たときの appearance（appearance_freeze 用）
        for old_id in list(wp.active_ids()):
            if old_id is not None and old_id not in selected_ids:
                wp.drop(old_id)
                self._wp_app0.pop(old_id, None)

        if not selected:
            # 【仕様「後半」2節】「物が無いとき（filesが空）は何もしない
            #   （ctx.last_world_pred=None）」。
            self.tick.last_world_pred = None
            self.tick.world_pred_by_file = {}
            self.tick._wp_surprise_trace = {}
            self.tick.surprise_trace = {}
            self.tick._wp_z_max = None
            return

        vanish_misses = getattr(self.tick, "vanish_misses", 1)

        # 【実装判断・2026-09-09、仕様に無かった点】仕様「後半」2節はwp.step_for(f.id,...)を
        #   物ごとに独立に呼ぶ形を示すが、机上確認でそのまま実装すると「ある物の
        #   flush（optimizer.step、重みをin-placeで更新）が、まだflushしていない
        #   別の物の計算グラフが参照している同じ重みを書き換えてしまい、後で
        #   backwardしようとした瞬間に`RuntimeError: ...modified by an inplace
        #   operation`になる」ことを実測した（仕様が警告する_build_input事故と
        #   同型）。tbptt=1が防ぐのは「1物の中で窓が2tick以上に伸びる」ことだけで、
        #   「複数の物が同じ重みを共有しながら独立にflushする」ことは防げない。
        #   対策：1tickにつき「全物のobserve_for（誤差を集めるだけ・まだbackward
        #   しない）」→「flush_pendingを1回だけ（全物ぶんまとめてbackward+
        #   optimizer.step）」→「全物のpredict_for（次tickの予測を作る）」の
        #   二段構えにした（world_predictor.py側の新設メソッド。詳細はそちらの
        #   クラス冒頭コメント参照）。単一状態（multi_object=False）の経路は
        #   一切変更していない。
        per_file_io = {}
        for f in selected:
            present = 1.0
            visible = 1.0 if f.misses == 0 else 0.0
            vanished = 1.0 if f.misses >= vanish_misses else 0.0
            pos_x = float(f.pos[0]) / IMG_SIZE
            pos_y = float(f.pos[1]) / IMG_SIZE
            area_norm = math.log1p(max(float(f.area or 0.0), 0.0)) / 10.0
            obj_state = [present, visible, vanished, pos_x, pos_y, area_norm]
            obj_vec = f.appearance
            # 【F2-92 の実測を受けた追加・2026-09-09】物体ファイルの appearance（マスク内パッチ平均の EMA、
            #   検出のたびに 0.3 で更新）は 0.5 秒ごとに跳ぶ。見えている間の見た目の誤差が F2-91（注意中の物の
            #   中心窩 CLS、0.006）に対し 0.722、物の状態の誤差も 0.153→0.586 に悪化した。`appearance_freeze`
            #   （既定False＝従来どおり）なら、その物を最初に見たときの appearance を凍結して入力・的に使う
            #   （「同じ物だ」という同一性の手がかりとしてだけ使い、細かい変動は予測させない）
            if bool(wp_cfg.get("appearance_freeze", False)):
                app0 = self._wp_app0.get(f.id)
                if app0 is None and f.appearance is not None:
                    import numpy as _np
                    app0 = _np.array(f.appearance, dtype=float).copy()
                    self._wp_app0[f.id] = app0
                if app0 is not None:
                    obj_vec = app0
            per_file_io[f.id] = (obj_state, obj_vec, visible, vanished)

        by_file_result = {}
        z_candidates = []          # (z_state, file_id) のリスト。z_maxの元
        attended_pack = None       # 注意中の物の[res, obj_state, err_slow]。last_world_pred用
        for f in selected:
            obj_state, obj_vec, visible, vanished = per_file_io[f.id]
            target = {
                "obj_state": obj_state,
                "obj_vec": obj_vec,
                "parent_spoke": parent_spoke,
                "parent_chunk_id_plus1": chunk_id_plus1 if parent_spoke else None,
            }
            res = wp.observe_for(f.id, target)

            is_attended = (f.id == attended_id)
            by_file_result[f.id] = {
                "visible": visible, "vanished": vanished,
                "err_state": res["err_state"], "z_state": res["z_state"],
                "attended": is_attended,
            }
            if res["z_state"] is not None:
                z_candidates.append((res["z_state"], f.id))
            if is_attended:
                attended_pack = [res, obj_state]

        wp.flush_pending()

        for f in selected:
            obj_state, obj_vec, visible, vanished = per_file_io[f.id]
            input_now = {
                "obj_state": obj_state, "obj_vec": obj_vec,
                "parent_spoke": parent_spoke, "chunk_id_plus1": chunk_id_plus1,
                "time_since_parent": time_since, "act": act_list,
            }
            err_slow = wp.predict_for(f.id, input_now)
            if f.id == attended_id and attended_pack is not None:
                attended_pack.append(err_slow)

        self.tick.world_pred_by_file = by_file_result

        if z_candidates:
            z_max, z_max_id = max(z_candidates, key=lambda pair: pair[0])
        else:
            z_max, z_max_id = None, None
        self.tick._wp_z_max = z_max

        # ---- 注意への加点用の余韻（仕様「後半」2節末尾）-------------------------
        # trace <- 0.8*trace + max(0, z_state-thresh)（毎tick、10tick≈1秒で1/10に減衰）。
        # object_files.pyがattend_surprise_gain（既定0）で読むだけ＝既定不変。
        thresh = float(wp_cfg.get("ne_surprise_thresh", 2.0))
        new_trace = {}
        for file_id, d in by_file_result.items():
            z_state = d["z_state"] if d["z_state"] is not None else 0.0
            prev = self.tick._wp_surprise_trace.get(file_id, 0.0)
            new_trace[file_id] = 0.8 * prev + max(0.0, z_state - thresh)
        self.tick._wp_surprise_trace = new_trace
        self.tick.surprise_trace = new_trace

        parent_text = (parent_events[0].get("text", "") if parent_events else "")
        ne_level = t.ne.get_ne_level()
        if attended_pack is not None and len(attended_pack) == 3:
            res, obj_state, err_slow = attended_pack
            self.tick.last_world_pred = {
                "step": self.now.step,
                "t_sec": round(self.now.sim_sec, 3),
                "present": obj_state[0],
                "visible": obj_state[1],
                "vanished": obj_state[2],
                "parent_spoke": parent_spoke,
                "parent_text": parent_text,
                "err_state": res["err_state"],
                "err_vec": res["err_vec"],
                "err_parent": res["err_parent"],
                "err_slow": err_slow,
                "err_total": res["err_total"],
                "baseline": res["baseline"],
                "z": res["z"],
                "n_files": len(selected),
                "z_max": z_max, "z_max_id": z_max_id, "ne_level": ne_level,
            }
        else:
            # 注意中の物が今回の選抜に無い（例：一度も注意していない）。
            # 【仕様「後半」2節】「物が無いとき（files空）は何もしない
            #   （ctx.last_world_pred=None）」。ここは「物はあるが注意中の物が
            #   無い」場合で、files自体が空のときと同じ扱いにする
            #   [実装判断・2026-09-09、仕様に明記が無いため理由を残す]。
            self.tick.last_world_pred = None

    def _world_predictor_step_ports(self, wp, wp_cfg, parent_events, parent_spoke,
                                     chunk_id_plus1, time_since, act_list):
        """世界の予測器・ポート型（M7b-1改）を1tick進める。

        仕様：F/docs/二語文/仕様_M7b-1改_ポート型の世界の予測器_2026-09-09.md
        「後半：実装担当向け技術付録」2節。_world_predictor_step（呼び出し元）が
        「wp_cfg.get('ports')が真」のときだけここへ分岐する。既存の単一・多物
        分岐は無改修のため、物の選抜規則（min_hits・appearance_freeze・注意中の
        物を必ず含める・上限・drop）はここに複製している（元の多物分岐と同じ
        規則。仕様「後半」2節「物の選抜はM7b-1の多物分岐と同じ」）。

        wp（PortWorldPredictor）とのやり取りは observe_all(...) → predict_all(...)
        の2回だけ（内部で三段。world_predictor.py側のPortWorldPredictorクラス
        docstring参照）。
        """
        import math
        IMG_SIZE = 224.0

        ofs = getattr(self.tick, "object_files", None)
        files = list(getattr(ofs, "files", None) or [])
        max_files = int(wp_cfg.get("max_files", 4))
        att = getattr(self.tick, "attended_object", None)
        attended_id = att.get("file_id") if att is not None else None

        min_hits = int(wp_cfg.get("min_hits", 0))
        if min_hits > 0:
            files = [f for f in files
                     if int(getattr(f, "hits", 0)) >= min_hits or f.id == attended_id]

        files_sorted = sorted(files, key=lambda f: f.since_seen)
        selected = files_sorted[:max_files]
        if attended_id is not None and not any(f.id == attended_id for f in selected):
            att_f = next((f for f in files if f.id == attended_id), None)
            if att_f is not None:
                if len(selected) >= max_files:
                    selected = selected[:max_files - 1]
                selected = selected + [att_f]

        selected_ids = set(f.id for f in selected)
        if not hasattr(self, "_wp_app0"):
            self._wp_app0 = {}          # file_id -> 最初に見たときのappearance（appearance_freeze用。多物分岐と共有属性）
        for old_id in list(wp.active_ids()):
            if old_id is not None and old_id not in selected_ids:
                wp.drop(old_id)
                self._wp_app0.pop(old_id, None)

        parent_text = (parent_events[0].get("text", "") if parent_events else "")

        if not selected:
            # 【仕様「後半」2節】既存の多物分岐と同じ扱い：物が無いとき何もしない。
            self.tick.last_world_pred = None
            self.tick.world_pred_by_file = {}
            self.tick._wp_surprise_trace = {}
            self.tick.surprise_trace = {}
            self.tick._wp_z_max = None
            return

        vanish_misses = getattr(self.tick, "vanish_misses", 1)

        per_file_io = {}
        for f in selected:
            present = 1.0
            visible = 1.0 if f.misses == 0 else 0.0
            vanished = 1.0 if f.misses >= vanish_misses else 0.0
            pos_x = float(f.pos[0]) / IMG_SIZE
            pos_y = float(f.pos[1]) / IMG_SIZE
            area_norm = math.log1p(max(float(f.area or 0.0), 0.0)) / 10.0
            obj_state = [present, visible, vanished, pos_x, pos_y, area_norm]
            obj_vec = f.appearance
            if bool(wp_cfg.get("appearance_freeze", False)):
                app0 = self._wp_app0.get(f.id)
                if app0 is None and f.appearance is not None:
                    import numpy as _np
                    app0 = _np.array(f.appearance, dtype=float).copy()
                    self._wp_app0[f.id] = app0
                if app0 is not None:
                    obj_vec = app0
            per_file_io[f.id] = (obj_state, obj_vec, visible, vanished)

        vision_targets = {fid: {"obj_state": io[0], "obj_vec": io[1]}
                          for fid, io in per_file_io.items()}
        hearing_target = {"parent_spoke": parent_spoke,
                           "parent_chunk_id_plus1": chunk_id_plus1 if parent_spoke else None}
        body_target = {"act": act_list}

        obs = wp.observe_all(vision_targets, hearing_target, body_target)

        # 【なぜ同じ辞書をvision_inputsにも使うか】obj_state/obj_vecはこのtick
        #   時点で確定した値であり、observe(前tickの予測との答え合わせ)とpredict
        #   (次tickの予測を作る入力)のどちらも「今tickの実際の観測」を使う
        #   （既存WorldPredictorのstep_forも同じ値をtarget・input両方に使っている）。
        vision_inputs = vision_targets
        hearing_input = {"parent_spoke": parent_spoke, "chunk_id_plus1": chunk_id_plus1,
                          "time_since_parent": time_since}
        body_input = {"act": act_list}
        pred = wp.predict_all(vision_inputs, hearing_input, body_input)

        # 【仕様_M7b-1改3「後半」1節・2026-09-10】予測器が毎tick受け取った入力を
        #   ctxに置くだけ（読むだけの世界の予測器_record.pyが拾う）。既存の変数
        #   （vision_inputs・hearing_input・body_input・attended_id・per_file_io）を
        #   参照するだけで、ports以外の分岐・既存の経路は一切触らない。
        #   「self.step」は仕様書の記述だがTrainerにその属性は無いため
        #   太郎の時計 self.now.step と読み替えた（実装判断・仕様に明記が無い点。理由：
        #   world_predictor_logの t_sec と同じ round(self.now.sim_sec, 3) の
        #   隣で使う「今のtick番号」はself.ctx.stepしかない）。
        self.tick.world_pred_inputs = {
            "step": self.now.step,
            "t_sec": round(self.now.sim_sec, 3),
            "vision": vision_inputs,
            "hearing": hearing_input,
            "body": body_input,
            "attended_id": attended_id,
            "visible": {fid: bool(io[2]) for fid, io in per_file_io.items()},
            "vanished": {fid: bool(io[3]) for fid, io in per_file_io.items()},
        }

        err_slow = pred["err_slow"]

        # 【勾配検査・仕様_M7b-1改2「後半」2節】走行の300歩目（既定。
        #   wp_cfg.grad_check_stepで変更可）に1回だけ、新しい重みが全部動いたかを
        #   機械で調べる。self.ctx.stepは単調増加のため、この等号チェックは
        #   自然に「1回だけ」になる。grad_report()はPortWorldPredictor専用
        #   （既存WorldPredictorには無い）。
        _gc_step = int(wp_cfg.get("grad_check_step", 300))
        if self.now.step == _gc_step and hasattr(wp, "grad_report"):
            self.tick.world_predictor_grad_report = wp.grad_report()

        z_hearing = obs.get("z_hearing")
        z_body = obs.get("z_body")

        # 【2026-09-09・仕様_見る側の段構成_実装 6節】驚きの計算への受け口。
        #   moving（段0：目が動いている最中）または、注意の切り替え直後
        #   switch_refractory_s秒以内は「今起きた変化」を驚きとして数えない
        #   （遠心性コピー・切り替えの遅れという輪の歯止め）。
        #   ctx.efferenceが無い（段0無効）・attention_switch_tが無い（段6無効）
        #   ときはmasked=False固定＝既定不変。
        _switch_refractory_s = float(wp_cfg.get("switch_refractory_s", 0.0))
        _eff = getattr(self.tick, "efference", None)
        _switch_t = getattr(self.tick, "attention_switch_t", -1e9)
        masked = bool((_eff and _eff.get("moving"))
                       or (self.now.sim_sec - _switch_t < _switch_refractory_s))

        by_file_result = {}
        z_candidates = []
        z_vec_candidates = []
        attended_entry = None
        for f in selected:
            _, _, visible, vanished = per_file_io[f.id]
            d = obs["by_file"].get(f.id, {})
            is_attended = (f.id == attended_id)
            by_file_result[f.id] = {
                "visible": visible, "vanished": vanished,
                "err_state": d.get("err_state"), "z_state": d.get("z_state"),
                # 【新奇さz_vec・仕様_M7b-1改4「後半」2節】驚き（出来事、z_state）と
                #   新奇さ（見た目、z_vec）を2列に分けて通す。注意・青斑核への候補は
                #   従来どおりz_state側だけ（z_candidatesはz_stateのみ、下も不変）。
                "z_vec": d.get("z_vec"),
                "attended": is_attended,
                # 【2026-09-09・仕様_見る側の段構成_実装 6節】記録用。元のz_state
                #   はこのまま残す（マスクするのはz_candidates・surprise_traceだけ）。
                "masked": masked,
            }
            # 【同6節】maskedのtickはz_stateを0として候補に渡す（z_max・attend_port
            #   から外す代わり、常に候補には入れる＝0を渡す、が仕様の文言）。
            # 【2026-09-09・仕様_見る側_道を1本にする 後半4節（中6の直し）】
            #   元のz_state自体がNone（この物体ファイルにまだ驚きの予測値が
            #   無い）なら、maskedでもNoneのまま（候補に入れない）。0を渡すのは
            #   「元は値があったがmaskedで隠す」場合だけ。
            _z_state = d.get("z_state")
            z_for_candidate = None if _z_state is None else (0.0 if masked else _z_state)
            if z_for_candidate is not None:
                z_candidates.append((z_for_candidate, f.id))
            if d.get("z_vec") is not None:
                z_vec_candidates.append(d["z_vec"])
            if is_attended:
                attended_entry = d

        self.tick.world_pred_by_file = by_file_result

        if z_candidates:
            z_max, z_max_id = max(z_candidates, key=lambda pair: pair[0])
        else:
            z_max, z_max_id = None, None
        self.tick._wp_z_max = z_max
        z_vec_max = max(z_vec_candidates) if z_vec_candidates else None

        # ---- 注意への加点用の余韻（既存の多物分岐と同じ式）----------------------
        thresh = float(wp_cfg.get("ne_surprise_thresh", 2.0))
        new_trace = {}
        for file_id, d in by_file_result.items():
            # 【2026-09-09・仕様_見る側の段構成_実装 6節】maskedのtickはz_stateを
            #   0として渡す（moving中・切り替え直後の偽の驚きを積ませない）。
            z_state = 0.0 if masked else (d["z_state"] if d["z_state"] is not None else 0.0)
            prev = self.tick._wp_surprise_trace.get(file_id, 0.0)
            new_trace[file_id] = 0.8 * prev + max(0.0, z_state - thresh)
        self.tick._wp_surprise_trace = new_trace
        self.tick.surprise_trace = new_trace

        ne_level = self.ne.get_ne_level()

        # 【ポートごとの驚き・仕様_M7b-1改2「後半」3節】候補
        #   {("vision", file_id): z_state} ∪ {("hearing",): z_hearing, ("body",): z_body}
        #   の最大を「今いちばん驚いているポート」とする（行動は変えない、記録だけ）。
        # 【修正2026-09-09・実装担当】z_candidates の要素は
        #   (z_state, f.id) の順（970行付近 z_candidates.append((d["z_state"], f.id))
        #   と、直後の max(z_candidates, key=lambda pair: pair[0]) が根拠）。
        #   ここを (fid, zs) で受けると z_state と file_id が入れ替わり、
        #   vision候補の比較値が file_id になってしまう（机上確認400stepで
        #   attend_port列に "vision:-8.51..." のような小数のidが出て発覚）。
        attend_candidates = {}
        for zs, fid in z_candidates:
            attend_candidates[("vision", fid)] = zs
        if z_hearing is not None:
            attend_candidates[("hearing",)] = z_hearing
        if z_body is not None:
            attend_candidates[("body",)] = z_body
        if attend_candidates:
            _best_key = max(attend_candidates, key=lambda k: attend_candidates[k])
            attend_port = (f"vision:{_best_key[1]}" if _best_key[0] == "vision"
                            else _best_key[0])
        else:
            attend_port = ""

        if attended_entry is not None and attended_entry.get("err_state") is not None:
            obj_state = per_file_io[attended_id][0]
            self.tick.last_world_pred = {
                "step": self.now.step,
                "t_sec": round(self.now.sim_sec, 3),
                "present": obj_state[0],
                "visible": obj_state[1],
                "vanished": obj_state[2],
                "parent_spoke": parent_spoke,
                "parent_text": parent_text,
                "err_state": attended_entry["err_state"],
                "err_vec": attended_entry["err_vec"],
                "err_parent": obs["err_parent"],
                "err_body": obs["err_body"],
                "err_slow": err_slow,
                "err_total": attended_entry["err_total"],
                "baseline": attended_entry["baseline"],
                "z": attended_entry["z"],
                "n_files": len(selected),
                "z_max": z_max, "z_max_id": z_max_id, "ne_level": ne_level,
                "z_hearing": z_hearing, "z_body": z_body, "attend_port": attend_port,
                "z_vec_att": attended_entry.get("z_vec"), "z_vec_max": z_vec_max,
            }
        else:
            # 注意中の物が今回の選抜に無い、または初回tick（まだ予測が無い）。
            # 【仕様_M7b-1改2「後半」3節、実装判断・仕様に明記が無いため理由を残す】
            #   多物分岐（既存）はこの分岐でctx.last_world_pred=Noneにしていたが、
            #   ポート型では聴覚・体は常に1ポートあるので、視覚が無い/初回tickでも
            #   z_hearing・z_body・attend_portの行を出す（既存列は空欄）。仕様
            #   「last_world_predがNoneになる分岐でも、portsのときは聴覚・体が
            #   あるのでdictを作る」に従う。
            self.tick.last_world_pred = {
                "step": self.now.step,
                "t_sec": round(self.now.sim_sec, 3),
                "present": None, "visible": None, "vanished": None,
                "parent_spoke": parent_spoke, "parent_text": parent_text,
                "err_state": None, "err_vec": None,
                "err_parent": obs["err_parent"], "err_body": obs["err_body"],
                "err_slow": err_slow, "err_total": None, "baseline": None, "z": None,
                "n_files": len(selected), "z_max": z_max, "z_max_id": z_max_id,
                "ne_level": ne_level,
                "z_hearing": z_hearing, "z_body": z_body, "attend_port": attend_port,
                "z_vec_att": None, "z_vec_max": z_vec_max,
            }

    def _ensure_chunk_capacity(self):
        """塊の名簿が塊GRUの入力口（embedding）を追い越していたら、入力口を伸ばす。

        _ensure_brain_capacity（音レベル）と同型（2026-09-07・仕様_M5_塊レベル層）。
        """
        t = self
        cv = t.chunk_vocab
        if cv.size <= t.chunk_brain.embedding.num_embeddings:
            return
        t.chunk_brain.resize_embedding(cv.size)
        if getattr(t, "_chunk_optimizer", None) is not None:
            lr = t._chunk_optimizer.param_groups[0]["lr"]
            t._chunk_optimizer = torch.optim.Adam(t.chunk_brain.parameters(), lr=lr)

    def _ensure_brain_capacity(self):
        """脳の名簿が入力口（embedding）を追い越していたら、入力口を伸ばす。

        【なぜ・2026-08-31】新しい音を聞いた瞬間に名簿は増えるが、embedding は
        起動時サイズのままだった。従来は「口に無い音は捨てる」ガードで凌いでいたが、
        それは「にゃんにゃん」を「にんにん」に静かに変換していた（実測で発覚）。
        人間は言えない音でも聞き分けられる（知覚＞産出）ので、聞いた音は全部
        脳のトークンにし、入力口の側を伸ばす。既存の重みは resize_embedding が
        先頭にコピーして保つ。伸びたら listen 学習の optimizer は古いパラメータを
        掴んだままなので作り直す（掴み直さないと学習が静かに空振りする）。
        """
        t = self
        pv = t.produce_vocab
        if pv.size <= t.brain.embedding.num_embeddings:
            return
        t.brain.resize_embedding(pv.size)
        t.brain.set_vocab_mapping(pv.char2idx)
        if getattr(t, "_listen_optimizer", None) is not None:
            import itertools
            _extra = (list(t._visual_projection.parameters())
                      if getattr(t, "_visual_projection", None) is not None else [])
            _params = itertools.chain(t.brain.embedding.parameters(),
                                      t.brain.gru.parameters(),
                                      t.brain.perception_head.parameters(),
                                      _extra)
            lr = t._listen_optimizer.param_groups[0]["lr"]
            t._listen_optimizer = torch.optim.Adam(_params, lr=lr)

    def _to_produce_ids(self, text):
        """発話テキストを脳の語彙のID列にする（文脈・聞く学習用）。

        【⓪・2026-08-31】以前は脳の語彙に無い文字（「ゃ」・カタカナ等）を捨てて
        いたが、聞いた音は名簿に登録して入力口を伸ばす方式に変えた（知覚＞産出）。
        長音の展開は耳と同じ規則。
        """
        from hearing import normalize_kana
        ids = self.produce_vocab.encode(normalize_kana(text))
        self._ensure_brain_capacity()
        return ids

    def _hear_parent_utterance(self, o, info):
        """親の発話（info["parent_utterance"]）を耳→連合器へ渡す（2026-08-18新設・F1-3）。

        taro.hearing/taro.lexicon は既定None（cfg.hearing=False）＝この関数は
        呼ばれても何もしない＝既存実験の挙動は1ビットも変わらない
        （E/scripts/parent_labeling.py が無効なら info にキー自体が無いので、
        いずれにせよ何も起きない）。

        仕様書（F/docs/仕様_F1-3_....md 技術付録「部品2」）の指定どおり：
          tokens = taro.hearing.hear(text)
          state  = taro.fusion.vision(左目画像, 右目画像)  ※視覚64次元のみ
          taro.lexicon.observe(tokens, state=state)
        confidences は F1-2 仕様書の単純化「発話全体を1チャンクとして扱ってよい」に従い、
        全要素同値（谷が生まれない＝発話全体が1つの単位として切り出される）を渡す。
        """
        t = self
        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return
        pu = info.get("parent_utterance") if isinstance(info, dict) else None
        if not pu:
            return
        text = pu.get("text")
        tokens = t.hearing.hear(text)
        confidences = [1.0] * len(tokens)
        # 【二語文・2026-08-31】分節の自信度を本物にする（produce.real_confidence、
        #   既定false＝従来の1.0決め打ちのまま1ビットも変わらない）。
        #   GRUの予測確率（文脈つき）を渡すと、語の境目で自信が谷になり
        #   lexicon が発話を複数の単位に切り出せる（統計的分節・Saffran 1996）。
        _pd = self.cfg.produce or {}
        if bool(_pd.get("real_confidence", False)) and                 getattr(t, "produce_vocab", None) is not None:
            _ids = self._to_produce_ids(text)
            if len(_ids) == len(tokens):
                _ps = t.brain.token_probs(_ids, hidden=t._context_hidden)
                if len(_ps) == len(tokens):
                    confidences = _ps
        # detach＝この統計（共起の平均）は学習の逆伝播に使わない値であるため
        #   （Lexiconは純Pythonの累積平均。計算グラフを持ち越さない）。
        # 【2026-08-19・F1-3b】視覚表現の作り方はt.vision_backend経由（差し替え可能）。
        #   既定null時はcustomバックエンドがtaro.fusion.visionをそのまま呼ぶだけなので、
        #   出力は従来コード（t.fusion.vision(...).detach().cpu().tolist()）と同一
        #   （custom backendのencode()内でも同じdetach().cpu()を行う）。
        # 【F2-13・2026-08-28】lexicon_peripheral=False（既定）なら
        #   従来と同じリストが返る＝1ビットも変わらない。
        _st = self._vision_channels(o)
        state = _st if isinstance(_st, dict) else _st.tolist()
        # 【F2-8・2026-08-25】「見慣れた景色」の平均を育てる（reverse_lookupの
        #   両側引き算に使う）。語彙にも想像にも触らない純粋な足し算で、
        #   すでに計算済みのstateを渡すだけ＝追加の計算コストはゼロ。
        t.lexicon.observe_view(state)
        # 【分節第2案・2026-09-03】設計_分節（語の切れ目の発見）.md 第2案 第2部
        #   「切り出しの差し替え」節。既定segment_mode="valley"では_end_probsが
        #   Noneのまま＝下のobserve呼び出しは従来と同じ2引数呼び出しになり
        #   1ビットも変わらない。
        _end_probs = None
        if (getattr(t, "_segment_mode", "valley") == "end_prob"
                and getattr(t, "produce_vocab", None) is not None
                and getattr(t, "_context_speaker_ids", None) is not None):
            _ids_ep = self._to_produce_ids(text)
            if len(_ids_ep) == len(tokens):
                _parent_id = t._context_speaker_ids.get("parent")
                if _parent_id is not None:
                    _ps_ep = t.brain.end_probs(_ids_ep, hidden=t._context_hidden,
                                               start=_parent_id)
                    if len(_ps_ep) == len(tokens):
                        _end_probs = _ps_ep
        if _end_probs is not None:
            chunk = t.lexicon.observe(tokens, confidences, state=state,
                                      end_probs=_end_probs, mode="end_prob")
        else:
            chunk = t.lexicon.observe(tokens, confidences, state=state)
        # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §3】observe直後に
        #   このtickの塊の列（lexicon.last_chunks）を塊idへ変換しておく。
        #   実際に文脈・海馬へ流す（_chunk_context_feed呼び出し）のは、下の
        #   _context_enabled ブロックで視覚ベクトル・gone/hereが揃ってから
        #   （observe直後の時点ではまだ計算されていない）。既定False
        #   （chunk_level無し）ではNoneのまま＝何も起きない。
        _chunk_cids = None
        if getattr(t, "chunk_level", False) and getattr(t, "chunk_vocab", None) is not None:
            _chunk_cids = t.chunk_vocab.encode_chunks(t.lexicon.last_chunks)
            self._ensure_chunk_capacity()
        # 【なぜstateもここに置くか】word_learningプラグイン（読むだけ）が「正解物の
        #   特徴EMA」を作るのに使う（プラグインがtaro.fusionを呼び直すと二重計算に
        #   なるうえ、この瞬間のobs（o）はstep_kのK tick内の値でctx.last["obs_out"]
        #   より新しい＝取り直すと精度が落ちる）。
        # 【親の言い直し・2026-09-03】設計の表には無いが、pu の cause/correct を
        #   ここで落とすと word_learning.py の発話イベントCSVに載らなくなる
        #   （検証で発覚）。pu に無ければ従来どおり None＝挙動不変。
        self.tick._last_parent_events.append(
            {"text": text, "target": pu.get("target"), "state": state,
             "cause": pu.get("cause"), "correct": pu.get("correct")})
        # 【文脈・2026-08-30・設計_文脈（コンテキスト）.md 決定1〜3】親の発話を
        #   文脈へ流す。親が見せる物（target）が変わったら場面の切り替わりと
        #   みなして文脈をリセットする（決定1「場面ごと」）。既定OFFでは
        #   _context_feed が最初のifで即returnし、1ビットも変わらない。
        if getattr(t, "_context_enabled", False) and getattr(t, "produce_vocab", None) is not None:
            _tgt = pu.get("target")
            if t._context_last_target is not None and _tgt != t._context_last_target:
                t._context_hidden = None
                # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §4】音レベルの
                #   文脈がリセットされる場所で塊レベルの文脈も一緒にリセットする。
                if getattr(t, "chunk_level", False):
                    t._chunk_context_hidden = None
            t._context_last_target = _tgt
            # 【V1】聞く側にだけ視覚を添える（設計の分岐A＝案2。言う側はV3で）。
            #   state は lexicon 用に計算済みの DINOv2 ベクトル（再計算なし）。
            _vv = state.get("vision") if isinstance(state, dict) else state
            # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う】
            #   （2026-09-06改訂：条件分岐を言語側から物体ファイル側へ移した後の形）
            #   言語は常に「注意している物体ファイルが保っている見た目」を読む。
            #   attended_objectが有れば、vanishedの真偽によらずvision_vecを
            #   last_seen_vecにする（見えている間は毎tick更新＝今までと同じ、
            #   消えた後は自然にバスの見た目が残る。仕様書「空の机問題の解き方」節）。
            #   vanishedが真のときだけ話者トークンの直後に<GONE>を挟む
            #   （「だね」と「ないね」の区別節）。既定False（produce.vanish_input無し）・
            #   attended_objectが無い・last_seen_vecがまだ無い（一度も見えていない）
            #   ときは従来どおり（網膜の生の像をそのまま使う）。
            _gone = False
            _here = False
            if bool((self.cfg.produce or {}).get("vanish_input", False)):
                _att = getattr(self.tick, "attended_object", None)
                if _att is not None and _att.get("last_seen_vec") is not None:
                    _vv = _att["last_seen_vec"]
                    _gone = bool(_att.get("vanished"))
                    # 【M4d・2026-09-06・仕様_M4d_あるの印】消えていないなら
                    #   「あるの印」を挟む（produce.here_input無しなら
                    #   _context_feed内でtaro._here_idがNone＝何もしない）。
                    _here = (not _gone) and bool(
                        (self.cfg.produce or {}).get("here_input", False))
            self._context_feed("parent", self._to_produce_ids(text),
                               vision_vec=_vv, gone=_gone, here=_here)
            # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §3】音レベルの
            #   _context_feed（切れ目の発見の維持）はそのまま呼んだ上で、
            #   塊レベルのGRU・海馬にも同じ発話を塊の列として流す。
            if getattr(t, "chunk_level", False) and _chunk_cids is not None:
                self._chunk_context_feed("parent", _chunk_cids,
                                         vision_vec=_vv, gone=_gone, here=_here)
        # 【発話の動機・2026-08-31】随伴の判定：自分が言った後、窓内に親の声が
        #   来たら報酬1（正誤は見ない＝設計B'）。言わなかった決定は窓切れ（上のA）
        #   で報酬0になる。social未設定なら不実行。
        _sg = getattr(t, "_speech_gate", None)
        if _sg is not None and getattr(t, "_social_pending", None) is not None:
            _act, _t0 = t._social_pending
            if _act and (self.now.sim_sec - _t0) <= float(
                    t._social_cfg.get("window_sec", 2.0)):
                _sg.resolve(True, 1.0, sim_sec=self.now.sim_sec)
                t._social_pending = None
        # 【2026-08-19新設・F1-4b】語から注意への読み出し回路（設計：
        #   F/docs/設計_F1-4b_語から注意への読み出し回路.md 後半「部品2」）。
        #   cfg.word_attentionがNoneのままなら、この行自体を実行しない
        #   （lexicon.assoc()の呼び出しコストすら払わない＝既定挙動・コスト不変）。
        if self.cfg.word_attention is not None and chunk is not None:
            vec = t.lexicon.assoc(chunk)
            if vec is not None:
                self.tick._active_assoc = (vec, self.now.sim_sec)

    def _apply_word_attention(self, o):
        """語から注意への読み出し回路（設計：F/docs/設計_F1-4b_....md 後半「部品2」）。

        「思い浮かべている」語のメモ（_active_assoc）と、いま実際に見ているものの
        DINOv2表現の一致度を、視線誘導反射（OrientingReflexV2）へ渡す。渡された
        反射側は、一致度が高い間は保持（hold）の解除を遅らせる
        （E/scripts/e_orienting_v2.py の set_recognition() / REC_THRESHOLD）。

        既定（cfg.word_attention is None）では最初のifで即returnし、DINOv2の
        encode等の追加計算は一切走らない＝既存実験の挙動・コストは1ビットも
        変わらない（仕様の要求どおり）。
        """
        wa = self.cfg.word_attention
        if wa is None or not wa.get("enabled", True):
            return
        t = self
        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return          # 耳無効なら思い浮かべる元(_active_assoc)自体が育たない
        orienting = getattr(self.env.unwrapped, "_orienting", None)
        if orienting is None:
            return          # 視線誘導反射(orienting_reflex)が無効なら渡す先が無い
        active_sec = float(wa.get("active_sec", 3.0))
        active = self.tick._active_assoc
        now = self.now.sim_sec
        if active is not None and (now - active[1]) < active_sec:
            vec = self._vision_backend_encode(o)
            sim = _cosine_sim(vec, active[0])
            orienting.set_recognition(max(0.0, sim))
        else:
            # 期限切れ（ACTIVE_SEC超過）。信号を明示的に0へ戻す
            #   （戻さないと反射側に前回のsimが居座り続けて保持が解除されなくなる）。
            orienting.set_recognition(0.0)

    def _apply_word_production(self, o):
        """見た物の名前を言う（F2「初語」、設計：F/docs/設計_F2_初語（見た物の名前を
        言う）.md 第2部）。

        既定（cfg.produce is None）では最初のifで即returnし、逆引き
        （lexicon.reverse_lookup）・generate()・報酬計算等の追加計算は一切走らない
        （既存実験の挙動もコストも1ビットも変わらない）。

        トリガー＝「じっと見ていて（視線誘導反射が保持中＝サッケード中でなく
        _should_hold()が真）、かつそれが何か分かっている（逆引きの確信度が
        閾値以上）」（設計「いつ言うか」節）。既存の信号を読むだけで、
        新しい仕組みは作らない（E/scripts/e_orienting_v2.py:1058 _should_hold()・
        :1132 holdingと同じ定義）。直前に喋っていなければ（クールダウン）発声する。
        """
        pd = self.cfg.produce
        self.tick.last_produce = None
        self.tick.last_babble = None
        # 【M3・2026-09-06・仕様_M3_注意中の物体ファイルと消失信号】このtickで
        #   脳が実際に使った視覚ベクトルの置き場。既定Noneにしておき、下で
        #   _vision_channels(o) を計算した直後に実値を入れる（早期returnした
        #   tickでは前回値を持ち越さずNoneのまま＝object_files.py側の「見えて
        #   いる間だけ控える」判定を誤らせないため）。
        self.tick.last_vision_vec = None
        if pd is None or not pd.get("enabled", True):
            return
        t = self
        if getattr(t, "produce_learner", None) is None:
            return          # 産出の配線が無ければ何もしない

        # 【F2-1・実装作業④・2026-08-23】喃語モード（mode="babble"）。
        #   設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「④」。
        #   mode未指定（既定"word"）ならこのifを一歩も通らず、以降の従来どおりの
        #   word_mode処理（視覚トリガー・逆引き・報酬・学習）に進む＝1ビットも
        #   変わらない。babbleモードは視覚トリガーを使わず、クールダウンだけで
        #   自発的に発声し、逆引き・報酬・学習は一切通らない（声を出して
        #   小脳の帳面に書くだけ）。
        if pd.get("mode", "word") == "babble":
            self._apply_babble(pd)
            return

        if getattr(t, "hearing", None) is None or getattr(t, "lexicon", None) is None:
            return          # 耳/連合器の配線が無ければ何もしない
        orienting = getattr(self.env.unwrapped, "_orienting", None)
        if orienting is None:
            return          # 視線誘導反射が無効なら「注視している」を判定できない
        # 【発話の動機・2026-08-31・設計_発話の動機.md】窓が切れた決定を報酬0で
        #   締める（毎ステップ通るここで行う）。social未設定なら sg=None で不実行。
        _sg = getattr(t, "_speech_gate", None)
        if _sg is not None and t._social_pending is not None:
            _act, _t0 = t._social_pending
            if self.now.sim_sec - _t0 > float(t._social_cfg.get("window_sec", 2.0)):
                _sg.resolve(_act, 0.0, sim_sec=self.now.sim_sec)
                t._social_pending = None

        # ① 逆引き：いま見ている視覚ベクトル → 一番似ているchunk（言いたい語）。
        # 【F2-13・2026-08-28】lexicon_peripheral=True なら中心窩＋周辺視の
        #   2チャンネルになる。既定Falseでは従来と同じ1本のベクトルが返る。
        vec = self._vision_channels(o)
        # 【M3・2026-09-06】脳が実際に語の選択に使った視覚ベクトルをそのまま控える
        #   （object_files.py が「注意中の物の見た目」として使う。仕様書決定2：
        #   物体ファイル自身のappearanceとは別物なのでここでしか取れない）。
        self.tick.last_vision_vec = vec
        # 【F2-8・2026-08-25】逆引きの前に「見慣れた景色」の平均へ今の見えを足す。
        #   ここは毎ステップ通るので、産出中は太郎が見たものすべてが平均に入る。
        t.lexicon.observe_view(vec)

        # 【M4・2026-09-06・仕様_M4_消えた物について「○○ないね」と言う】
        #   黙る門：注意している物が無いとき（走行の最初や、記録が消えた後）は
        #   発話しない（M2で見つかった「空の視界で『くつだね』」の対策）。
        #   ctx.attended_objectはobject_files.py（プラグイン）が前tickに置いた値
        #   （プラグインは_apply_word_productionの後に走るため。事前確認4）。
        #   既定False（produce.vanish_input無し）ではこの分岐を一歩も通らず、
        #   従来どおり逆引きへ進む＝1ビットも変わらない。
        _vanish_input = bool(pd.get("vanish_input", False))
        _att = getattr(self.tick, "attended_object", None) if _vanish_input else None
        if _vanish_input and _att is None:
            self.tick.last_produce = {
                "target_word": "", "sim": 0.0, "generated_word": "",
                "choice": None, "reward": None, "plan_length": 0, "known_moras": 0,
                "gate": "no_object", "gone": 0, "here": 0, "attended_id": "",
            }
            return
        _vanished = bool(_att.get("vanished")) if _att is not None else False
        _attended_id = _att.get("file_id", "") if _att is not None else ""
        # 【M4】自己回帰へ渡す実際の視覚（vanished時はlast_seen_vecに差し替える。
        #   last_seen_vecがまだ無い（一度も見えていない物に注意している異常系）
        #   ときは無理に差し替えず、フラグだけFalseに落として従来どおり素通り。
        _gone_now = False
        if _vanish_input and _vanished and _att.get("last_seen_vec") is not None:
            _gone_now = True
        # 【M4d・2026-09-06・仕様_M4d_あるの印】消えていないなら「あるの印」を
        #   立てる（gone/hereは排他）。here_input無効なら常にFalse＝従来どおり。
        _here_input = bool(pd.get("here_input", False))
        _here_now = bool(
            _here_input and _att is not None and not _gone_now
            and _att.get("last_seen_vec") is not None)
        chunk, sim = t.lexicon.reverse_lookup(vec)
        # 【文脈・2026-08-30・設計_文脈（コンテキスト）.md 決定4】語の選択に
        #   文脈の「言いやすさ」を足す：点数 = コサイン + λ × 幾何平均確率。
        #   λ（produce.context_lambda・既定0.0）が0なら一切通らない＝従来どおり。
        #   閾値の判定は選ばれた語の**コサイン側**で行う（0.80の較正を保つ）。
        # 【2026-09-04・混乱防止】word_choice="gru_hippo"のときはこのブロックの
        #   結果（chunk, sim）が直後の段階2ブロックで丸ごと上書きされ、λは
        #   一切効かない（起動時に_setup_produceが組み合わせを検証して止める・
        #   taro_setup.py参照）。無駄な計算を避けるためここでも通らないようにする。
        _wc = pd.get("word_choice", "lexicon")
        _lam = float(pd.get("context_lambda", 0.0))
        if (_wc != "gru_hippo" and _lam > 0.0
                and getattr(t, "_context_enabled", False)
                and t._context_hidden is not None):
            _scores = t.lexicon.reverse_scores(vec)
            _best, _bestv = None, None
            for _c, _cos in _scores.items():
                _ids = self._to_produce_ids(t.hearing.vocab.decode(list(_c)))
                _p = t.brain.sequence_prob(_ids, hidden=t._context_hidden)
                _v = _cos + _lam * _p if _p is not None else _cos
                if _bestv is None or _v > _bestv:
                    _best, _bestv = _c, _v
            if _best is not None:
                chunk, sim = _best, _scores[_best]
        # 【段階2・海馬の即答・2026-09-02・設計_言語海馬と睡眠リプレイ.md 段階2】
        #   produce.word_choice="gru_hippo" のとき、語の選択を表（lexicon逆引き）から
        #   「本体（GRU）自力 vs 海馬の想起」の2候補比較へ切り替える（既定"lexicon"＝従来不変）。
        #     本体：視覚トークン＋<PARENT>から1音ずつ最尤に手繰った語。自信＝先頭音の確率
        #     海馬：いまの視覚に最も似た記憶の語。自信＝似ている度×記憶の強さ×hippo_gain
        #   自信の高い方を言う（人間側：定着語は皮質・新語は海馬。H.M.型）。
        #   選ばれた自信を sim に入れ、以降の閾値判定（produce.threshold）はそのまま使う。
        # 【2026-09-04・文脈の配線】これまでこの生成は常に hidden=None（ゼロ）から
        #   始まり、聞く学習で育てた「文脈を使う能力」が発話時には一度も使われて
        #   いなかった（コード精読で発覚。設計の誤りではなく配線の欠落）。
        #   hidden=t._context_hidden を渡すことで、直前までの会話を出発点にする。
        #   context無効（既定）では t._context_hidden は常にNoneのままなので、
        #   この変更は文脈OFFの実験には1ビットも影響しない。
        if (_wc == "gru_hippo" and vec is not None
                and getattr(t, "_visual_projection", None) is not None
                and getattr(t, "produce_vocab", None) is not None):
            _pv = t.produce_vocab
            _par = _pv.char2idx.get("<PARENT>")
            _dev = t.brain._device()
            _gone_id_v = getattr(t, "_gone_id", None)
            _here_id_v = getattr(t, "_here_id", None)
            # 【M4・2026-09-06改訂】言語は常に注意中の物体ファイルの見た目を読む。
            #   attended_objectがありlast_seen_vecがあれば、vanishedの真偽に
            #   かかわらずvecをそれに差し替える（常に。仕様書(b)）。見えている間は
            #   last_seen_vecが毎tick更新されるので中身は今までのvecと同じになる。
            #   vanish_input無効、またはlast_seen_vecがまだ無い（一度も見えていない）
            #   ときは従来のvecのまま＝1ビットも変わらない。
            _vec_for_key = vec
            if _vanish_input and _att is not None and _att.get("last_seen_vec") is not None:
                _vec_for_key = _att["last_seen_vec"]
            _vin = torch.tensor(list(_vec_for_key), dtype=torch.float32, device=_dev)
            # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】生成の4か所
            #   すべて同じ値。produce.state_channel（既定False）が偽ならNone。
            _state_channel_g = bool(pd.get("state_channel", False))
            _gen_state_id = None
            if _state_channel_g:
                _gen_state_id = 2 if _gone_now else (1 if _here_now else None)
            # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §6】chunk_level真の
            #   ときだけ、本体側（GRU＋海馬の候補生成）を塊GRU＋塊の名簿に丸ごと
            #   差し替える。以降（980行付近のchunk=決定後）は無改修（発音側は
            #   文字列だけを見るため）。既定False・chunk_vocab/chunk_brain未構築
            #   ではelse節（従来のモーラGRU経路）がそのまま実行され、既存実験の
            #   挙動は1ビットも変わらない。
            self._last_chunk_seq_used = None
            _chunk_ready = (bool(getattr(t, "chunk_level", False))
                            and getattr(t, "chunk_vocab", None) is not None
                            and getattr(t, "chunk_brain", None) is not None)
            if _chunk_ready:
                cv = t.chunk_vocab
                _par_c = cv.specials.get("parent")
                _gone_id_c = cv.specials.get("gone")
                _here_id_c = cv.specials.get("here")
                # 【言いたいことの層・2026-09-07・仕様_M6_言いたいことの層.md §3】
                #   message_level真のときだけ、本体側の生成を「名詞の塊＋述語の
                #   塊を別々に選ぶ」組み立てに差し替える。既定False・
                #   message_layer未構築ではelse節（M5のままの塊GRU自己回帰）が
                #   そのまま実行され、既存実験の挙動は1ビットも変わらない。
                _msg_ready = (bool(getattr(t, "message_level", False))
                              and getattr(t, "message_layer", None) is not None)
                self._last_message = None
                if _msg_ready:
                    with torch.no_grad():
                        _key = t._visual_projection(_vin)
                        # 先頭の塊の分布だけ取る（<PARENT>直後、prefix_vec=見た目、
                        #   印は入れない＝状態を名詞選びに漏らさない。state_idも
                        #   同じ理由で渡さない）。
                        _out0, _hh0 = t.chunk_brain.forward_hidden(
                            torch.tensor([[_par_c]], dtype=torch.long, device=_dev),
                            hidden=t._chunk_context_hidden, prefix_vec=_key)
                        # 【M6d・2026-09-07】<PARENT>の直後は学習時に必ず印（<HERE>/<GONE>）が
                        #   来るので、印を入れずに読むと分布の質量は印に集まり名詞は 0.001〜0.01
                        #   （F2-88c：全窓で海馬に負け、教えていない語の窓は空）。ここでは
                        #   **常に <HERE>** を入れて「見えている物の名前」の分布を読む。本当の
                        #   状態（消えた）は述語の表にだけ渡す＝名詞選びに状態を漏らさない、は保つ。
                        _here_c = t.chunk_vocab.specials.get("here")
                        if _here_c is not None:
                            _out0, _hh0 = t.chunk_brain.forward_hidden(
                                torch.tensor([[_here_c]], dtype=torch.long, device=_dev),
                                hidden=_hh0)
                        _logits0 = t.chunk_brain.perception_head(_out0)[0, -1]
                        _probs0 = torch.softmax(_logits0, dim=-1)
                        # 【Tier3・仕様書§3】softmax上位から役割nounの塊だけ残す。
                        #   上位候補数20は恣意的定数（塊の名簿の規模に対して
                        #   十分大きく取っただけ、文献根拠なし）。
                        _topk = min(20, _probs0.shape[-1])
                        _top_vals, _top_idx = torch.topk(_probs0, _topk)
                    _noun_candidates = [
                        (int(_i), float(_p))
                        for _p, _i in zip(_top_vals.tolist(), _top_idx.tolist())
                        if t.message_layer.role(int(_i)) == "noun"]
                    _msg_state = "gone" if _gone_now else ("here" if _here_now else None)
                    _seq, _conf_g, _roles_g = t.message_layer.compose(
                        _msg_state, _noun_candidates)
                    _seq_clean = [tk for tk in _seq if tk != _gone_id_c and tk != _here_id_c]
                    _word_g = "".join(cv.chunk_string(tk, t.hearing.vocab) for tk in _seq_clean)
                    _noun_str, _pred_str = "", ""
                    for _tk, _rl in zip(_seq, _roles_g):
                        _s = cv.chunk_string(_tk, t.hearing.vocab)
                        if _rl == "noun":
                            _noun_str = _s
                        elif _rl == "pred":
                            _pred_str = _s
                    # 【仕様書§3】last_produceに"noun"・"pred"（文字列）・"roles"を
                    #   持たせる。海馬側が選ばれても（下のsim比較後）、名詞・述語の
                    #   内訳は塊レベルGRUの組み立て結果を報告する（海馬の想起は
                    #   役割情報を持たないため。仕様に無かった判断・作業記録に記載）。
                    self._last_message = {
                        "noun": _noun_str, "pred": _pred_str, "roles": list(_roles_g)}
                else:
                    with torch.no_grad():
                        _key = t._visual_projection(_vin)
                        _out, _hh = t.chunk_brain.forward_hidden(
                            torch.tensor([[_par_c]], dtype=torch.long, device=_dev),
                            hidden=t._chunk_context_hidden, prefix_vec=_key,
                            state_id=_gen_state_id)
                        _logits = t.chunk_brain.perception_head(_out)[0, -1]
                        # 【M4/M4d同型】開始トークンの直後にgone/hereを強制する。
                        if _gone_now and _gone_id_c is not None:
                            _out, _hh = t.chunk_brain.forward_hidden(
                                torch.tensor([[_gone_id_c]], dtype=torch.long, device=_dev),
                                hidden=_hh, state_id=_gen_state_id)
                            _logits = t.chunk_brain.perception_head(_out)[0, -1]
                        elif _here_now and _here_id_c is not None:
                            _out, _hh = t.chunk_brain.forward_hidden(
                                torch.tensor([[_here_id_c]], dtype=torch.long, device=_dev),
                                hidden=_hh, state_id=_gen_state_id)
                            _logits = t.chunk_brain.perception_head(_out)[0, -1]
                        _conf_g = float(torch.softmax(_logits, dim=-1).max())
                        _seq = []
                        for _ in range(4):    # 【仕様書§6】塊は最大4個で止める（Tier3、恣意的定数）
                            _tk = int(torch.argmax(_logits))
                            if _tk == 2:
                                break
                            _seq.append(_tk)
                            _out, _hh = t.chunk_brain.forward_hidden(
                                torch.tensor([[_tk]], dtype=torch.long, device=_dev),
                                hidden=_hh, state_id=_gen_state_id)
                            _logits = t.chunk_brain.perception_head(_out)[0, -1]
                    _seq_clean = [tk for tk in _seq if tk != _gone_id_c and tk != _here_id_c]
                    _word_g = "".join(cv.chunk_string(tk, t.hearing.vocab) for tk in _seq_clean)
                _word_h, _conf_h = "", 0.0
                _toks_clean = []
                _hip = getattr(t, "language_hippocampus", None)
                if _hip is not None and len(_hip) > 0:
                    _want = "gone" if _gone_now else ("here" if _here_now else "none")
                    _toks, _simh, _str = self._hippo_recall_filtered(
                        _hip, _key.detach().cpu().numpy(), want=_want,
                        gone_id=_gone_id_c, here_id=_here_id_c)
                    _toks_clean = [tk for tk in _toks if tk != _gone_id_c and tk != _here_id_c]
                    _word_h = "".join(cv.chunk_string(tk, t.hearing.vocab) for tk in _toks_clean)
                    _conf_h = float(pd.get("hippo_gain", 1.0)) * float(_simh) * float(_str)
                if _word_h and _conf_h > _conf_g:
                    _word, sim, _src = _word_h, _conf_h, "hippo"
                    self._last_chunk_seq_used = list(_toks_clean)
                else:
                    _word, sim, _src = _word_g, _conf_g, "gru"
                    self._last_chunk_seq_used = list(_seq_clean)
                chunk = tuple(t.hearing.vocab.encode(_word)) if _word else None
                self._last_choice = {"gru": [_word_g, round(_conf_g, 3)],
                                     "hippo": [_word_h, round(_conf_h, 3)], "chosen": _src}
            else:
                with torch.no_grad():
                    _key = t._visual_projection(_vin)
                    _out, _hh = t.brain.forward_hidden(
                        torch.tensor([[_par]], dtype=torch.long, device=_dev),
                        hidden=t._context_hidden, prefix_vec=_key, state_id=_gen_state_id)
                    _logits = t.brain.perception_head(_out)[0, -1]
                    # 【M4】本体の自己回帰は、開始トークン(<PARENT>)の直後に<GONE>を
                    #   強制してから続きを手繰る（仕様書(b)）。自信(_conf_g)は
                    #   <GONE>強制後の最初の実文字への確信度で測る。
                    # 【M4d・2026-09-06】<GONE>と対で、消えていないときは<HERE>を
                    #   同じ位置に強制する（_gone_now/_here_nowは排他）。
                    if _gone_now and _gone_id_v is not None:
                        _out, _hh = t.brain.forward_hidden(
                            torch.tensor([[_gone_id_v]], dtype=torch.long, device=_dev),
                            hidden=_hh, state_id=_gen_state_id)
                        _logits = t.brain.perception_head(_out)[0, -1]
                    elif _here_now and _here_id_v is not None:
                        _out, _hh = t.brain.forward_hidden(
                            torch.tensor([[_here_id_v]], dtype=torch.long, device=_dev),
                            hidden=_hh, state_id=_gen_state_id)
                        _logits = t.brain.perception_head(_out)[0, -1]
                    _conf_g = float(torch.softmax(_logits, dim=-1).max())
                    _seq = []
                    for _ in range(int(pd.get("max_length", 8))):
                        _tk = int(torch.argmax(_logits))
                        if _tk == 2:
                            break
                        _seq.append(_tk)
                        _out, _hh = t.brain.forward_hidden(
                            torch.tensor([[_tk]], dtype=torch.long, device=_dev),
                            hidden=_hh, state_id=_gen_state_id)
                        _logits = t.brain.perception_head(_out)[0, -1]
                # 【M4】デコードでは<GONE>を表示しない（採点用の別列gone=1で残す）。
                # 【M4d】<HERE>も同様に表示しない（採点用の別列here=1で残す）。
                _seq_clean = [tk for tk in _seq
                              if tk != _gone_id_v and tk != _here_id_v]
                _word_g = _pv.decode(_seq_clean) if _seq_clean else ""
                _word_h, _conf_h = "", 0.0
                _hip = getattr(t, "language_hippocampus", None)
                if _hip is not None and len(_hip) > 0:
                    if _vanish_input and _gone_id_v is not None:
                        # 【M4】海馬側は、vanished時はtokens[0]==<GONE>の記憶だけ、
                        #   通常時はtokens[0]!=<GONE>の記憶だけを候補にする（仕様書(b)）。
                        #   language_hippocampus.pyは変更禁止のため、recall_with_confと
                        #   同じコサイン最近傍探索を、候補を絞った上でここで行う
                        #   （_hippo_recall_filtered、taro_core側は一切変更しない）。
                        # 【M4d・2026-09-06】あるの印の記憶と消えたの印の記憶を混ぜない
                        #   （仕様書決定3）。want を3値にする（here_input無しなら
                        #   _here_now は常にFalse＝want="gone"|"none"の従来2値のまま）。
                        _want = "gone" if _gone_now else ("here" if _here_now else "none")
                        _toks, _simh, _str = self._hippo_recall_filtered(
                            _hip, _key.detach().cpu().numpy(),
                            want=_want, gone_id=_gone_id_v, here_id=_here_id_v)
                    else:
                        _toks, _simh, _str = _hip.recall_with_conf(_key.detach().cpu().numpy())
                    # 【M4d】海馬から拾った記憶にも<HERE>が混じりうる（あるの印の
                    #   記憶）ため、<GONE>と同様にデコード前に除く。
                    _toks_clean = [tk for tk in _toks
                                  if tk != _gone_id_v and tk != _here_id_v]
                    _word_h = _pv.decode(_toks_clean) if _toks_clean else ""
                    _conf_h = float(pd.get("hippo_gain", 1.0)) * float(_simh) * float(_str)
                if _word_h and _conf_h > _conf_g:
                    _word, sim, _src = _word_h, _conf_h, "hippo"
                else:
                    _word, sim, _src = _word_g, _conf_g, "gru"
                chunk = tuple(t.hearing.vocab.encode(_word)) if _word else None
                self._last_choice = {"gru": [_word_g, round(_conf_g, 3)],
                                     "hippo": [_word_h, round(_conf_h, 3)], "chosen": _src}
        if chunk is None:
            return
        # 【Tier3・2026-08-22】閾値はorienting_reflexの認識信号REC_THRESHOLD
        #   （e_orienting_v2.py:461＝0.80、F1-3cの保存済み画像で較正）に揃えた。
        #   「見た物と語の一致度」を扱う信号を2種類持たせない判断（設計「残る
        #   未決」節）。
        threshold = float(pd.get("threshold", 0.80))
        if sim < threshold:
            return
        # ② 注視が続いている（サッケード実行中でなく、いままさに同じ対象へ
        #   留まっている）。e_orienting_v2.py _update_habituation() が保持判定に
        #   使っているのと同じ2条件をそのまま読む（新しい仕組みは作らない）。
        holding = orienting._sacc_remaining <= 0.0 and orienting._should_hold()
        if not holding:
            return
        # ③ クールダウン（直前に喋っていない）。
        #   【Tier3・2026-08-22・文献根拠なし】同じ語を連呼し続けないための
        #   最小限の歯止め。値は仮（人間の平均発話間隔より長めに取っただけ）。
        cooldown_sec = float(pd.get("cooldown_sec", 2.0))
        now = self.now.sim_sec
        if t._last_produce_sec is not None and (now - t._last_produce_sec) < cooldown_sec:
            return

        # 【発話の動機・2026-08-31】言うかどうかの門。social未設定なら素通り
        #   ＝従来どおり必ず言う。決定は（言った/言わなかった とも）記録し、
        #   窓の結果（親の反応の有無）で学習する。
        if _sg is not None:
            if t._social_pending is not None:
                return          # 直前の決定の結果待ち
            _spoke = _sg.decide(self.env.unwrapped.np_random)
            t._social_pending = (_spoke, now)
            if not _spoke:
                return

        # ---- ここからトリガー成立：② generate() で実際に発声する -----------------
        t._last_produce_sec = now
        target_word = t.hearing.vocab.decode(list(chunk))
        target_tokens = t.produce_vocab.encode(target_word)
        self._ensure_brain_capacity()   # 【⓪】encodeで名簿が伸びたら入力口も伸ばす
        max_length = int(pd.get("max_length", 8))

        # 【F2-2・実装作業③・2026-08-23】ブローカ野で発話計画を立てる。
        #   帳面（produce_cerebellum）に無いモーラは motor=None のまま計画に入り、
        #   generate() 側で大脳皮質が自力で選ぶ（探索的な動き）。use_hear_fallback
        #   は既定Falseのまま（設計第2部決定3：正解表 vocal_tract.hear() は使わない）。
        #   設計：F/docs/設計_F2-2_見た物の名前を言う.md 第4部「③」。
        t.produce_broca.plan(
            target_word, t.produce_cerebellum, t.produce_vocal_tract,
            use_hear_fallback=False)
        plan_length = t.produce_broca.get_plan_length()
        known_moras = sum(
            1 for item in t.produce_broca.motor_buffer if item["motor"] is not None)

        # 【F2-2・実装作業③】speech_plan を渡すと generate() 内部で
        #   max_chars = min(計画長, stamina)（stamina未指定なのでmax_lengthが
        #   代わりに使われる。実装作業⑦の実測：taro_core/F にはまだ肺(stamina)の
        #   機構が移植されておらず taro に stamina属性が無いため、ここではstamina
        #   引数を渡さない＝計画が切られることはない。詳細は作業記録参照）。
        # 【F2-2・2026-08-23】発話時の探索ノイズ（NE）をどうするか。
        #   NEノイズは青斑核（locus_coeruleus.py）の仕組みで、本来は
        #   **学習のための探索**（報酬が得られていないほど色々試す）。
        #   目標Bは発音を報酬で学習していたので探索が要ったが、
        #   **F2-2は学習しない（帳面から引いて実行するだけ）ので探索は不要**。
        #
        #   さらに、ここで使われる `t.ne.get_ne_level()` は
        #   **手足の運動学習の報酬**で上下する値（trainer.py の運動探索と共用）。
        #   発話とは無関係な信号で口が震えることになる。
        #
        #   実測（2026-08-23）：ノイズを消して計画をそのまま実行すれば
        #   「わんわん」は完璧に出るのに、実走行では27回中0回しか一致せず
        #   「わおざあ」「らあぱん」等になっていた。
        #   （ノイズの"隣にずらす"実装自体にも問題がある。
        #     taro_core/src/brain/taro_brain.py の `_apply_ne_noise` の⚠を参照）
        #
        #   `produce.exploration_noise`（既定False＝ノイズ無し）で切り替える。
        #   学習を有効にする（produce.learn=true）ときは探索が要るので、
        #   そのときだけ True にすること。
        _ne = t.ne.get_ne_level() if bool(pd.get("exploration_noise", False)) else 0.0
        # 【2026-09-04・文脈の配線】gru_hippo経路と同じ理由でhidden=Noneを
        #   t._context_hidden に変更（context無効なら常にNone＝挙動不変）。
        generated, log_probs, hidden = t.brain.generate(
            hidden=t._context_hidden, max_length=max_length, eos_idx=2,
            vocal_tract=t.produce_vocal_tract, ne_level=_ne,
            cerebellum=t.produce_cerebellum, speech_plan=t.produce_broca)
        generated_word = t.produce_vocab.decode(generated)

        # 【F2-2・実装作業⑤・2026-08-23】自分の声を聞く（hidden state更新）。
        #   _apply_babble に入れたものと同じ（B原本 B/src/environment/core_b.py:
        #   610-615 self_babble() の移植）。学習はしない。次に発声するときの
        #   初期hiddenには使わない（wordモードは従来どおり毎回 hidden=None、
        #   taro_setup.py _setup_produce のコメント参照）。ここでは t._produce_hidden
        #   を更新するだけに留める（babbleモードとの一貫性のため）。
        if generated:
            full_tokens = [1] + generated + [2]
            listen_input = torch.tensor([full_tokens], device=t.brain._device())
            with torch.no_grad():
                _, h = t.brain.forward_hidden(listen_input)
            t._produce_hidden = h
            # 【文脈・2026-08-30・決定2・3】自分の発話も話者の印つきで文脈へ。
            self._context_feed("self", generated)
            # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §6】自分の発話も
            #   塊の列で塊レベルの文脈へ流す（海馬には書かない。_context_feedの
            #   speaker=="self"と同じ制約は_chunk_context_feed側で担保済み）。
            #   既定False・選んだ塊列が無い（言語海馬経路を通らなかった）ときは
            #   何もしない。
            if getattr(t, "chunk_level", False):
                _cs = getattr(self, "_last_chunk_seq_used", None)
                if _cs:
                    self._chunk_context_feed("self", _cs)
            # 【発話の動機・2026-08-31】声が出たことを世界（親）へ合図する。
            #   親側は parent_labeling.update() がこれを拾って返事を予約する。
            # 【親の言い直し・2026-09-03】合図は門（produce.social）の有無に関係なく立てる。
            #   親側が respond_to_voice=False（既定）なら合図は読まれず、挙動は従来と同じ。
            #   門の学習（応答で発話率が変わる）は言い直しとは別の機構なので分けて試す。
            self.env.unwrapped._taro_voice_signal = True
            self.env.unwrapped._taro_voice_text = generated_word   # 親の正誤判定用

        # 【F2-2・実装作業④・2026-08-23】報酬・学習は既定OFF（設計第2部決定1：
        #   「学習ではなく実行」。帳面が正解を持っているので学ぶものが無い）。
        #   produce.learn（既定False）で有効化できる形で**コードは消さずに残す**
        #   （将来、聴覚経路(逸脱その29)や automatization(逸脱その33)に手をつける
        #   ときにそのまま使える）。
        reward = None
        if bool(pd.get("learn", False)):
            # ③ 出た音 vs 言いたかった語 の一致度を報酬にする（B原本の移植、
            #   taro_core/src/brain/imitation_reward.py:出典コメント参照）。
            from imitation_reward import compute_imitation_reward, compute_alignment_credit
            reward = compute_imitation_reward(
                target_tokens, generated, vocab=t.produce_vocab,
                vocal_tract=t.produce_vocal_tract)

            # 【12ヶ月産出実験の準備①・2026-08-23】発話まるごとに単一のδだけで
            #   学習すると「4文字のどれが良くてどれが悪かったか」が太郎に分からない
            #   （探索範囲 588^4 通り）。文字ごとの一致度（compute_alignment_credit、
            #   B2-2・imitation_reward.py:出典コメント参照）を使い、良かった文字は
            #   より強く強化・目標語にない余分な文字は抑制する（探索範囲 588×4通り）。
            credits = compute_alignment_credit(
                target_tokens, generated, vocab=t.produce_vocab,
                vocal_tract=t.produce_vocal_tract)

            # ④ 学習：**運動学習(taro.learner)とは別の学習器**（taro.produce_learner）で
            #   REINFORCE（設計「⚠学習器の共有」案a。taro_setup.py _setup_produce
            #   のコメント参照＝勾配の対象が運動系と重ならない）。
            rpe = t.produce_dop.compute_rpe(reward)
            pl = t.produce_learner.learn_action(log_probs, rpe, credits=credits)
            t.produce_learner.update(0.0, pl)

        self.tick.last_produce = {
            "target_word": target_word, "sim": float(sim),
            "generated_word": generated_word,
            "choice": getattr(self, "_last_choice", None),
            "reward": float(reward) if reward is not None else None,
            "plan_length": int(plan_length), "known_moras": int(known_moras),
            # 【M4・2026-09-06】vanish_input無効なら常にgate="ok"/gone=0/attended_id=""
            #   （既存実験のCSV読み手が新列を無視すれば1ビットも変わらない）。
            # 【M4d・2026-09-06】here_input無効なら常にhere=0（_here_nowが常にFalse）。
            "gate": "ok", "gone": int(_gone_now), "here": int(_here_now),
            "attended_id": _attended_id,
            # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §6】既定"mora"
            #   （chunk_level無しでは常に"mora"＝word_production.pyのCSVに新列
            #   unitが増えるだけで、既存の全数値は1ビットも変わらない）。
            "unit": "chunk" if getattr(t, "chunk_level", False) else "mora",
            # 【言いたいことの層・2026-09-07・仕様_M6_言いたいことの層.md §3】
            #   message_level無しでは_last_messageが常にNone（gru_hippo以外の
            #   経路では属性自体が無い）＝noun/pred/rolesは常に空。
            #   word_production.pyのCSVに新列noun/predが増えるだけで、
            #   既存の全数値は1ビットも変わらない。
            "noun": (getattr(self, "_last_message", None) or {}).get("noun", ""),
            "pred": (getattr(self, "_last_message", None) or {}).get("pred", ""),
            "roles": (getattr(self, "_last_message", None) or {}).get("roles", [])}

    def _apply_babble(self, pd):
        """喃語モード（F2-1）：視覚トリガーなし・クールダウンだけで自発的に声を出す。

        設計：F/docs/設計_F2-1_喃語で口の内部モデルを作る.md 第4部「④」。
        逆引き・報酬・学習は通らない。声を出して小脳の帳面（forward_map/
        inverse_map）に書くだけ（決定：喃語フェーズは帳面を溜めるだけ）。
        """
        t = self
        cooldown_sec = float(pd.get("cooldown_sec", 2.0))
        now = self.now.sim_sec
        if t._last_produce_sec is not None and (now - t._last_produce_sec) < cooldown_sec:
            return
        t._last_produce_sec = now

        # 【F2-1・実装作業⑤】顎のサイクル数を1〜2から一様ランダムに引く
        # （設計第2部決定1）。実験ファイルの produce.jaw_cycles で候補を
        # 差し替え可能（既定[1, 2]）。
        jaw_choices = pd.get("jaw_cycles", [1, 2])
        jaw_cycles = random.choice(jaw_choices)

        generated, log_probs, hidden = t.brain.generate(
            hidden=t._produce_hidden, max_length=int(pd.get("max_length", 8)),
            eos_idx=2, vocal_tract=t.produce_vocal_tract, ne_level=t.ne.get_ne_level(),
            cerebellum=t.produce_cerebellum, jaw_cycles=int(jaw_cycles))
        generated_word = t.produce_vocab.decode(generated)

        if not generated:
            return

        # 【F2-1・実装作業⑦】自分の声を聞く（hidden state更新）。学習はしない。
        #   B原本 B/src/environment/core_b.py:610-615 self_babble() の移植。
        #   BOS(1)/EOS(2)は Vocabulary の固定インデックス（taro_core/src/senses/
        #   hearing.py）。次の喃語発声はこのhiddenを引き継ぐ（word モードは
        #   従来どおり毎回 hidden=None のまま・不変）。
        full_tokens = [1] + generated + [2]
        listen_input = torch.tensor([full_tokens], device=t.brain._device())
        with torch.no_grad():
            _, h = t.brain.forward_hidden(listen_input)
        t._produce_hidden = h

        self.tick.last_babble = {
            "generated_word": generated_word, "jaw_cycles": int(jaw_cycles),
            "length": len(generated)}

    def _visual_attention_step(self, obs):
        """太郎に1コマぶん「見る」をさせ、結果を掲示板へ置く（段A・2026-09-13）。

        【なぜ学習ループがやるのか】「見る→注意を決める→眼球へ命令」は太郎の能力
        なので、持ち主は太郎（`taro.visual_attention`）。ただし**呼ぶ位置**が
        振る舞いを決めるため（設計_太郎をCoreで完結させる.md の実測表）、
        元と同じ位置＝プラグインの on_step ループの直前で呼ぶ。
        段Bでこの手順ごと taro_core へ移す。

        中身は `run/plugins/common/object_files.py` の on_step（2026-09-13時点）から
        **計算と掲示板への書き込みだけ**をそのまま持ってきたもの。値・順・条件は
        1つも変えていない（CSV・PNGへの書き出しは道具に残す）。
        """
        va = getattr(self, "visual_attention", None)
        if va is None:
            return
        # 【段B-2c3・2026-09-13】世界も観測も、掲示板越しでなく直接受け取る。
        #   obs は呼ぶ側（run/trainer.py）が渡す。元は ctx.last["obs_out"] から
        #   取っていたが、それは同じ tick に置かれた**同じ観測**（trainer.py で
        #   `self.ctx.last = {..."obs_out": state["obs"]...}` の直後に呼ばれる）。
        _u = getattr(getattr(self, "env", None), "unwrapped", None)
        orienting = getattr(_u, "_orienting", None) if _u is not None else None

        # cam_fovy の解決（元：object_files.py の on_step。1回だけ・無ければ60度で代用）
        fovy = None
        if va.efference_copy_cfg is not None and va._ec is None:
            fovy = 60.0
            if _u is not None:
                try:
                    cid = int(_u.model.camera("eye_left").id)
                    fovy = float(_u.model.cam_fovy[cid])
                except Exception as e:
                    print("[efference] WARNING cam_fovy が読めない。60度で代用: %r" % (e,))
            else:
                print("[efference] WARNING ctx.env が無く cam_fovy を読めない。60度で代用")

        taro = self
        lexicon_obj = getattr(taro, "lexicon", None)
        hearing = getattr(taro, "hearing", None)
        vocab_obj = getattr(hearing, "vocab", None) if hearing is not None else None

        t = float(_u.data.time) if _u is not None else 0.0

        result = va.step(
            obs=obs, orienting=orienting, t=t, step=self.now.step, fovy=fovy,
            surprise_trace=getattr(self.tick, "surprise_trace", None),
            vision_vec=getattr(self.tick, "last_vision_vec", None),
            parent_target=getattr(self.tick, "last_parent_utterance", None),
            lexicon=lexicon_obj, vocab=vocab_obj)
        va.last_result = result

        # 段0は毎tick動く（検出コマでなくても）。efference_copy=None なら触らない
        if va.efference_copy_cfg is not None:
            self.tick.efference = result["efference"]
        if not result["detected"]:
            return

        self.tick.priority_map_result = result["priority_map_result"]
        self.tick.priority_map_result_t = result["priority_map_result_t"]
        if va._spri is not None:
            self.tick.goal_point = result["goal_point"]
            self.tick.salience_map = result["salience_map"]
            self.tick.attention_point = result["attention_point"]
        # 切り替えが一度も起きていなければ触らない（既定の -1e9 を壊さないため）
        if result["attention_switch_t"] is not None:
            self.tick.attention_switch_t = result["attention_switch_t"]

        if va.attend:
            self.tick.attended_object = result["attended_object"]
            # 【M4c・台帳その45】消失に気づいた瞬間だけ世界（親）へ合図する。
            #   人間の親は子の頭の中を直接読めないので、これは人間模倣からの逸脱。
            #   既定OFF。書く場所も値もタイミングも移設前のまま（別件・触らない）。
            if result["noticed_gone"] and _u is not None:
                _u._taro_noticed_gone_time = t

    @staticmethod
    def _hippo_recall_filtered(hip, key_vis, want, gone_id, here_id=None):
        """language_hippocampus.LanguageHippocampus.recall_with_confと同じ
        コサイン最近傍探索を、候補をtokens[0]の種類で絞った上で行う
        （2026-09-06・仕様_M4_消えた物について「○○ないね」と言う(b)、
        2026-09-06・仕様_M4d_あるの印で3値に拡張）。

        taro_core/src/brain/language_hippocampus/language_hippocampus.pyは
        変更禁止（recall_with_confに絞り込み引数を足せない）ため、公開属性
        hip.episodes（list[dict]）を読むだけでここに複製する。
        アルゴリズム自体はrecall_with_confの実装を1文字も変えず、候補を
        「tokens[0]の種類」で絞る点のみ追加。

        want: "gone"|"here"|"none"。tokens[0]がgone_idなら"gone"、here_idなら
        "here"、どちらでもなければ"none"として、wantと一致するエピソードだけを
        候補にする（あるの印の記憶と消えたの印の記憶を混ぜない・仕様書決定3）。
        here_id=None（produce.here_input無し）のときはgone_idとの一致だけを
        見る＝従来の2値と同じ結果になる。

        候補が無ければ([], 0.0, 0.0)。
        """
        import numpy as np

        def _cat(tok0):
            if gone_id is not None and tok0 == gone_id:
                return "gone"
            if here_id is not None and tok0 == here_id:
                return "here"
            return "none"

        eps = [ep for ep in hip.episodes
               if bool(ep["tokens"]) and _cat(ep["tokens"][0]) == want]
        if not eps:
            return [], 0.0, 0.0
        q = np.asarray(key_vis, dtype=np.float32)
        qn = np.linalg.norm(q) + 1e-8
        best, best_sim = None, -1e9
        for ep in eps:
            k = ep["key_vis"]
            sim = float(np.dot(q, k) / (qn * (np.linalg.norm(k) + 1e-8)))
            if sim > best_sim:
                best_sim, best = sim, ep
        return list(best["tokens"]), best_sim, float(best["strength"])

    # ------------------------------------------------------------ 睡眠（言語）
    def _consolidate_language(self):
        """言語海馬のリプレイ（段階1・2026-09-01・設計_言語海馬と睡眠リプレイ.md）。

        海馬が無ければ何もしない＝既定OFF（produce.hippocampus無し）では
        1行も実行されず挙動は不変。乱数は torch の既定Generatorのみを使う
        （env.np_random は触らない＝決定性を壊さないため。設計「決定性の注意」）。
        既存の聞く学習（_context_feed内、368〜385行相当）と同一の損失を、
        エピソード単位でreplay_passes周まわす。lrは睡眠中だけsleep_lrに
        差し替え、終わったら必ず元へ戻す。
        """
        t = self
        hippo = getattr(t, "language_hippocampus", None)
        if hippo is None or len(hippo) == 0:
            return
        # 【塊レベル層・2026-09-07・仕様_M5_塊レベル層.md §5】chunk_level真なら
        #   エピソードは塊の列なので、鍛えるのは塊GRU（_chunk_optimizer）。
        #   音レベルGRUはここでは再生しない（聞く学習・切れ目の発見は_context_feed
        #   経由で引き続き走る）。偽（既定）ならこれまでどおり音レベルGRUを鍛える。
        _chunk_level = bool(getattr(t, "chunk_level", False))
        brain = t.chunk_brain if _chunk_level else t.brain
        opt = (getattr(t, "_chunk_optimizer", None) if _chunk_level
               else getattr(t, "_listen_optimizer", None))
        if brain is None or opt is None:
            return
        import torch.nn.functional as F
        dev = brain._device()
        orig_lrs = [g["lr"] for g in opt.param_groups]
        for g in opt.param_groups:
            g["lr"] = hippo.sleep_lr
        try:
            for _ in range(hippo.replay_passes):
                eps = hippo.sample(torch.default_generator, len(hippo))
                perm = torch.randperm(len(eps)).tolist()
                eps = [eps[i] for i in perm]
                for ep in eps:
                    # 【仕様書§5】塊の列は書き込み時にEOSを省いているので
                    #   （_chunk_context_feedのwrite呼び出し参照）、ここで戻す。
                    #   音レベル（既定）はEOSの有無を書き込み時の値のまま使う
                    #   （従来どおり）。
                    if _chunk_level:
                        ids = [ep["speaker"]] + list(ep["tokens"]) + [2]
                    else:
                        ids = [ep["speaker"]] + list(ep["tokens"])
                    if len(ids) < 2:
                        continue
                    xin = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
                    tgt = torch.tensor([ids[1:]], dtype=torch.long, device=dev)
                    pfx = torch.tensor(ep["key_vis"], dtype=torch.float32, device=dev)
                    # 【M4e・2026-09-07・仕様_M4e_状態の線と驚きの書き込み】
                    #   tokens[0]がgone/hereの印なら、再生でも同じ状態の線を
                    #   立てたまま学習する。既定False（state_channel無し）ではNone。
                    #   塊レベルではgone/hereのidは chunk_vocab.specials 側の空間
                    #   （音の名簿の_gone_id/_here_idとは別空間）を見る。
                    _sc_replay = bool((self.cfg.produce or {}).get("state_channel", False))
                    _st_replay = None
                    if _sc_replay and ep["tokens"]:
                        _tok0 = ep["tokens"][0]
                        if _chunk_level:
                            _gone_ref = t.chunk_vocab.specials.get("gone")
                            _here_ref = t.chunk_vocab.specials.get("here")
                        else:
                            _gone_ref = getattr(t, "_gone_id", None)
                            _here_ref = getattr(t, "_here_id", None)
                        if _tok0 == _gone_ref:
                            _st_replay = 2
                        elif _tok0 == _here_ref:
                            _st_replay = 1
                    out, _ = brain.forward_hidden(xin, hidden=None, prefix_vec=pfx,
                                                  state_id=_st_replay)
                    out = out[:, 1:, :]      # 先頭＝視覚トークン位置は損失に使わない
                    loss = F.cross_entropy(brain.perception_head(out)[0], tgt[0])
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (p_ for grp in opt.param_groups for p_ in grp["params"]),
                        t._listen_grad_clip)
                    opt.step()
        finally:
            for g, lr in zip(opt.param_groups, orig_lrs):
                g["lr"] = lr
        hippo.decay()

    # ------------------------------------------------------------ 睡眠
    def consolidate(self, n_batches=200, bs=128):
        """睡眠中の記憶定着：貯めた経験をバッチで再生し、予測経路を復習で固める。"""
        t = self
        self._consolidate_language()
        eps = t.hippo.replay()
        N = len(eps)
        if N < bs:
            return
        SV = torch.stack([e[0] for e in eps]); PA = torch.stack([e[1] for e in eps])
        AA = torch.stack([e[2] for e in eps]); CF = torch.stack([e[3] for e in eps])
        CLP = torch.stack([e[4] for e in eps]); NLP = torch.stack([e[5] for e in eps])
        H = torch.cat([e[6] for e in eps], dim=1)          # (layers, N, hidden)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (bs,))
            hb = H[:, idx].contiguous()
            emb = t.emb_proj(torch.cat([SV[idx], PA[idx]], dim=-1)).unsqueeze(1)
            out, _ = t.brain.motor_gru(emb, hb)
            z, kl, rc = t.brain.pc_latent.infer(hb[-1], out[:, 0], CF[idx])
            pred = CLP[idx] + t.nat_head(torch.cat([z, AA[idx]], dim=-1))
            loss = t.block_pe(pred, NLP[idx]) + kl + rc       # 学習ループと同じ基準
            t.learner.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(t.learner.brain.parameters(),
                                           t.learner.grad_clip)
            t.learner.optimizer.step()

