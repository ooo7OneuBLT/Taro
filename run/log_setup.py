# -*- coding: utf-8 -*-
"""ログの置き場所と等級をここ1か所で決める（2026-09-12）。

【なぜ作ったか】ユーザー指摘「今ってエラーログは一つにまとめてファイル化されてる？
  ほかのログと区別して」「実行中はエラーログを絶対読むようにしよう」。
  実測（2026-09-12）：エラー専用のログは**無かった**。`print` が1609個あり、
  等級が無いのでエラーも進捗も診断も同じ流れに混ざっていた。黙って見逃す
  `except Exception: pass` が30か所あった。`logging` は一度も使われていなかった。

【何を解決し、何を解決しないか】ここは正直に書く。
  解決する：落ちたときの痕跡／黙って飲み込まれていた例外／「走行中に出た警告を
            あとで探せない」
  解決しない：**計算を間違えた**こと。2026-09-12 に起きた5件のミス（境界を0.295に
            置いた・全体の中央値で測った・二重importで設定が効いていなかった・
            期待値の式にFRACを掛けた）は**どれも例外を投げていない**。
            これを捕まえるのは「走る前に何が出るはずかを宣言して、合わなければ落ちる」
            仕掛け（`run/tools/smoke.py` の expect、掃引の自己検査）と、
            下の `warn_if` のような**場面ごとの見張り**。ログは土管にすぎない。

【3つのファイル】走行ごとのフォルダ（out_dir）の下に自動で作る
    エラー.log    WARNING 以上だけ。**これを読めば異常が全部分かる**ことを保証する
    走行.log      DEBUG 以上。あとで追いかけるとき用
                  注意：`taro.未実装` のロガーだけは除く（未実装.log にしか残らない）
    未実装.log    「ここは後で作る」の置き場。エラーとは混ぜない
                  （logger 名を `taro.未実装` にしたものが流れる）

【使い方】
    # 走行の入口で1回
    from run.log_setup import setup_logging, log_tail
    paths = setup_logging(out_dir)      # out_dir=None なら F/logs/_ログ/<日時>/
    ...
    log_tail(paths)                     # 走行の最後に。エラー.logの中身を画面に出す

    # 各モジュールで
    from run.log_setup import get_logger
    log = get_logger(__name__)
    log.debug("...")                     # 走行.log だけ
    log.info("...")                      # 画面＋走行.log
    log.warning("...")                   # ＋エラー.log
    log.error("...", exc_info=True)      # ＋エラー.log（traceback付き）

    # 例外を飲み込んでいた所（`except Exception: pass`）はこれに置き換える
    from run.log_setup import swallowed
    try: ...
    except Exception: swallowed(log, "視覚キャッシュの更新")

    # 場面ごとの見張り（**ここが一番効く**。条件が崩れたら警告を残す）
    from run.log_setup import warn_if
    warn_if(log, n_used < 30, "使えた発が %d 本しかない（30本未満）", n_used)

【注意】`logging` は root に handler を付ける。既に付いていれば何もしない（二重に
  書かない）。太郎は同じモジュールが2つの名前で import されることがある
  （`midbrain.orienting` と `taro_core.src.brain.midbrain.orienting`）が、
  どちらも root まで伝播するのでログは出る。名前が2通り見えるのは正常。
"""
import logging
import os
import sys
import time
import traceback

# 等級の呼び名（日本語で画面に出す。英語の DEBUG/INFO/... を覚えなくてよいように）
_LEVEL_JA = {
    logging.DEBUG: "診断",
    logging.INFO: "進捗",
    logging.WARNING: "警告",
    logging.ERROR: "エラー",
    logging.CRITICAL: "致命",
}

_STATE = {"paths": None, "counts": None}


class _JaFormatter(logging.Formatter):
    """等級を日本語にして、行を短く保つ。"""

    def format(self, record):
        """ログレコードを受け取り、等級（levelno）を日本語ラベルに変換してレコードに付け、親クラスで整形した文字列を返す。"""
        record.等級 = _LEVEL_JA.get(record.levelno, record.levelname)
        return super().format(record)


class _ShortFormatter(_JaFormatter):
    """エラー.log 用。1件が長すぎると一覧性が死ぬので切る（2026-09-12 の実測で必要に）。

    実測：「シーンが見つからない」の例外はシーン一覧を全部並べるので1件40KB。
    これが入るとエラー.logは1件で埋まり、「読めば異常が全部分かる」が嘘になる。
    全文は常に 走行.log に残る（そちらは切らない）。
    """

    LIMIT = 1800

    def format(self, record):
        """ログレコードを受け取り、親クラスで整形した文字列を作る。長さが上限（LIMIT）を超えていればそこで切り、省略した旨を付け足して返す。"""
        s = super().format(record)
        if len(s) <= self.LIMIT:
            return s
        return (s[:self.LIMIT] +
                "\n    …（ここで切りました。全文は同じフォルダの 走行.log）")


class _Counter(logging.Handler):
    """警告・エラーが何件出たかを数えるだけ（走行の最後に出すため）。"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.n = {"警告": 0, "エラー": 0}

    def emit(self, record):
        """ログレコードを受け取り、レベルがエラー以上かどうかで「警告」または「エラー」の件数のどちらかを1増やす。戻り値は無い。"""
        key = "エラー" if record.levelno >= logging.ERROR else "警告"
        self.n[key] += 1


class _MikanFilter(logging.Filter):
    """`taro.未実装` だけを通す／落とす。"""

    def __init__(self, keep):
        super().__init__()
        self.keep = keep

    def filter(self, record):
        """ログレコードを受け取り、ロガー名が taro.未実装 で始まるかを判定する。self.keep が真ならその判定を、偽なら反転した値を、レコードを通すかどうかの真偽値として返す。
        """
        is_mikan = record.name.startswith("taro.未実装")
        return is_mikan if self.keep else not is_mikan


def get_logger(name):
    """モジュールごとのログ口。`__name__` を渡す。"""
    return logging.getLogger(name)


def mikan_logger(name):
    """「ここは後で作る」用のログ口。未実装.log にだけ残り、エラーとは混ざらない。"""
    return logging.getLogger("taro.未実装." + str(name))


def setup_logging(out_dir=None, *, console_level=logging.INFO, quiet=False):
    """走行の入口で1回だけ呼ぶ。2回目以降は何もせず同じパスを返す（冪等）。

    戻り値は {"エラー": path, "走行": path, "未実装": path, "dir": out_dir}。
    """
    if _STATE["paths"] is not None:
        return _STATE["paths"]

    if not out_dir:
        out_dir = os.path.join("F", "logs", "_ログ", time.strftime("%Y-%m-%d_%H%M%S"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "dir": out_dir,
        "エラー": os.path.join(out_dir, "エラー.log"),
        "走行": os.path.join(out_dir, "走行.log"),
        "未実装": os.path.join(out_dir, "未実装.log"),
    }

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # 既に誰かが handler を付けていたら、そこへ足すだけで壊さない
    for h in list(root.handlers):
        if getattr(h, "_taro", False):
            return _STATE["paths"] or paths

    # Windows の既定文字コード（cp932）だと日本語が化けるので画面側を直す。
    # **stderr も必ず直す**：log_tail は異常時に stderr へ出すので、
    # ここを忘れると警告の本文が読めない文字列になる（2026-09-12に実際そうなった）
    for _stream in ("stdout", "stderr"):
        try:
            getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    long_fmt = _JaFormatter("%(asctime)s %(等級)s [%(name)s] %(message)s",
                            datefmt="%H:%M:%S")
    short_fmt = _JaFormatter("%(等級)s [%(name)s] %(message)s")

    def _fh(path, level, fmt, filt=None):
        h = logging.FileHandler(path, mode="w", encoding="utf-8")
        h.setLevel(level)
        h.setFormatter(fmt)
        if filt is not None:
            h.addFilter(filt)
        h._taro = True
        root.addHandler(h)
        return h

    short_file_fmt = _ShortFormatter("%(asctime)s %(等級)s [%(name)s] %(message)s",
                                     datefmt="%H:%M:%S")
    _fh(paths["エラー"], logging.WARNING, short_file_fmt, _MikanFilter(keep=False))
    _fh(paths["走行"], logging.DEBUG, long_fmt, _MikanFilter(keep=False))
    _fh(paths["未実装"], logging.DEBUG, long_fmt, _MikanFilter(keep=True))

    if not quiet:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(short_fmt)
        ch.addFilter(_MikanFilter(keep=False))
        ch._taro = True
        root.addHandler(ch)

    counter = _Counter()
    counter._taro = True
    root.addHandler(counter)
    _STATE["counts"] = counter
    _STATE["paths"] = paths

    _install_excepthook()
    logging.getLogger("taro.log").debug("ログの置き場所: %s", out_dir)
    return paths


def log_crash(exc):
    """落ちた例外を エラー.log に記録する。同じ例外は二度書かない。

    【2026-09-12・これが無いと報告が嘘になる】`sys.excepthook` は
      `try/finally` の finally より**後**に発火する。走行の最後に呼ぶ `log_tail` を
      finally に置くと、まだ例外が記録されていないので「エラー.log：空」と
      表示してしまう（実際に落ちた走行でそう出たのを確認した）。
      ⇒ 落ちた側でも明示的にここを呼ぶ。excepthook は取りこぼし用の保険。
    """
    if exc is None or getattr(exc, "_taro_logged", False):
        return False
    try:
        exc._taro_logged = True
    except Exception:
        pass
    # SystemExit は「バグで落ちた」ではなく「点検が止めた」＝読ませたいのは本文だけ。
    # traceback を付けると本文が流れて読めなくなるので分ける（2026-09-12）。
    if isinstance(exc, SystemExit):
        if exc.code in (0, None):
            return False
        logging.getLogger("taro.中止").error("走行が中止されました:\n%s", exc)
        return True
    logging.getLogger("taro.落ちた").critical(
        "走行が例外で終了しました: %s: %s", type(exc).__name__, exc,
        exc_info=(type(exc), exc, exc.__traceback__))
    return True


def _install_excepthook():
    """落ちたときも traceback がエラー.log に残るようにする。

    これが無いと「落ちた走行のエラー.logが空」になり、
    「エラー.logを読めば異常が分かる」という約束が嘘になる。
    """
    prev = sys.excepthook

    def hook(exc_type, exc, tb):
        """例外の型・値・トレースバック（exc_type, exc, tb）を受け取る。KeyboardInterrupt でなければ log_crash を呼んで記録し、そのあと元の excepthook を同じ引数で呼ぶ。戻り値は無い。
        """
        if not issubclass(exc_type, KeyboardInterrupt):
            log_crash(exc)
        prev(exc_type, exc, tb)

    if not getattr(sys.excepthook, "_taro", False):
        hook._taro = True
        sys.excepthook = hook


def swallowed(log, 場面, level=logging.WARNING):
    """`except Exception: pass` の置き換え。黙って消さず、traceback を残して続行する。

    使い方：
        try:
            ...
        except Exception:
            swallowed(log, "視覚キャッシュの更新")
    """
    log.log(level, "%s で例外が出ましたが、続行しました", 場面, exc_info=True)


def warn_if(log, 条件, fmt, *args):
    """条件が真なら警告を残す。**2026-09-12 の教訓からいえば、ここが一番効く。**

    例外を投げないミス（間違った統計を取った・設定が効いていなかった・
    期待値の式が違った）は、このような「場面ごとの見張り」でしか捕まらない。
    測定スクリプトを書くときは、**結果を出す前に**これを数行置く。
    """
    if 条件:
        log.warning(fmt, *args)
    return bool(条件)


def counts():
    """{"警告": n, "エラー": n}。まだ setup していなければ 0。"""
    c = _STATE["counts"]
    return dict(c.n) if c is not None else {"警告": 0, "エラー": 0}


def paths():
    """引数は無い。setup_logging が作ったログ置き場のパス辞書を返す。まだ setup_logging を呼んでいなければ None を返す。"""
    return _STATE["paths"]


def log_tail(p=None, *, max_lines=40):
    """走行の**最後**に呼ぶ。エラー.logの中身を画面の一番下に出す。

    「絶対に読む」を注意力ではなく**位置**で守らせる仕掛け。画面の最後に出ていれば、
    次に読むのは必ずこれになる。
    """
    p = p or _STATE["paths"]
    if not p:
        return 0
    n = counts()
    total = n["警告"] + n["エラー"]
    if total == 0:
        print("-" * 74)
        print("  エラー.log：空（警告0・エラー0） → %s" % p["エラー"])
        return 0
    # 【2026-09-12・実測で必要に】異常があるときは **stderr** に出す。
    #   stdout に出すと、落ちたときに Python が stderr へ吐く traceback より
    #   前に流れてしまい、「画面の一番下に出る」という仕掛けが壊れる
    #   （実際に壊れたのを確認した）。同じ流れに出せば順序が保証される。
    def out(s):
        """文字列 s を受け取り、標準エラー出力へ即座に（flush して）書き出す。戻り値は無い。"""
        print(s, file=sys.stderr, flush=True)

    out("-" * 74)
    out("  エラー.log：警告 %d 件 / エラー %d 件  ← 下に中身を出します" %
        (n["警告"], n["エラー"]))
    out("  %s" % p["エラー"])
    try:
        with open(p["エラー"], encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return total
    if len(lines) > max_lines:
        out("  （%d行のうち最後の%d行）" % (len(lines), max_lines))
        lines = lines[-max_lines:]
    for ln in lines:
        out("  | " + (ln if len(ln) <= 240 else ln[:240] + " …（略）"))
    out("-" * 74)
    return total
