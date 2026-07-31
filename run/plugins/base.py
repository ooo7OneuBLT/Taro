"""プラグインの規約。測る道具・見る道具はすべてこの形にする。

【なぜ、2026-07-30】実験スクリプトが118本あり、各々が環境を組んで各々が測っていた。
そのため「学習は関節モード、測定は筋肉モード」という**別の体で動いていた**事故が
起きた（`E/docs/実行基盤_設計.md`）。
⇒ 環境を作るのは1箇所、測るのは差分（プラグイン）だけ、という形にする。

【約束】
  ・プラグインは ctx を**読むだけ**。太郎も環境も変えない
    （測る道具が対象を変えたら測定にならない）
  ・太郎を変える設定は実験ファイルの `taro` 欄が担う
  ・4つのうち必要なものだけ書けばよい（書かないものは何もしない）

使い方（プラグインを書く側）:
    from run.plugins.base import Plugin

    class MyPlugin(Plugin):
        name = "my_plugin"
        def setup(self, ctx): ...
        def on_step(self, ctx): ...
        def on_checkpoint(self, ctx): ...
        def report(self, ctx): return {"値": 1.0}
"""


class Plugin:
    """測る／見る道具の共通の形。"""

    #: ログや報告に出る名前。実験ファイルのキーと同じにする
    name = "plugin"

    def __init__(self, config=None):
        """config: 実験ファイルでこのプラグインに与えた値。
        `true` だけを書いたときは {} が入る。
        """
        self.config = config if isinstance(config, dict) else {}

    # --- ここから下は、必要なものだけ書けばよい ---

    def setup(self, ctx):
        """環境ができた直後に1回。センサの位置を覚えるなど。"""

    def on_step(self, ctx):
        """毎ステップ。重い処理は書かない（学習が遅くなる）。"""

    def on_body_change(self, ctx):
        """体を作り直した直後（体を育てる実験）。model/data が別物になっている。

        注意：setup で覚えた geom / body の id は**作り直しで変わりうる**。
          ここで引き直す。累積した測定値は**消さない**（消すと推移が読めない）。
        """

    def on_checkpoint(self, ctx):
        """記録の区切りごと。評価など重い測定はここで。"""

    def metrics(self, ctx):
        """いまの値を辞書で返す（記録の区切りごとに呼ばれる）。

        【なぜ要るか、2026-07-30】以前は各プラグインが `ctx.log(行)` を自分で呼んで
        いたので、複数の道具を使うと**CSVの行が道具ごとに分裂**していた。
        ⇒ 学習ループが全プラグインの `metrics` を集めて**1行**にする。
          これで「どの道具の値も同じ行に並ぶ」＝グラフを描く側は列を決め打ちしなくてよい。

        注意：名前は他の道具とぶつからないようにする（自分の name を頭に付けるのが安全）。
        注意：数値（int/float）を返す。文字列は CSV に入るが図には描けない。
        """
        return None

    def report(self, ctx):
        """最後に1回。まとめの辞書を返す（None でもよい）。"""
        return None

    def line(self, ctx):
        """1行のログに混ぜる短い文字列（None なら混ぜない）。"""
        return None
