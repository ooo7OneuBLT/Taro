"""学習の様子の絵（HTML）を、記録の区切りごとに自動で作り直す。

【なぜプラグインにしたか、2026-07-30】ユーザーの要望：

> 視覚化の部分も run システムに組み込んで完全に自動化できないかな？

図を作るのは「外から見る道具」なので、測る道具と同じ枠（プラグイン）に収める。
学習ループは図のことを知らないままでよい（責務を混ぜない）。

注意：ctx を**読むだけ**。太郎も環境も変えない。
  実際に読むのは CSV（同じフォルダに書かれたもの）で、絵を作るのは
  `run/tools/dashboard.py`。ここは**呼ぶだけ**。

【自動でONになる】`run.csv` を指定した学習では、実験ファイルに書かなくても
`run/main.py` が自動で足す。切りたいときは実験ファイルに
    "plugins": {"dashboard": false}
と明示する。

【ブラウザが自動で開く、2026-07-31】学習を始めると絵を1枚作ってブラウザで開く。
絵は500回ごとに作り直され、HTML 側に `<meta http-equiv="refresh" content="30">`
が入っているので**開きっぱなしで進み具合が見える**。
⇒ 別プロセスで `run.tools.dashboard --watch` を立てる必要はない。

注意：同じフォルダへ**複数のシードを並列で流す**とき、全部がブラウザを開くと
  タブが増えて邪魔になる。そこで「直近10分に誰かが開いていたら開かない」印
  （`.dashboard_opened`）をフォルダに置いて、1回だけ開くようにしている。

実験ファイルでの書き方（明示する場合）:
    "plugins": {"dashboard": true}
    "plugins": {"dashboard": {"dir": "E/logs/別のフォルダ", "title": "見出し"}}
    "plugins": {"dashboard": {"open": false}}     ← ブラウザを開かせない
"""
import os
import time

from run.plugins.base import Plugin

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))


class Dashboard(Plugin):
    name = "dashboard"

    def setup(self, ctx):
        # 出力先は「CSVを置いたフォルダ」が既定（同じ実験の仲間が並ぶ場所）
        """ctx を受け取り、出力先ディレクトリとタイトルを設定または実験名から決めてダッシュボードHTMLを作成し、必要ならブラウザで一度だけ開く。戻り値は無い。"""
        d = self.config.get("dir")
        if not d:
            csv_path = (ctx.spec.get("run") or {}).get("csv")
            d = os.path.dirname(csv_path) if csv_path else None
        self.dir = d
        # 見出しは実験ファイルの name を既定にする（どの実験の絵か分かるように）
        self.title = self.config.get("title") or ctx.spec.get("name")
        self.made = None
        # 【2026-08-24・ユーザーの指示で既定を True→False に変更】
        #   学習を始めるたびにブラウザが勝手に開くのをやめる。
        #   HTMLの生成は今までどおり続ける（ファイルは作られる）ので、
        #   見たいときは出力先のHTMLを自分で開けばよい。
        #   実験ファイルの plugins.dashboard.open を true にすれば従来どおり開く。
        #   （このダッシュボードは将来アプリ化する予定＝ブラウザ版はもう使わない）
        self.open_browser = bool(self.config.get("open", False))
        if not self.dir:
            print("注意[dashboard] 出力先が決まらないので絵を作りません"
                  "（run.csv か plugins.dashboard.dir を指定してください）", flush=True)
            return
        # 【2026-09-14・ステップ1】run.meta.json は本体（run/main.py の
        #   _write_run_meta）が書くようになった。ここで書くと、姿勢・setup・
        #   環境変数を含まない古い形で**上書きしてしまう**ので呼ばない。
        #   _write_meta 自体は下に残す（古いログを読む道具の参照先のため）。
        # 学習の開始時に1枚作ってブラウザで開く（あとは30秒ごとに自動で読み直される）
        self._make()
        self._open_once()

    def _write_meta(self, ctx):
        """「どんな条件で回したか」を CSV の隣に書く。

        【なぜ要るか、2026-07-31】絵に条件（シーン名・体の設定）を出したかったが、
        読める場所が**画面に出た文字を保存したログ**しかなかった。
        ⇒ `> seed0.log` のようにリダイレクトした人だけが条件を見られる、という
        不安定な作りだった。実験ファイルの中身をそのまま横に置いて解決する。
        """
        csv_path = (ctx.spec.get("run") or {}).get("csv")
        if not csv_path:
            return
        import json
        p = csv_path if os.path.isabs(csv_path) else os.path.join(_ROOT, csv_path)
        try:
            with open(p[:-4] + ".meta.json", "w", encoding="utf-8") as fp:
                json.dump({"name": ctx.spec.get("name"),
                           "scene": ctx.spec.get("scene"),
                           "taro": ctx.spec.get("taro"),
                           "run": ctx.spec.get("run"),
                           "tools": sorted(getattr(ctx, "plugin_names", []) or []),
                           "scene_note": (ctx.scene or {}).get("note"),
                           "world": (ctx.scene or {}).get("world"),
                           "body": (ctx.scene or {}).get("body")},
                          fp, ensure_ascii=False, indent=2)
        except Exception as e:      # noqa: BLE001
            print(f"注意[dashboard] 条件を書けませんでした: {type(e).__name__}: {e}",
                  flush=True)

    def _open_once(self):
        """ブラウザで開く。ただし同じフォルダで直近10分に誰かが開いていたら開かない。

        【なぜ印を置くか】2シードを並列で流すと**2つのプロセスがそれぞれ開く**。
        3本流せば3つ開く。見たいのは1枚なので、先に開いた者だけが開く。
        """
        if not (self.open_browser and self.made):
            return
        mark = os.path.join(self.dir, ".dashboard_opened")
        try:
            if os.path.exists(mark) and (time.time() - os.path.getmtime(mark)) < 600:
                return                      # 並列の相方が既に開いている
            with open(mark, "w", encoding="utf-8") as fp:
                fp.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            import webbrowser
            url = "file:///" + os.path.abspath(self.made).replace("\\", "/")
            webbrowser.open(url)
            print(f"[dashboard] ブラウザで開きました（30秒ごとに自動で最新になります）\n"
                  f"            {self.made}", flush=True)
        except Exception as e:              # noqa: BLE001
            # 注意：開けなくても学習は続ける（画面が無い環境・既定ブラウザが無い等）
            print(f"注意[dashboard] ブラウザを開けませんでした: {type(e).__name__}: {e}\n"
                  f"            手で開いてください: {self.made}", flush=True)

    def _make(self):
        if not self.dir:
            return
        from run.tools import dashboard as dash
        try:
            self.made = dash.make(self.dir, self.title)
        except Exception as e:      # noqa: BLE001
            # 注意：握りつぶすが黙らない。絵作りの失敗で**学習を落とさない**のが大事。
            #   （2時間の学習が図の不具合で消えるのは本末転倒）
            print(f"注意[dashboard] 絵を作れませんでした: {type(e).__name__}: {e}",
                  flush=True)

    def on_checkpoint(self, ctx):
        # 注意：CSV はこのあと書かれるので、ここで作る絵は**1つ前**の区切りまでを映す。
        #   （記録の順番＝測る→CSVに書く→次の学習。1区切り遅れるだけなので許容）
        """ctx を受け取り、ダッシュボードHTMLを作り直す。戻り値は無い。"""
        self._make()

    def report(self, ctx):
        """ctx を受け取り、ダッシュボードHTMLを最後にもう一度作り直し、作成できていれば出力パスを含む辞書を、できていなければNoneを返す。"""
        self._make()            # 最後にもう一度（最新の行まで入った絵にする）
        if not self.made:
            return None
        return {"絵": os.path.relpath(self.made, os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, os.pardir)))}
