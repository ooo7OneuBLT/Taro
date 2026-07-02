"""
トレースログ — replayViewer用に「いつ・どの部品が発火し・情報がどう流れ・
数値がどうだったか」を1行1イベントの jsonl で書き出す。

既存のログ（turns/events/babble.jsonl）とは別ファイル（trace.jsonl）。
オプトイン（trace_pathを指定したときだけ有効）なので、通常の実行や既存の
分析パイプラインには一切影響しない。記録するのは小さなスカラーと部品IDだけ
（重いテンソルは保存しない）＝性能への影響は軽微。
"""
import json


class TraceLogger:
    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")
        self.count = 0

    def write_event(self, rec):
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
