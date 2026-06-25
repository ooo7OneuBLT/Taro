"""
成長アーカイブ — スナップショット保存とフォーク

太郎の脳の状態をある時点で凍結保存し、
そこから枝分かれ（フォーク）して別条件の検証ができる。
"""

import os
import json
import torch
from datetime import datetime


class Archive:

    def __init__(self, archive_dir="archive"):
        self.archive_dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)

    def save_snapshot(self, brain, vocab, dopamine, sim_clock, config, tag="auto"):
        """
        太郎の脳を丸ごとスナップショット保存する。

        tag: スナップショットの名前（例: "day1", "first_echo"）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{tag}_{timestamp}"
        path = os.path.join(self.archive_dir, name)
        os.makedirs(path, exist_ok=True)

        torch.save(brain.state_dict(), os.path.join(path, "brain.pt"))
        torch.save({
            "char2idx": vocab.char2idx,
            "idx2char": vocab.idx2char,
            "size": vocab.size,
        }, os.path.join(path, "vocab.pt"))

        meta = {
            "tag": tag,
            "timestamp": timestamp,
            "sim_clock": sim_clock.state_dict(),
            "dopamine_baseline": dopamine.get_baseline(),
            "temperature": brain.temperature,
            "config": config,
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return path

    def load_snapshot(self, path, brain, vocab, dopamine, sim_clock):
        """
        スナップショットから太郎を復元する（フォークの起点として使える）。
        """
        brain_state = torch.load(os.path.join(path, "brain.pt"),
                                 map_location="cpu", weights_only=True)
        vocab_state = torch.load(os.path.join(path, "vocab.pt"),
                                 map_location="cpu", weights_only=True)

        vocab.char2idx = vocab_state["char2idx"]
        vocab.idx2char = vocab_state["idx2char"]
        vocab.size = vocab_state["size"]

        brain.resize_embedding(vocab.size)
        brain.load_state_dict(brain_state)

        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        sim_clock.load_state_dict(meta["sim_clock"])
        dopamine.baseline = meta["dopamine_baseline"]
        brain.temperature = meta["temperature"]

        return meta

    def list_snapshots(self):
        """保存済みスナップショットの一覧を返す。"""
        snapshots = []
        if not os.path.exists(self.archive_dir):
            return snapshots
        for name in sorted(os.listdir(self.archive_dir)):
            meta_path = os.path.join(self.archive_dir, name, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                snapshots.append({
                    "name": name,
                    "path": os.path.join(self.archive_dir, name),
                    "tag": meta.get("tag", ""),
                    "turn": meta.get("sim_clock", {}).get("total_turns", 0),
                    "age": meta.get("sim_clock", {}).get("total_seconds", 0),
                })
        return snapshots
