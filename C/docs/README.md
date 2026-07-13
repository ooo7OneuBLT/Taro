# MIMo統合 概要

太郎（[Taro](https://github.com/ooo7OneuBLT/Taro)）に「身体」を与えるため、MIMo（乳児身体シミュレータ, [trieschlab/MIMo](https://github.com/trieschlab/MIMo), MIT License）を統合する検証プロジェクト。

- 構成：環境＝MIMo（視覚・触覚・固有感覚・前庭覚・物理身体）＋親sim＋太郎の内受容/発声モジュール。エージェント＝太郎の脳（GRU＋本能）をほぼそのまま流用
- `MIMo/`（本体のclone）は`.gitignore`対象。別途 `git clone --branch v2.0.0 https://github.com/trieschlab/MIMo.git MIMo` で取得する
- 進捗・詰まりと解消・実測データは[研究日誌.md](MIMo研究日誌.md)に記録
