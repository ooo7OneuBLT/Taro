# -*- coding: utf-8 -*-
"""親のセリフを全部ひらがなにした台本と、鎖の実験JSONを作る（2026-09-06）。

【仕様】F/docs/二語文/仕様_ひらがな化してM4をやり直す_2026-09-06.md
「後半：実装担当向け技術付録」「2. 台本の変換」節。

やること：
  1. 語彙の鎖8本のシーン＋消失発話シーンの計9本を複製し、
     world.parent_labeling.utterances と vanish_template だけを
     normalize_kana（カタカナ→ひらがな→長音展開）で変換して新シーンとして書き出す
  2. 変換後にカタカナ・「ー」が0文字であることを assert し、語の対応表を表示する
  3. F2-50_r{1..8}・F2-76・F2-76b の実験JSONを複製し、F2-77_r{1..8}・F2-78・F2-78b を作る
  4. F/logs/F2-77_ひらがな鎖/_chain.txt に実行順のパスを書く

実行：
    .venv\\Scripts\\python.exe F\\scripts\\f80_hiragana_scenes.py
"""
import os
import sys
import json
import copy

os.chdir(r"C:\claude\AI\Taro")
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("taro_core/src/senses"))
from hearing import normalize_kana  # noqa: E402

ROOT = r"C:\claude\AI\Taro"
SCENES_DIR = os.path.join(ROOT, "run", "scenes")
EXP_DIR = os.path.join(ROOT, "F", "experiments")
LOGDIR_NAME = "F2-77_ひらがな鎖"
CHAIN_DIR = os.path.join(ROOT, "F", "logs", LOGDIR_NAME)

KATAKANA_LO, KATAKANA_HI = u"\u30A1", u"\u30F6"


def has_katakana_or_onbiki(text):
    return any((KATAKANA_LO <= c <= KATAKANA_HI) or c == u"ー" for c in text)


def load_json(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=1)


# --- 1. シーンの変換 ---------------------------------------------------------

def transform_scene(data):
    """world.parent_labeling.utterances と vanish_template を normalize_kana で
    変換する。それ以外のフィールドは一切触らない。戻り値は (変換後data, 語の対応表)。
    """
    labeling = data["world"]["parent_labeling"]
    utterances = labeling["utterances"]
    word_map = {}
    base_word_map = {}  # 表示用：トイごとの一番短い（=素の名詞）エントリだけ
    new_utterances = {}
    for toy, entries in utterances.items():
        new_entries = []
        shortest_orig, shortest_new, shortest_len = None, None, None
        for text, prob in entries:
            new_text = normalize_kana(text)
            word_map[text] = new_text
            new_entries.append([new_text, prob])
            if shortest_len is None or len(text) < shortest_len:
                shortest_orig, shortest_new, shortest_len = text, new_text, len(text)
        new_utterances[toy] = new_entries
        base_word_map[toy] = (shortest_orig, shortest_new)
    labeling["utterances"] = new_utterances

    if "vanish_template" in labeling:
        old_t = labeling["vanish_template"]
        new_t = normalize_kana(old_t)
        labeling["vanish_template"] = new_t
        word_map[old_t] = new_t

    # 検査：カタカナ・「ー」が0文字であること
    for toy, entries in new_utterances.items():
        for text, _prob in entries:
            assert not has_katakana_or_onbiki(text), \
                "変換後にカタカナ/ーが残っている: toy=%s text=%r" % (toy, text)
    if "vanish_template" in labeling:
        assert not has_katakana_or_onbiki(labeling["vanish_template"]), \
            "vanish_templateにカタカナ/ーが残っている: %r" % labeling["vanish_template"]

    return data, word_map, base_word_map


def hiragana_scene_name(orig_name):
    """拡張子を除いた元名の末尾に _ひらがな_2026-09-06 を足す。"""
    stem = orig_name[:-5] if orig_name.endswith(".json") else orig_name
    return stem + "_ひらがな_2026-09-06"


def convert_one_scene(src_filename):
    src_path = os.path.join(SCENES_DIR, src_filename)
    data = load_json(src_path)
    data, word_map, base_word_map = transform_scene(data)
    new_name = hiragana_scene_name(src_filename)
    dst_path = os.path.join(SCENES_DIR, new_name + ".json")
    save_json(dst_path, data)
    return new_name, word_map, base_word_map


# --- 2. 実験JSONのパス置換ヘルパー -------------------------------------------

def replace_paths(obj, old, new):
    if isinstance(obj, dict):
        return {k: replace_paths(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_paths(v, old, new) for v in obj]
    if isinstance(obj, str):
        return obj.replace(old, new)
    return obj


SPEC_NOTE_REF = ("仕様：F/docs/二語文/"
                  "仕様_ひらがな化してM4をやり直す_2026-09-06.md")


def build_f277(k, scene_new_name):
    """F2-50_r{k} を複製して F2-77_r{k} を作る。"""
    src = load_json(os.path.join(EXP_DIR, "F2-50_r%d_2026-09-03.json" % k))
    data = copy.deepcopy(src)
    data = replace_paths(data, "F2-50_分節", LOGDIR_NAME)
    data["name"] = "F2-77_r%d" % k
    data["note"] = ("2026-09-06 ひらがな台本で F2-50 をやり直す（ラウンド%d/8）。" % k
                     + SPEC_NOTE_REF)
    data["scene"] = scene_new_name
    if k == 1:
        pass  # k=1はモデル不変（F2-6 喃語）
    else:
        data["taro"]["model"] = ("F/models/F2-77_r%d_seed%d_2026-09-06.pt"
                                  % (k - 1, 89 + k))
    data["taro"]["save"] = "F/models/F2-77_r%d_seed%d_2026-09-06.pt" % (k, 90 + k)
    out_path = os.path.join(EXP_DIR, "F2-77_r%d_2026-09-06.json" % k)
    save_json(out_path, data)
    return out_path


def build_f278(scene_new_name):
    src = load_json(os.path.join(EXP_DIR, "F2-76_M4_消失発話学習_2026-09-06.json"))
    data = copy.deepcopy(src)
    data = replace_paths(data, "F2-76_M4_消失発話学習", "F2-78_M4_ひらがな学習")
    data["name"] = "F2-78_M4_ひらがな学習"
    data["note"] = ("2026-09-06 F2-76_M4_消失発話学習をひらがな台本でやり直す。"
                     "土台は F2-77_r8。" + SPEC_NOTE_REF)
    data["scene"] = scene_new_name
    data["taro"]["model"] = "F/models/F2-77_r8_seed98_2026-09-06.pt"
    out_path = os.path.join(EXP_DIR, "F2-78_M4_ひらがな学習_2026-09-06.json")
    save_json(out_path, data)
    return out_path


def build_f278b(scene_new_name):
    src = load_json(os.path.join(EXP_DIR, "F2-76b_M4_テスト_2026-09-06.json"))
    data = copy.deepcopy(src)
    data = replace_paths(data, "F2-76b_M4_テスト", "F2-78b_M4_ひらがなテスト")
    data["name"] = "F2-78b_M4_ひらがなテスト"
    data["note"] = ("2026-09-06 F2-76b_M4_テストをひらがな台本でやり直す。"
                     "taro.modelはF2-78の保存モデル。" + SPEC_NOTE_REF)
    data["scene"] = scene_new_name
    data["taro"]["model"] = "F/logs/F2-78_M4_ひらがな学習/model.pt"
    out_path = os.path.join(EXP_DIR, "F2-78b_M4_ひらがなテスト_2026-09-06.json")
    save_json(out_path, data)
    return out_path


def main():
    # ---- シーンの変換（8本の語彙鎖 + 1本の消失発話） -----------------------
    all_word_map = {}
    round_scene_new_names = {}
    for k in range(1, 9):
        f250 = load_json(os.path.join(EXP_DIR, "F2-50_r%d_2026-09-03.json" % k))
        scene_field = f250["scene"]  # 例: 座位_12ヶ月_F2-49_実物8択_r1_個体1_2026-09-03
        src_filename = scene_field + ".json"
        new_name, word_map, base_word_map = convert_one_scene(src_filename)
        round_scene_new_names[k] = new_name
        all_word_map.update(word_map)
        if k == 1:
            print("=== ラウンド1のシーン：語の対応表（素の名詞のみ） ===")
            for toy, (orig, new) in sorted(base_word_map.items()):
                print("  %s: %s -> %s" % (toy, orig, new))

    vanish_src = "座位_12ヶ月_F2-49_実物8択_r1_個体1_消失発話1.5s_2026-09-06.json"
    vanish_new_name, vanish_word_map, vanish_base = convert_one_scene(vanish_src)
    all_word_map.update(vanish_word_map)

    print("\n=== 生成したシーン（9本） ===")
    for k in range(1, 9):
        print("  r%d: %s.json" % (k, round_scene_new_names[k]))
    print("  消失発話: %s.json" % vanish_new_name)

    print("\n=== 語の対応表（全ユニーク文字列） ===")
    for orig, new in sorted(all_word_map.items()):
        mark = " (変化あり)" if orig != new else ""
        print("  %s -> %s%s" % (orig, new, mark))

    # ---- 実験JSONの生成 -----------------------------------------------------
    chain_paths = []
    for k in range(1, 9):
        p = build_f277(k, round_scene_new_names[k])
        chain_paths.append(p)
        print("生成: %s" % p)

    p78 = build_f278(vanish_new_name)
    chain_paths.append(p78)
    print("生成: %s" % p78)

    p78b = build_f278b(vanish_new_name)
    chain_paths.append(p78b)
    print("生成: %s" % p78b)

    # ---- 一覧表示（scene・model・save・出力先） ------------------------------
    print("\n=== 実験JSONの一覧（scene / taro.model / taro.save / run.csv） ===")
    for p in chain_paths:
        d = load_json(p)
        print("  %s" % os.path.basename(p))
        print("    scene : %s" % d.get("scene"))
        print("    model : %s" % d.get("taro", {}).get("model"))
        print("    save  : %s" % d.get("taro", {}).get("save"))
        print("    csv   : %s" % d.get("run", {}).get("csv"))

    # ---- _chain.txt -----------------------------------------------------
    os.makedirs(CHAIN_DIR, exist_ok=True)
    chain_txt = os.path.join(CHAIN_DIR, "_chain.txt")
    rel_paths = ["F/experiments/%s" % os.path.basename(p) for p in chain_paths]
    with open(chain_txt, "w", encoding="utf-8") as fp:
        fp.write("\n".join(rel_paths) + "\n")
    print("\n生成: %s" % chain_txt)
    for rp in rel_paths:
        print("  %s" % rp)


if __name__ == "__main__":
    main()
