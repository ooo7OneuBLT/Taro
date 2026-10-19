"""盤面の世界の親の言葉と、耳の下駄の答えを、LLM で先に作って保存する（3段目、2026-10-19）。

育てる間に LLM を呼ぶと遅すぎる（クラウドのCPUで1文15秒ほど）ので、場面ごとの親の一言を先にまとめて作り、
その一言を耳の下駄（文→単語）に通した答えも一緒に core_v2/data/盤面_親の言葉.json に保存する。
口の下駄（言う語→赤ちゃんの言い方）もここで作る。

LLM：sbintuitions/sarashina2.2-3b-instruct-v0.1（日本語、30億。transformers で読む）
使い方：
  python -m core_v2.tools.make_board_speech [--per 20] [--out core_v2/data/盤面_親の言葉.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re

MODEL = "sbintuitions/sarashina2.2-3b-instruct-v0.1"

# 親の役（{who} はママかパパ）。答えは赤ちゃんへのせりふだけ（例を1つ見せる）
PARENT_SYS = ("あなたは1歳の赤ちゃんの{who}です。場面を読んで、赤ちゃんに言うせりふだけを、ひらがなとカタカナで、"
              "20文字以内で1つ書きます。説明やかぎかっこは書きません。")
PARENT_SHOT = ("場面：赤ちゃんがお風呂でおもちゃのアヒルを見ている", "アヒルさんだね、ぷかぷかしてるね")
# 場面 → 場面の説明
SITUATIONS = {
    "feed": "場面：赤ちゃんがおなかをすかせているので、今からミルク（まんま）をあげる",
    "called": "場面：赤ちゃんに呼ばれたので、そばに来た",
    "busy": "場面：赤ちゃんが呼んでいるが、今は手が離せない",
    "chat": "場面：そばで遊んでいる赤ちゃんに、目に入った物（ボール・わんわん・くつ・コップなど）について話しかける",
}
EAR_SYS = "文を単語に区切り、名詞と動詞と形容詞だけを、文に出てきた形のまま、空白区切りで1行で出す。ほかは何も書かない。"
EAR_SHOTS = [("ねえ、くつはいてお外いこうか", "くつ はいて おそと いこう"),
             ("わんわんいたね、かわいいね", "わんわん いた かわいい")]
MOUTH_SYS = "あなたは1歳の赤ちゃんの口です。渡された単語1つだけを、赤ちゃんが言うように短く言ってください。ほかの言葉は足さないでください。"


def load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("TARO_LLM_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    return tok, model


def chat(tok, model, msgs, n=30, sample=False):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    kw = {"do_sample": True, "temperature": 0.9, "top_p": 0.95} if sample else {"do_sample": False}
    out = model.generate(ids, max_new_tokens=n, **kw)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


def clean_line(s: str) -> str:
    m = re.search(r"「([^」]+)」", s)
    s = (m.group(1) if m else s).strip().strip("「」\"' ")
    # 説明が混ざった答え・長すぎる答えは捨てる
    if len(s) > 25 or re.search(r"赤ちゃん|ひらがな|カタカナ|場面|[一-龥]{3,}", s):
        return ""
    return s


def ear_words(tok, model, line: str) -> list[str]:
    msgs = [{"role": "system", "content": EAR_SYS}]
    for a, b in EAR_SHOTS:
        msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": line})
    raw = chat(tok, model, msgs, n=25)
    words = [re.sub(r"[、。！？!?「」，,.~〜]+", "", x) for x in raw.split()]
    return [w for w in words if w and len(w) <= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20, help="場面×人ごとの文の数")
    ap.add_argument("--out", default="core_v2/data/盤面_親の言葉.json")
    ap.add_argument("--mouth", nargs="*", default=["まんま", "ママ", "パパ", "ミルク", "ボール", "わんわん", "ねんね"])
    args = ap.parse_args()
    import torch
    torch.manual_seed(0)
    tok, model = load()
    data = {"model": MODEL, "lines": {}, "ear": {}, "mouth": {}}
    if os.path.exists(args.out):
        data = json.load(open(args.out))
    for sit, prompt in SITUATIONS.items():
        for who in ("ママ", "パパ"):
            key = f"{sit}:{who}"
            lines = data["lines"].setdefault(key, [])
            while len(lines) < args.per:
                msgs = [{"role": "system", "content": PARENT_SYS.format(who=who)},
                        {"role": "user", "content": PARENT_SHOT[0]}, {"role": "assistant", "content": PARENT_SHOT[1]},
                        {"role": "user", "content": prompt}]
                line = clean_line(chat(tok, model, msgs, sample=True))
                if line and line not in lines:
                    lines.append(line)
                    data["ear"][line] = ear_words(tok, model, line)
                    print(key, line, "→", data["ear"][line], flush=True)
                    json.dump(data, open(args.out, "w"), ensure_ascii=False, indent=1)
    for w in args.mouth:
        if w not in data["mouth"]:
            data["mouth"][w] = clean_line(chat(tok, model, [{"role": "system", "content": MOUTH_SYS},
                                                             {"role": "user", "content": w}], n=10))
            print("口", w, "→", data["mouth"][w], flush=True)
    json.dump(data, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
