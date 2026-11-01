"""End-to-end check on a laptop CPU in about 2 minutes, no downloads, no GPU.

Builds a tiny fake corpus, trains all four tokenizers, trains a tiny model with
each, scores the JLPT difficulty set, and draws the figures, all in a temporary
folder. If this passes, the same code will run on Modal.

  python code/smoke_test.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SUBS = """はい、わかりました。 ありがとう。 ちょっと待って！ 何してるの？ 大丈夫？ 本当に？ 行こう。
約束したよね。 もう遅いよ。 早く来て。 どこに行くの？ 誰だ？ お腹すいた。 また明日ね。
そんなこと言わないで。 ごめんなさい。 先生、質問があります。 今日は雨だね。 一緒に帰ろう。
彼女に会いたい。 それは秘密だ。 信じてくれ。 俺がやる。 心配しないで。 映画を見に行こう。
昨日は友達と映画を見に行きました。 この本はとても面白いです。 駅までどのくらいかかりますか。""".split()
WIKI = """東京は日本の首都であり、世界有数の大都市である。 富士山は静岡県と山梨県にまたがる活火山である。
日本語は主に日本で話されている言語である。 江戸時代は1603年から1868年まで続いた。
新幹線は日本の高速鉄道である。 京都には多くの寺院や神社がある。 経済は第二次世界大戦後に急速に成長した。
漢字は中国から伝わった文字である。 平仮名と片仮名は漢字から作られた。 人口は約一億二千万人である。""".split()


def corpus(lines, n, rng):
    return "\n".join(rng.choice(lines) + rng.choice(["", "", "ね", "よ"]) for _ in range(n)) + "\n"


def main():
    from data import prepare_tokenizer
    from difficulty import build_set, evaluate_set
    from evaluate import sentence_bits_per_byte
    from model import GPT, GPTConfig
    from tokenizers_ja import SPECS, load_tokenizer
    from train import train

    import torch

    rng = random.Random(0)
    tmp = Path(tempfile.mkdtemp(prefix="tjlm_smoke_"))
    raw = tmp / "data" / "raw"
    raw.mkdir(parents=True)
    for src, lines in (("subs", SUBS), ("wiki", WIKI)):
        for split, n in (("train", 20000), ("val", 400), ("test", 400)):
            (raw / f"{src}_{split}.txt").write_text(corpus(lines, n, rng), encoding="utf-8")

    for spec in SPECS:
        meta = prepare_tokenizer(tmp / "data", spec, mb_per_source=1)
        tok = load_tokenizer(tmp / "data" / "tok" / spec / "tokenizer.json")
        sample = "約束したよね。漢字テスト🍣"  # includes a character the tokenizer never saw
        assert tok.decode(tok.encode(sample)) == sample, f"{spec} is not lossless"
        print(f"  {spec}: vocab {meta['vocab_size']}, round-trip ok")

    results = {}
    for spec in SPECS:
        names = {"byte": "tok_byte", "char": "tok_char", "bpe4k": "tok_bpe4k", "bpe16k": "base"}
        cfg = dict(name=names[spec], tokenizer=spec, size="tiny", lr=3e-3, batch_size=8,
                   seq_len=64, budget_s=8, eval_every_s=4, val_tokens=2000, test_tokens=4000,
                   subs_weight=0.5 if spec == "byte" else "natural")
        results[spec] = train(cfg, str(tmp / "data"), str(tmp), "cpu")
        assert 0 < results[spec]["test_bpb_subs"] < 9, results[spec]["test_bpb_subs"]

    # Difficulty: real JLPT lists, our own sentences as the pool.
    pool = SUBS + WIKI + ["猫が好きです。", "水を飲みたい。", "明日は学校に行きます。", "彼は医者になった。"]
    n = build_set(pool, HERE.parent / "data" / "jlpt", tmp / "difficulty.jsonl", per_level=50)
    rows = [json.loads(x) for x in (tmp / "difficulty.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 5, n
    ck = torch.load(tmp / "ckpt" / "base.pt")
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["state"])
    tok = load_tokenizer(tmp / "data" / "tok" / "bpe16k" / "tokenizer.json")
    scores = sentence_bits_per_byte(model, tok, [r["sentence"] for r in rows], "cpu")
    diff = evaluate_set(rows, scores)
    print(f"  difficulty set: {len(rows)} sentences, spearman {diff['spearman']}")
    (tmp / "results" / "difficulty_smoke.json").write_text(json.dumps(diff))

    os.environ["TJLM_RESULTS"] = str(tmp / "results")
    os.environ["TJLM_FIGURES"] = str(tmp / "figures")
    import plot

    plot.main()
    print(f"\nSMOKE TEST PASSED. Everything was written to {tmp}")


if __name__ == "__main__":
    main()
