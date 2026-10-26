"""Reference outputs for test/sql/parity_*.test, from the official tokenizer and ONNX graph of DecisionTune 1.0.

    uv run make_fixtures.py ~/.duckdb/decision_tune/1.0

The directory must hold tokenizer.json and model.onnx (the extension's download cache does). The prompt rendering
below is the one of the reference engine (engine.py in the model repository); it is restated here rather than
imported so that this script only needs `tokenizers`, `onnxruntime` and `numpy`.
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

OUT = Path(__file__).resolve().parents[2] / "test" / "data" / "parity"
SEP = "\u241e"  # joins options in one CSV cell
SPECIAL = ("[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]")


def clean(s):
    for t in SPECIAL:
        s = s.replace(t, " ")
    return s


class Reference:
    def __init__(self, model_dir):
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.CLS, self.SEP, self.MASK = (self.tok.token_to_id(t) for t in ("[CLS]", "[SEP]", "[MASK]"))

    def tokens(self, s):
        return self.tok.encode(clean(s), add_special_tokens=False).ids

    def probabilities(self, state, question, options):
        ids, pos = [self.CLS] + self.tokens(question) + [self.SEP], []
        for o in options:
            pos.append(len(ids))
            ids += [self.MASK] + self.tokens(o)
        ids += [self.SEP] + self.tokens(state) + [self.SEP]
        x = np.array([ids], dtype=np.int64)
        logits = self.sess.run(["logits"], {"input_ids": x, "attention_mask": np.ones_like(x),
                                            "positions": np.array(pos, dtype=np.int64)})[0].astype(np.float64)
        z = np.exp(logits - logits.max())
        return z / z.sum(), len(ids)


TOKENIZER_TEXTS = [
    "", " ", "  ", "   ", " " * 24, " " * 25, " " * 60, "\t", "\n", "\r\n", "\n\n\n", " \n ", "a\n\nb",
    "The order arrived broken.", "The order arrived broken. I want my money back.",
    "hello  world", "hello   world", "hello    world  ", "  leading", "trailing  ", "a \tb", "a\t\tb",
    "don't", "I'll", "we're", "they've", "I'm", "he'd", "it's", "IT'S", "'s", "''", "rock'n'roll", "o'clock",
    "12345", "3.14159", "1,000,000", "v2.0.1", "2026-10-26T09:15:00Z", "$1,299.99", "50%", "No. 7",
    "café", "cafe\u0301", "naïve", "Ångström", "Straße", "ñandú", "Привет, мир!", "Γειά σου", "שלום עולם",
    "مرحبا بالعالم", "नमस्ते दुनिया", "你好，世界", "日本語のテキスト", "한국어 텍스트", "ไทย", "😀👍🏽🇫🇷",
    "e-mail: someone@example.com", "https://decisiontune.com/docs?x=1&y=2#top", "C:\\Users\\me\\file.txt",
    "def f(x):\n    return x ** 2\n", "<html><body>Hi</body></html>", '{"a": 1, "b": [true, null]}',
    "SELECT * FROM t WHERE a = 'b';", "!!!???...", "---", "a—b–c", "«quote» “quote” ‘q’", "\u00a0nbsp\u00a0",
    "ideographic\u3000space", "zero\u200bwidth", "line\u2028separator", "tab\there", "ＦＵＬＬＷＩＤＴＨ",
    "<|endoftext|>", "x<|endoftext|>y", "<|padding|>", "[MASK]", "a [CLS] b [SEP] c", "[PAD][UNK]",
    "|||EMAIL_ADDRESS|||", "call |||PHONE_NUMBER||| now", "|||IP_ADDRESS|||", "[unused0]", "x[unused12]y",
    "Billing: charges, refunds, invoices", "Shipping: delivery, lost or damaged packages",
    "Tech support: bugs, crashes, login problems", "subject: Broken mug | message: The order arrived broken.",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "ab" * 200,
]


def random_texts(n, seed=7):
    pool = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) +
            list("\t\n\r'\".,;:!?-_()[]{}<>/\\|@#$%^&*+=~`") + list("éèàüßçñøåÉ中文日本語한국어€£¥½²³") +
            ["\u0301", "\u00a0", "\u3000", "\u200b", "😀", "👍🏽", "[MASK]", "<|endoftext|>", "'s", "'ll", "  ", "   "])
    rng = random.Random(seed)
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, 80))) for _ in range(n)]


DECISIONS = [
    ("The order arrived broken.", "Which team should handle this customer message?",
     ["Billing: charges, refunds, invoices", "Shipping: delivery, lost or damaged packages",
      "Tech support: bugs, crashes, login problems"]),
    ("The order arrived broken. I want my money back.", "Is the customer asking for a refund?", ["yes", "no"]),
    ("What is the weather tomorrow in Paris?", "Which tool should be called?",
     ["Get the weather forecast for a city and date", "Send an email to a person",
      "Create a meeting or event in the calendar", "No tool: answer the user directly"]),
    ("subject: Broken mug | message: The order arrived broken. I want my money back.",
     "Is the customer asking for a refund or an exchange?", ["yes", "no"]),
    ("subject: Broken mug | message: The order arrived broken. I want my money back.",
     "What is the tone of the customer?",
     ["Negative: the customer is unhappy or upset", "Neutral: a plain request or question",
      "Positive: the customer is happy or thankful"]),
    ("I was charged twice this month.", "Which team handles this?", ["billing", "shipping", "tech support"]),
    ("Thanks a lot, everything works great now!", "Does this message ask to delete the account?", ["yes", "no"]),
    ("", "Is the state empty?", ["yes", "no"]),
    ("Le colis est arrivé cassé, je veux être remboursé.", "Which team should handle this customer message?",
     ["Billing: charges, refunds, invoices", "Shipping: delivery, lost or damaged packages",
      "Tech support: bugs, crashes, login problems", "None of these: spam, praise, or a message for no team"]),
    ("My app crashes [MASK] on login since the update <|endoftext|>", "Which team?", ["billing", "tech"]),
    ("a  b   c    d", "Which letter comes first?", ["a", "b", "c", "d"]),
]


def main():
    ref = Reference(Path(sys.argv[1]).expanduser())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tokenizer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "text", "ids"])
        for i, text in enumerate(TOKENIZER_TEXTS + random_texts(300)):
            w.writerow([i, text, str(ref.tokens(text))])
    with open(OUT / "decisions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "state", "question", "options", "probabilities", "prompt_tokens"])
        for i, (state, question, options) in enumerate(DECISIONS):
            assert not any(SEP in o for o in options)
            p, n = ref.probabilities(state, question, options)
            w.writerow([i, state, question, SEP.join(options), "[" + ", ".join(repr(float(v)) for v in p) + "]", n])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
