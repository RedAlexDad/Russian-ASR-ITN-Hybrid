"""
Character-level n-gram language model for parser confidence estimation.

Usage:
    lm = CharNGram(order=4, smoothing_k=1.0)
    lm.train(["25 рублей", "суммарно от 300 до 1300", ...])
    lm.save("models/char_ngram.pkl")
    prob = lm.score("25 рублей")  # log-probability
"""

import math
import pickle
import warnings
from collections import Counter


class CharNGram:
    def __init__(self, order=4, smoothing_k=1.0):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.k = smoothing_k
        self._ngram_counts = Counter()
        self._context_counts = Counter()
        self._vocab = set()
        self._vocab_size = 0
        self._trained = False

    def _pad(self, text):
        return "\x02" * (self.order - 1) + text + "\x03"

    def _get_ngrams(self, text):
        padded = self._pad(text)
        for i in range(self.order - 1, len(padded)):
            context = padded[i - self.order + 1 : i]
            token = padded[i]
            yield context, token

    def train(self, texts):
        for text in texts:
            for ctx, tok in self._get_ngrams(text):
                ngram = ctx + tok
                self._ngram_counts[ngram] += 1
                self._context_counts[ctx] += 1
                self._vocab.add(tok)
                for ch in ctx:
                    self._vocab.add(ch)
        self._vocab_size = len(self._vocab)
        self._trained = True

    def score(self, text):
        if not self._trained:
            raise ValueError("model not trained")
        log_prob = 0.0
        n_tokens = 0
        for ctx, tok in self._get_ngrams(text):
            ngram = ctx + tok
            numer = self._ngram_counts.get(ngram, 0) + self.k
            denom = self._context_counts.get(ctx, 0) + self.k * self._vocab_size
            log_prob += math.log(numer / denom)
            n_tokens += 1
        return log_prob / n_tokens if n_tokens > 0 else float("-inf")

    def save(self, path):
        data = {
            "order": self.order,
            "k": self.k,
            "ngram_counts": self._ngram_counts,
            "context_counts": self._context_counts,
            "vocab": self._vocab,
            "vocab_size": self._vocab_size,
        }
        for attr in ("threshold_min", "threshold_p5", "threshold_p10",
                     "threshold_mean", "threshold_std", "n_train"):
            val = getattr(self, attr, None)
            if val is not None:
                data[attr] = val
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        lm = cls(order=data["order"], smoothing_k=data["k"])
        lm._ngram_counts = data["ngram_counts"]
        lm._context_counts = data["context_counts"]
        lm._vocab = data["vocab"]
        lm._vocab_size = data["vocab_size"]
        lm._trained = True
        for attr in ("threshold_min", "threshold_p5", "threshold_p10",
                     "threshold_mean", "threshold_std", "n_train"):
            if attr in data:
                setattr(lm, attr, data[attr])
        return lm
