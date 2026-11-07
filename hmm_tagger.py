import itertools
import numpy as np
from collections import Counter
from dataio import load_conll, score, save

NEG = -np.inf


def word_shape(w):
    if any(ch.isdigit() for ch in w): return "num"
    if len(w) > 1 and w.isupper():    return "ALL"
    if w[0].isupper():                return "Init"
    return "low"


def suffix_keys(w):
    return [(n, word_shape(w), w.lower()[-n:]) for n in (1, 2, 3)]


def viterbi_padded(emis, log_a):
    """emis: (T,K) log-emissions; log_a: (K+2,K+2) log-transitions, START=K, END=K+1."""
    T, K = emis.shape
    S, E, n = K, K + 1, K + 2
    grid = np.full((T + 2, n), NEG)         
    grid[0, S] = 0.0
    grid[T + 1, E] = 0.0
    grid[1:T + 1, :K] = emis
    score_t = grid[0].copy()
    back = np.zeros((T + 2, n), dtype=np.int64)
    for t in range(1, T + 2):
        cand = score_t[:, None] + log_a     
        back[t] = cand.argmax(axis=0)
        score_t = cand.max(axis=0) + grid[t]
    state, walk = E, []
    for t in range(T + 1, 0, -1):
        state = int(back[t, state])
        walk.append(state)
    return walk[:-1][::-1]                 


class CountHMM:
    def __init__(self, n_tags, smooth=0.1, rare_max=1, use_suffix=True):
        self.K, self.lam, self.rare_max, self.use_suffix = n_tags, smooth, rare_max, use_suffix

    def fit(self, sents, tags):
        K, lam = self.K, self.lam
        S, E = K, K + 1
        freq = Counter(w for s in sents for w in s)
        self.vocab = {w: i for i, w in enumerate(freq)}
        V = len(self.vocab)
        words = np.fromiter((self.vocab[w] for s in sents for w in s), dtype=np.int64)
        gold = np.fromiter(itertools.chain.from_iterable(tags), dtype=np.int64)

        table = np.zeros((V, K))
        np.add.at(table, (words, gold), 1.0)
        self.log_emit = np.log(table + lam) - np.log(table.sum(0) + lam * (V + 1))
        
        prev = np.fromiter(itertools.chain.from_iterable([S] + list(t) for t in tags), dtype=np.int64)
        nxt = np.fromiter(itertools.chain.from_iterable(list(t) + [E] for t in tags), dtype=np.int64)
        counts = np.zeros((K + 2, K + 2))
        np.add.at(counts, (prev, nxt), 1.0)
        allowed = np.ones((K + 2, K + 2))    
        allowed[:, S] = 0; allowed[E, :] = 0; allowed[S, E] = 0
        sm = counts + lam * allowed
        rows = sm.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.log_a = np.log(sm / np.where(rows == 0, 1, rows))   
        self.suf = {}
        self.rare_total = np.zeros(K)
        for w, t in zip((w for s in sents for w in s), gold):
            if freq[w] <= self.rare_max:
                self.rare_total[t] += 1
                for key in suffix_keys(w):
                    self.suf.setdefault(key, np.zeros(K))[t] += 1
        self.n_keys = Counter(k[0] for k in self.suf)
        return self

    def _unseen(self, w):
        if self.use_suffix:
            ps = [(self.suf[k] + self.lam) / (self.rare_total + self.lam * self.n_keys[k[0]])
                  for k in suffix_keys(w) if k in self.suf]
            if ps:
                return np.log(np.mean(ps, axis=0))
        return np.full(self.K, -np.log(self.K))

    def emissions(self, words):
        return np.stack([self.log_emit[self.vocab[w]] if w in self.vocab else self._unseen(w)
                         for w in words])

    def tag(self, sents):
        return [viterbi_padded(self.emissions(s), self.log_a) for s in sents]


def main():
    c = load_conll()
    K = len(c.labels)
    print("smoothing sweep on dev:")
    for lam in (0.01, 0.1, 1.0):
        m = CountHMM(K, lam).fit(c.train.tokens, c.train.tags)
        print(f"  lambda={lam:<5} F1={score(c.dev.tags, m.tag(c.dev.tokens), c.labels)['f1']:.4f}")
    m = CountHMM(K, 0.1).fit(c.train.tokens, c.train.tags)
    m.use_suffix = False
    print("no suffix model, dev F1:", round(score(c.dev.tags, m.tag(c.dev.tokens), c.labels)["f1"], 4))
    m.use_suffix = True
    res = score(c.test.tags, m.tag(c.test.tokens), c.labels, report=True)
    print("TEST:", {k: round(v, 4) for k, v in res.items()})
    save("task1", res)


if __name__ == "__main__":
    main()
