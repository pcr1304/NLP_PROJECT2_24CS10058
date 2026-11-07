import numpy as np
import torch
from dataio import load_conll, score, save, load
from hmm_tagger import CountHMM, viterbi_padded
from bilstm_crf import restore, infer


class NeuralEmissionHMM:
    def __init__(self, count_hmm, train_tags, K):
        self.log_a = count_hmm.log_a                       
        freq = np.bincount([t for s in train_tags for t in s], minlength=K).astype(float)
        self.log_prior = np.log(freq / freq.sum())

    def tag(self, posteriors, alpha=1.0):
        return [viterbi_padded(lp - alpha * self.log_prior, self.log_a) for lp in posteriors]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = load_conll()
    K = len(c.labels)
    hybrid = NeuralEmissionHMM(CountHMM(K, 0.1).fit(c.train.tokens, c.train.tags), c.train.tags, K)
    net, vocab = restore("best_softmax.pt", K, dev)
    post_dev = infer(net, c.dev.tokens, vocab, dev, what="logprobs")
    post_test = infer(net, c.test.tokens, vocab, dev, what="logprobs")

    print("prior exponent alpha on dev (alpha=1.0 is the specified model):")
    for alpha in (0.0, 0.5, 1.0):
        print(f"  alpha={alpha}: F1={score(c.dev.tags, hybrid.tag(post_dev, alpha), c.labels)['f1']:.4f}")

    res = score(c.test.tags, hybrid.tag(post_test), c.labels, report=True)
    print("TEST:", {k: round(v, 4) for k, v in res.items()})
    save("task3", res)

    print("\n| model | P | R | F1 |\n|---|---|---|---|")
    for name, key in [("Task 1 HMM", "task1"), ("Task 2 BiLSTM-CRF", "task2"),
                      ("BiLSTM-softmax (extra)", "task2_softmax"), ("Task 3 neural-emission HMM", "task3")]:
        r = load(key)
        if r:
            print(f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")


if __name__ == "__main__":
    main()
