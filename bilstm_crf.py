import argparse, copy, random, time
from collections import Counter
from dataclasses import dataclass
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from dataio import load_conll, score, save

MAXC = 30

@dataclass
class Vocab:
    words: dict
    chars: dict
    singletons: set

    def word(self, w):
        return self.words.get(w.lower(), 1)

    def char_ids(self, w):
        return [self.chars.get(c, 1) for c in w[:MAXC]] or [1]


def make_vocab(corpus, glove_file, dim=100):
    train_cnt = Counter(w.lower() for s in corpus.train.tokens for w in s)
    other = {w.lower() for sp in (corpus.dev, corpus.test) for s in sp.tokens for w in s}
    wanted = set(train_cnt) | other
    glove = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            head, rest = line.split(" ", 1)
            if head in wanted:
                glove[head] = np.array(rest.split(), dtype=np.float32)
    vocab_words = ["<pad>", "<unk>"] + sorted(set(train_cnt) | (other & set(glove)))
    words = {w: i for i, w in enumerate(vocab_words)}
    bound = (3.0 / dim) ** 0.5
    matrix = np.random.RandomState(0).uniform(-bound, bound, (len(words), dim)).astype(np.float32)
    matrix[0] = 0
    for w, v in glove.items():
        matrix[words[w]] = v
    chars = {c: i + 2 for i, c in enumerate(sorted({c for s in corpus.train.tokens for w in s for c in w}))}
    singles = {w for w, n in train_cnt.items() if n == 1}
    return Vocab(words, chars, singles), matrix


class Sentences(Dataset):
    def __init__(self, sents, tags, vocab, train=False):
        self.sents, self.tags, self.vocab, self.train = sents, tags, vocab, train
        self.chars = [[torch.tensor(vocab.char_ids(w)) for w in s] for s in sents]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        ids = []
        for w in self.sents[i]:
            j = self.vocab.word(w)
            if self.train and w.lower() in self.vocab.singletons and random.random() < 0.5:
                j = 1                                
            ids.append(j)
        tg = torch.tensor(self.tags[i]) if self.tags is not None else torch.zeros(len(ids), dtype=torch.long)
        return torch.tensor(ids), self.chars[i], tg


def collate(items):
    words = pad_sequence([it[0] for it in items], batch_first=True)
    tags = pad_sequence([it[2] for it in items], batch_first=True)
    lens = torch.tensor([len(it[0]) for it in items])
    flat = [c for it in items for c in it[1]]           
    clens = torch.tensor([len(c) for c in flat])
    return words, pad_sequence(flat, batch_first=True), clens, lens, tags

class Encoder(nn.Module):
    def __init__(self, n_words, n_chars, matrix=None, wdim=100, cdim=25, chid=25, whid=100, p=0.5):
        super().__init__()
        self.wemb = nn.Embedding(n_words, wdim, padding_idx=0)       
        if matrix is not None:
            self.wemb.weight.data.copy_(torch.from_numpy(matrix))
        self.cemb = nn.Embedding(n_chars, cdim, padding_idx=0)
        self.clstm = nn.LSTM(cdim, chid, batch_first=True, bidirectional=True)
        self.wlstm = nn.LSTM(wdim + 2 * chid, whid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.out_dim = 2 * whid

    def forward(self, words, chars, clens, lens):
        B, T = words.shape
        mask = torch.arange(T, device=words.device)[None, :] < lens[:, None]
        packed = pack_padded_sequence(self.cemb(chars), clens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.clstm(packed)
        per_word = torch.cat([h[0], h[1]], -1)                      
        char_feats = per_word.new_zeros(B, T, per_word.size(1))
        char_feats[mask] = per_word                                  
        x = self.drop(torch.cat([self.wemb(words), char_feats], -1))
        out, _ = self.wlstm(pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False))
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.drop(out), mask


class LinearChainCRF(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.trans = nn.Parameter(torch.empty(K, K).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(K).uniform_(-0.1, 0.1))
        self.stop = nn.Parameter(torch.empty(K).uniform_(-0.1, 0.1))

    def _gold(self, em, tags, mask):
        m = mask.float()
        s = self.start[tags[:, 0]] + (em.gather(2, tags.unsqueeze(-1)).squeeze(-1) * m).sum(1)
        s = s + (self.trans[tags[:, :-1], tags[:, 1:]] * m[:, 1:]).sum(1)
        last = tags.gather(1, (mask.sum(1) - 1)[:, None]).squeeze(1)
        return s + self.stop[last]

    def _log_z(self, em, mask):
        alpha = self.start + em[:, 0]                                  
        for t in range(1, em.size(1)):
            step = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0), dim=1) + em[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), step, alpha)    
        return torch.logsumexp(alpha + self.stop, dim=1)

    def nll(self, em, tags, mask):
        return (self._log_z(em, mask) - self._gold(em, tags, mask)).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        score_t = self.start + em[:, 0]
        hist = []
        for t in range(1, em.size(1)):
            best, arg = (score_t.unsqueeze(2) + self.trans.unsqueeze(0)).max(dim=1)
            score_t = torch.where(mask[:, t].unsqueeze(1), best + em[:, t], score_t)
            hist.append(arg)                                            
        last = (score_t + self.stop).argmax(1)
        lens = mask.sum(1).tolist()
        paths = []
        for b, L in enumerate(lens):
            tag = int(last[b]); path = [tag]
            for t in range(L - 1, 0, -1):
                tag = int(hist[t - 1][b, tag]); path.append(tag)
            paths.append(path[::-1])
        return paths


class CRFHead(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.proj, self.crf = nn.Linear(d, K), LinearChainCRF(K)

    def loss(self, feats, tags, mask):
        return self.crf.nll(self.proj(feats), tags, mask)

    def predict(self, feats, mask):
        return self.crf.decode(self.proj(feats), mask)


class SoftmaxHead(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.proj = nn.Linear(d, K)

    def loss(self, feats, tags, mask):
        return F.cross_entropy(self.proj(feats)[mask], tags[mask], reduction="sum") / feats.size(0)

    def predict(self, feats, mask):
        pred = self.proj(feats).argmax(-1)
        return [p[m].tolist() for p, m in zip(pred, mask)]

    def log_probs(self, feats, mask):
        lp = F.log_softmax(self.proj(feats), -1)
        return [l[m].cpu().numpy() for l, m in zip(lp, mask)]


class Tagger(nn.Module):
    def __init__(self, vocab_sizes, K, head, matrix=None):
        super().__init__()
        self.enc = Encoder(vocab_sizes[0], vocab_sizes[1], matrix)
        self.head = (CRFHead if head == "crf" else SoftmaxHead)(self.enc.out_dim, K)
        self.kind = head

def _loader(sents, tags, vocab, train=False, bs=64):
    return DataLoader(Sentences(sents, tags, vocab, train), batch_size=bs, shuffle=train, collate_fn=collate)


@torch.no_grad()
def infer(model, sents, vocab, device, what="tags"):
    model.eval()
    out = []
    for batch in _loader(sents, None, vocab):
        words, chars, clens, lens, _ = [b.to(device) for b in batch]
        feats, mask = model.enc(words, chars, clens, lens)
        out += model.head.predict(feats, mask) if what == "tags" else model.head.log_probs(feats, mask)
    return out


def restore(path, K, device):
    ck = torch.load(path, map_location=device)
    v = Vocab(ck["words"], ck["chars"], set(ck["singletons"]))
    model = Tagger((len(v.words), len(v.chars) + 2), K, ck["head"]).to(device)
    model.load_state_dict(ck["state"])
    return model, v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", required=True)
    ap.add_argument("--head", choices=["crf", "softmax"], default="crf")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c = load_conll()
    K = len(c.labels)
    vocab, matrix = make_vocab(c, a.glove)
    model = Tagger((len(vocab.words), len(vocab.chars) + 2), K, a.head, matrix).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 / (1.0 + 0.05 * e))
    train_loader = _loader(c.train.tokens, c.train.tags, vocab, train=True, bs=a.bs)

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for batch in train_loader:
            words, chars, clens, lens, tags = [b.to(dev) for b in batch]
            feats, mask = model.enc(words, chars, clens, lens)
            loss = model.head.loss(feats, tags, mask)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * words.size(0)
        sched.step()
        f1 = score(c.dev.tags, infer(model, c.dev.tokens, vocab, dev), c.labels)["f1"]
        print(f"ep {epoch+1:2d} loss {total/len(c.train.tokens):.3f} dev-F1 {f1:.4f} ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1, best_state, stale = f1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= a.patience:
                break

    model.load_state_dict(best_state)
    torch.save({"state": best_state, "head": a.head, "words": vocab.words, "chars": vocab.chars,
                "singletons": sorted(vocab.singletons)}, f"best_{a.head}.pt")
    res = score(c.test.tags, infer(model, c.test.tokens, vocab, dev), c.labels, report=True)
    print(f"best dev {best_f1:.4f}; TEST:", {k: round(v, 4) for k, v in res.items()})
    save("task2" if a.head == "crf" else "task2_softmax", res)


if __name__ == "__main__":
    main()
