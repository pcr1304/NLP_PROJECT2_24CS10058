import json
from dataclasses import dataclass
from pathlib import Path
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

OUT = Path("out")


@dataclass
class Split:
    tokens: list  
    tags: list     


@dataclass
class Corpus:
    train: Split
    dev: Split
    test: Split
    labels: list  


def load_conll():
    from datasets import load_dataset
    ds = None
    for kwargs in ({}, {"trust_remote_code": True}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset("eriktks/conll2003", **kwargs)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("CoNLL-2003 failed to load; try: pip install 'datasets<4'")
    labels = ds["train"].features["ner_tags"].feature.names
    grab = lambda name: Split(ds[name]["tokens"], ds[name]["ner_tags"])
    return Corpus(grab("train"), grab("validation"), grab("test"), labels)


def score(gold, pred, labels, report=False):
    to_str = lambda seqs: [[labels[i] for i in s] for s in seqs]
    g, p = to_str(gold), to_str(pred)
    if report:
        print(classification_report(g, p, digits=4))
    return dict(precision=float(precision_score(g, p)),
                recall=float(recall_score(g, p)),
                f1=float(f1_score(g, p)))


def save(name, metrics):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(metrics, indent=2))


def load(name):
    f = OUT / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None
