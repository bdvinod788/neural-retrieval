import os
import json

from torch.utils.data import Dataset


def load_tsv(path, keep=None):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, value = line.rstrip("\n").split("\t", 1)
            if keep is None or key in keep:
                out[key] = value
    return out


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
            elif len(parts) == 3:
                qid, docid, rel = parts
            else:
                continue
            if not rel.lstrip("-").isdigit():
                continue
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_beir(dataset, split="test", data_dir=None):
    data_dir = data_dir or os.path.join(
        os.environ.get("NR_PROJECT", "."), "data", "beir", dataset
    )

    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = (doc.get("title", "") + " " + doc.get("text", "")).strip()

    queries = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    qrels = load_qrels(os.path.join(data_dir, "qrels", f"{split}.tsv"))
    queries = {qid: text for qid, text in queries.items() if qid in qrels}

    return corpus, queries, qrels


def write_jsonl_corpus(corpus, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for docid, text in corpus.items():
            f.write(json.dumps({"id": docid, "contents": text}) + "\n")


def load_msmarco_pairs(qrels_path):
    qrels = load_qrels(qrels_path)
    return [
        (qid, pid)
        for qid, docs in qrels.items()
        for pid, rel in docs.items()
        if rel > 0
    ]


class TripleDataset(Dataset):
    def __init__(self, rows, queries, collection):
        self.rows = rows
        self.queries = queries
        self.collection = collection

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = {
            "query": self.queries[row[0]],
            "positive": self.collection[row[1]],
        }
        if len(row) > 2:
            item["negative"] = self.collection[row[2]]
        return item


def make_biencoder_collator(tokenizer, max_query_len=32, max_passage_len=128):
    def collate(batch):
        queries = tokenizer(
            [b["query"] for b in batch],
            padding=True, truncation=True, max_length=max_query_len,
            return_tensors="pt",
        )
        passages = [b["positive"] for b in batch]
        if "negative" in batch[0]:
            passages += [b["negative"] for b in batch]
        passages = tokenizer(
            passages,
            padding=True, truncation=True, max_length=max_passage_len,
            return_tensors="pt",
        )
        return {"queries": dict(queries), "passages": dict(passages)}

    return collate
