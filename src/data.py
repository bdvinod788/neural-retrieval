import os
import json


def load_tsv(path):
    # Two-column TSV: id -> text. Used for collection.tsv and queries.*.tsv.
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, value = line.rstrip("\n").split("\t", 1)
            out[key] = value
    return out


def load_qrels(path):
    # Handles both TREC 4-column qrels and BEIR 3-column TSV, which ships with
    # a "query-id corpus-id score" header row.
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
    # Returns (corpus, queries, qrels) for a BEIR dataset already downloaded
    # into $NR_PROJECT/data/beir/<dataset>/.
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
    # Pyserini's JsonCollection wants {"id": ..., "contents": ...} per line.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for docid, text in corpus.items():
            f.write(json.dumps({"id": docid, "contents": text}) + "\n")
