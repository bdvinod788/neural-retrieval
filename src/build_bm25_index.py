import os
import sys
import argparse
import subprocess

from data import load_beir, load_tsv, write_jsonl_corpus
from utils import BEIR_DATASETS


parser = argparse.ArgumentParser(description="Build Pyserini BM25 Indexes")
parser.add_argument("--datasets", type=str, nargs="+", default=BEIR_DATASETS)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(os.environ.get("NR_PROJECT", "."), "indexes", "bm25"),
)

args = parser.parse_args()


def to_jsonl(dataset):
    jsonl_dir = os.path.join(args.data, "jsonl", dataset)
    if os.path.exists(os.path.join(jsonl_dir, "docs.jsonl")):
        print(f"  jsonl already built for {dataset}")
        return jsonl_dir

    if dataset == "msmarco":
        corpus = load_tsv(os.path.join(args.data, "msmarco", "collection.tsv"))
    else:
        corpus, _, _ = load_beir(dataset, data_dir=os.path.join(args.data, "beir", dataset))

    write_jsonl_corpus(corpus, jsonl_dir)
    return jsonl_dir


for dataset in args.datasets:
    print(f"Indexing {dataset}...")
    jsonl_dir = to_jsonl(dataset)
    index_dir = os.path.join(args.output, dataset)

    subprocess.run(
        [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", jsonl_dir,
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(args.threads),
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ],
        check=True,
    )
    print(f"  -> {index_dir}")
