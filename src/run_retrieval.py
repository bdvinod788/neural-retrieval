import os
import argparse

from data import load_beir
from utils import write_run


parser = argparse.ArgumentParser(description="Run Retrieval")
parser.add_argument("--arm", type=str, choices=["bm25"], default="bm25")
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--top_k", type=int, default=1000)
parser.add_argument("--k1", type=float, default=0.9)
parser.add_argument("--b", type=float, default=0.4)
parser.add_argument("--threads", type=int, default=8)
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

project = os.environ.get("NR_PROJECT", ".")

_, queries, _ = load_beir(
    args.dataset, args.split, os.path.join(args.data, "beir", args.dataset)
)


def bm25_search():
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(os.path.join(project, "indexes", "bm25", args.dataset))
    searcher.set_bm25(args.k1, args.b)

    qids = list(queries.keys())
    run = {}

    # batch_search parallelises across the JVM's thread pool.
    for start in range(0, len(qids), 100):
        batch = qids[start:start + 100]
        hits = searcher.batch_search(
            [queries[q] for q in batch], batch, k=args.top_k, threads=args.threads
        )
        for qid in batch:
            run[qid] = [(h.docid, h.score) for h in hits[qid]]

    return run


run = bm25_search()

output = args.output or os.path.join(project, "runs", f"{args.arm}_{args.dataset}.trec")
write_run(output, run, f"{args.arm}_{args.dataset}")

print(f"Wrote {len(run)} queries to {output}")
