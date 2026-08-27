import os
import argparse

from data import load_beir
from utils import write_run


parser = argparse.ArgumentParser(description="Run Retrieval")
parser.add_argument("--arm", type=str, choices=["bm25", "dense"], default="bm25")
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--top_k", type=int, default=1000)
parser.add_argument("--k1", type=float, default=0.9)
parser.add_argument("--b", type=float, default=0.4)
parser.add_argument("--threads", type=int, default=8, help="bm25 only")
parser.add_argument("--checkpoint", type=str, default=None, help="dense only")
parser.add_argument("--batch_size", type=int, default=256, help="dense only")
parser.add_argument("--max_query_len", type=int, default=32, help="dense only")
parser.add_argument("--nprobe", type=int, default=64, help="dense only, ivf")
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

if args.arm == "dense" and not args.checkpoint:
    raise SystemExit("--checkpoint is required for --arm dense")

project = os.environ.get("NR_PROJECT", ".")

_, queries, _ = load_beir(
    args.dataset, args.split, os.path.join(args.data, "beir", args.dataset)
)


def bm25_search():
    os.environ.setdefault("OPENAI_API_KEY", "unused")

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(os.path.join(project, "indexes", "bm25", args.dataset))
    searcher.set_bm25(args.k1, args.b)

    qids = list(queries.keys())
    run = {}

    for start in range(0, len(qids), 100):
        batch = qids[start:start + 100]
        hits = searcher.batch_search(
            [queries[q] for q in batch], batch, k=args.top_k, threads=args.threads
        )
        for qid in batch:
            run[qid] = [
                (h.docid, h.score) for h in hits[qid] if h.docid != qid
            ]

    return run


def dense_search():
    import torch
    import faiss
    from transformers import AutoTokenizer

    from models import BiEncoder

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = BiEncoder(args.checkpoint)
    model.cuda().eval()

    index = faiss.read_index(
        os.path.join(project, "indexes", "dense", f"{args.dataset}.faiss")
    )
    if hasattr(index, "nprobe"):
        index.nprobe = args.nprobe

    with open(
        os.path.join(project, "embeddings", args.dataset, "docids.txt"),
        encoding="utf-8",
    ) as f:
        docids = f.read().split("\n")

    if index.ntotal != len(docids):
        raise SystemExit(
            f"FATAL: index has {index.ntotal} vectors but docids.txt has "
            f"{len(docids)} ids. Re-encode {args.dataset}."
        )

    qids = list(queries.keys())
    run = {}

    for start in range(0, len(qids), args.batch_size):
        batch = qids[start:start + args.batch_size]
        inputs = tokenizer(
            [queries[q] for q in batch], padding=True, truncation=True,
            max_length=args.max_query_len, return_tensors="pt",
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vectors = model.encode(**inputs).float().cpu().numpy()

        scores, neighbours = index.search(vectors, args.top_k)
        for qid, row_scores, row_ids in zip(batch, scores, neighbours):
            run[qid] = [
                (docids[i], float(s))
                for i, s in zip(row_ids, row_scores)
                if i >= 0 and docids[i] != qid
            ]

    return run


run = bm25_search() if args.arm == "bm25" else dense_search()

output = args.output or os.path.join(project, "runs", f"{args.arm}_{args.dataset}.trec")
write_run(output, run, f"{args.arm}_{args.dataset}")

print(f"Wrote {len(run)} queries to {output}")
