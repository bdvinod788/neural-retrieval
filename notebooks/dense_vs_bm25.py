import os
import sys
import json

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]

ARMS = ["bm25", "dense_v1"]


def load(arm):
    out = {}
    for dataset in DATASETS:
        path = f"results/{arm}/{dataset}.json"
        if os.path.exists(path):
            out[dataset] = json.load(open(path))["metrics"]
    return out


results = {arm: load(arm) for arm in ARMS}

if not results["dense_v1"]:
    print("No dense_v1 results yet. Run: sbatch --export=ALL,ARM=dense scripts/eval.job")
    sys.exit(1)

if not results["bm25"]:
    print("No bm25 results. The week 1 baseline has to exist before this "
          "comparison means anything.")
    sys.exit(1)


def table(metric):
    print(f"\n=== {metric.upper()} ===\n")
    print(f"{'Dataset':<14}{'BM25':>10}{'Dense v1':>12}{'delta':>10}   winner")
    print("-" * 60)

    totals = {arm: [] for arm in ARMS}
    for dataset in DATASETS:
        bm25 = results["bm25"].get(dataset, {}).get(metric)
        dense = results["dense_v1"].get(dataset, {}).get(metric)
        if bm25 is None or dense is None:
            print(f"{dataset:<14}{'-':>10}{'-':>12}{'-':>10}   incomplete")
            continue

        totals["bm25"].append(bm25)
        totals["dense_v1"].append(dense)
        delta = dense - bm25
        print(f"{dataset:<14}{bm25:>10.4f}{dense:>12.4f}{delta:>+10.4f}"
              f"   {'dense' if delta > 0 else 'BM25'}")

    if totals["bm25"]:
        avg_b = sum(totals["bm25"]) / len(totals["bm25"])
        avg_d = sum(totals["dense_v1"]) / len(totals["dense_v1"])
        print("-" * 60)
        print(f"{'mean':<14}{avg_b:>10.4f}{avg_d:>12.4f}{avg_d - avg_b:>+10.4f}")


table("ndcg@10")
table("recall@100")


print("\n=== WHERE BM25 WINS ===\n")
losses = []
for dataset in DATASETS:
    bm25 = results["bm25"].get(dataset, {}).get("ndcg@10")
    dense = results["dense_v1"].get(dataset, {}).get("ndcg@10")
    if bm25 is not None and dense is not None and bm25 > dense:
        losses.append((dataset, bm25 - dense))

if losses:
    for dataset, gap in sorted(losses, key=lambda x: -x[1]):
        print(f"  {dataset:<14} BM25 ahead by {gap:.4f} nDCG@10")
else:
    print("  Dense wins everywhere. Treat that as a bug report, not a result:")
    print("  check that BEIR was never trained on and that the qrels match")
    print("  the split being retrieved.")
