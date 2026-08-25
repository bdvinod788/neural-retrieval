import os
import sys
import json

# Published BM25 nDCG@10 on BEIR (Thakur et al. 2021, Anserini/Pyserini flat
# index). VERIFY these against Table 2 of the BEIR paper before trusting the
# gate -- BM25 numbers vary by a few points between implementations, and
# Elasticsearch multi-field figures are NOT comparable to a flat Pyserini
# index. The gate is only as good as this table.
PUBLISHED = {
    "scifact": 0.665,
    "nfcorpus": 0.325,
    "arguana": 0.315,
    "scidocs": 0.158,
    "fiqa": 0.236,
    "trec-covid": 0.656,
}

TOLERANCE = 0.01  # "within about a point"

results = {}
for dataset in PUBLISHED:
    path = f"results/bm25/{dataset}.json"
    if os.path.exists(path):
        results[dataset] = json.load(open(path))


print("=== WEEK 1 GATE: BM25 vs PUBLISHED BEIR ===\n")
print(f"{'Dataset':<14}{'ours':>10}{'published':>12}{'delta':>10}   status")
print("-" * 60)

failed = []
for dataset, reference in PUBLISHED.items():
    if dataset not in results:
        print(f"{dataset:<14}{'-':>10}{reference:>12.3f}{'-':>10}   NOT RUN")
        failed.append(dataset)
        continue

    ours = results[dataset]["metrics"]["ndcg@10"]
    delta = ours - reference
    ok = abs(delta) <= TOLERANCE
    if not ok:
        failed.append(dataset)
    print(f"{dataset:<14}{ours:>10.4f}{reference:>12.3f}{delta:>+10.4f}   {'PASS' if ok else 'FAIL'}")


print("\n=== ALL METRICS ===\n")
print(f"{'Dataset':<14}{'nDCG@10':>10}{'R@100':>10}{'MAP':>10}{'MRR@10':>10}{'queries':>10}")
print("-" * 64)
for dataset in PUBLISHED:
    if dataset not in results:
        continue
    m = results[dataset]["metrics"]
    n = results[dataset]["n_queries"]
    print(f"{dataset:<14}{m['ndcg@10']:>10.4f}{m['recall@100']:>10.4f}"
          f"{m['map']:>10.4f}{m['mrr@10']:>10.4f}{n:>10}")


if failed:
    print(f"\nGATE FAILED on: {', '.join(failed)}")
    print("Do not build anything on top of this baseline until it passes.")
    print("A weak BM25 invalidates every comparison made afterwards.")
    sys.exit(1)

print("\nGATE PASSED. Baseline is trustworthy; week 2 can start.")
