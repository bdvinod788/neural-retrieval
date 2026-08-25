import os
import sys
import json

FALLBACK = {
    "trec-covid": (0.5947, 0.1091),
    "nfcorpus": (0.3218, 0.2457),
    "fiqa": (0.2361, 0.5395),
    "arguana": (0.3970, 0.9324),
    "scidocs": (0.1490, 0.3477),
    "scifact": (0.6789, 0.9253),
}

TOLERANCE = 0.01  

DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]


def load_reference():
    try:
        import yaml
        import pyserini

        path = os.path.join(os.path.dirname(pyserini.__file__), "2cr", "beir.yaml")
        doc = yaml.safe_load(open(path))
        for condition in doc["conditions"]:
            if condition["name"] == "bm25-flat":
                out = {}
                for entry in condition["datasets"]:
                    scores = entry["scores"][0]
                    out[entry["dataset"]] = (scores["nDCG@10"], scores["R@100"])
                return out, f"pyserini {path}"
    except Exception as exc:
        print(f"(could not read pyserini reference: {exc})")
    return FALLBACK, "built-in fallback"


reference, source = load_reference()

results = {}
for dataset in DATASETS:
    path = f"results/bm25/{dataset}.json"
    if os.path.exists(path):
        results[dataset] = json.load(open(path))


print("=== WEEK 1 GATE: BM25 vs PYSERINI bm25-flat ===")
print(f"reference: {source}\n")
print(f"{'Dataset':<14}{'ours':>10}{'expected':>11}{'delta':>10}"
      f"{'R@100':>10}{'exp':>10}   status")
print("-" * 76)

failed = []
for dataset in DATASETS:
    if dataset not in reference:
        print(f"{dataset:<14}{'?':>10}   no reference")
        failed.append(dataset)
        continue

    ref_ndcg, ref_recall = reference[dataset]

    if dataset not in results:
        print(f"{dataset:<14}{'-':>10}{ref_ndcg:>11.4f}{'-':>10}"
              f"{'-':>10}{ref_recall:>10.4f}   NOT RUN")
        failed.append(dataset)
        continue

    metrics = results[dataset]["metrics"]
    ndcg = metrics["ndcg@10"]
    recall = metrics["recall@100"]
    delta = ndcg - ref_ndcg
    ok = abs(delta) <= TOLERANCE
    if not ok:
        failed.append(dataset)

    print(f"{dataset:<14}{ndcg:>10.4f}{ref_ndcg:>11.4f}{delta:>+10.4f}"
          f"{recall:>10.4f}{ref_recall:>10.4f}   {'PASS' if ok else 'FAIL'}")


print("\n=== ALL METRICS ===\n")
print(f"{'Dataset':<14}{'nDCG@10':>10}{'R@100':>10}{'MAP':>10}"
      f"{'MRR@10':>10}{'queries':>10}")
print("-" * 64)
for dataset in DATASETS:
    if dataset not in results:
        continue
    m = results[dataset]["metrics"]
    print(f"{dataset:<14}{m['ndcg@10']:>10.4f}{m['recall@100']:>10.4f}"
          f"{m['map']:>10.4f}{m['mrr@10']:>10.4f}"
          f"{results[dataset]['n_queries']:>10}")


if failed:
    print(f"\nGATE FAILED on: {', '.join(failed)}")
    print("Do not build anything on top of this baseline until it passes.")
    print("A weak BM25 invalidates every comparison made afterwards.")
    sys.exit(1)

print("\nGATE PASSED. Baseline reproduces pyserini bm25-flat; week 2 can start.")
