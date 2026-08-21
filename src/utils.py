import os
import json
import hashlib
import datetime


METRIC_KEYS = ["ndcg@10", "recall@100", "map", "mrr@10"]

BEIR_DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]


def config_hash(args):
    blob = json.dumps(vars(args), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def write_run(path, run, tag):
    # TREC run format: qid Q0 docid rank score tag
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for qid, ranked in run.items():
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


def read_run(path):
    run = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, _, score, _ = line.split()
            run.setdefault(qid, []).append((docid, float(score)))
    for qid in run:
        run[qid].sort(key=lambda x: -x[1])
    return run


def write_results(path, run_id, arm, dataset, split, model, checkpoint, seed,
                  metrics, n_queries, latency_ms=None, cfg_hash=""):
    # Frozen schema. Fixed in week 1 and not changed afterwards: every arm
    # writes this exact shape, so one analysis script handles all of them and
    # adding an arm never touches the analysis code.
    record = {
        "run_id": run_id,
        "arm": arm,
        "dataset": dataset,
        "split": split,
        "model": model,
        "checkpoint": checkpoint,
        "seed": seed,
        "metrics": {k: float(metrics.get(k, 0.0)) for k in METRIC_KEYS},
        "n_queries": n_queries,
        "latency_ms": latency_ms or {"p50": 0.0, "p95": 0.0},
        "config_hash": cfg_hash,
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def load_results(results_dir):
    records = []
    for root, _, files in os.walk(results_dir):
        for name in sorted(files):
            if name.endswith(".json"):
                with open(os.path.join(root, name)) as f:
                    records.append(json.load(f))
    return records
