import os
import argparse

import ir_measures
from ir_measures import parse_measure

from data import load_beir
from utils import read_run, write_results, config_hash, METRIC_KEYS


MEASURES = {"ndcg@10": "nDCG@10", "recall@100": "R@100", "map": "AP", "mrr@10": "RR@10"}


parser = argparse.ArgumentParser(description="Score a TREC Run File")
parser.add_argument("--run", type=str, required=True)
parser.add_argument("--arm", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--model", type=str, default="bm25")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument("--output", type=str, default=None)

args = parser.parse_args()

_, _, qrels = load_beir(
    args.dataset, args.split, os.path.join(args.data, "beir", args.dataset)
)

run = read_run(args.run)
scored = {qid: dict(ranked) for qid, ranked in run.items()}

measures = [parse_measure(MEASURES[k]) for k in METRIC_KEYS]
aggregate = ir_measures.calc_aggregate(measures, qrels, scored)
metrics = {k: float(aggregate[m]) for k, m in zip(METRIC_KEYS, measures)}

output = args.output or os.path.join("results", args.arm, f"{args.dataset}.json")

write_results(
    output,
    run_id=f"{args.arm}_{args.dataset}",
    arm=args.arm,
    dataset=args.dataset,
    split=args.split,
    model=args.model,
    checkpoint=args.checkpoint,
    seed=args.seed,
    metrics=metrics,
    n_queries=len(run),
    cfg_hash=config_hash(args),
)

print(f"{args.arm:<8} {args.dataset:<12} " + "  ".join(
    f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS
))
