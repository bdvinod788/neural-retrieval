import os
import glob
import argparse

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer

from models import BiEncoder
from data import load_beir, load_tsv


parser = argparse.ArgumentParser(description="Encode a Corpus and Build a FAISS Index")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataset", type=str, default="scifact")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--max_passage_len", type=int, default=128)
parser.add_argument("--shard_size", type=int, default=500000)
parser.add_argument("--index_type", type=str, choices=["flat_ip", "ivf"], default="flat_ip")
parser.add_argument("--nlist", type=int, default=4096, help="ivf only")
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument("--output", type=str, default=os.environ.get("NR_PROJECT", "."))

args = parser.parse_args()

emb_dir = os.path.join(args.output, "embeddings", args.dataset)
index_path = os.path.join(args.output, "indexes", "dense", f"{args.dataset}.faiss")
os.makedirs(emb_dir, exist_ok=True)
os.makedirs(os.path.dirname(index_path), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = BiEncoder(args.checkpoint)
model.cuda().eval()

if args.dataset == "msmarco":
    corpus = load_tsv(os.path.join(args.data, "msmarco", "collection.tsv"))
else:
    corpus, _, _ = load_beir(
        args.dataset, data_dir=os.path.join(args.data, "beir", args.dataset)
    )

docids = list(corpus.keys())
texts = [corpus[d] for d in docids]

with open(os.path.join(emb_dir, "docids.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(docids))

print(f"{len(texts)} passages -> {emb_dir}")


def encode(batch):
    inputs = tokenizer(
        batch, padding=True, truncation=True,
        max_length=args.max_passage_len, return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model.encode(**inputs).float().cpu().numpy()


for start in range(0, len(texts), args.shard_size):
    shard_id = start // args.shard_size
    shard_path = os.path.join(emb_dir, f"shard_{shard_id:04d}.npy")
    if os.path.exists(shard_path):
        print(f"shard {shard_id} exists, skipping")
        continue

    shard_texts = texts[start:start + args.shard_size]
    vectors = []
    for i in range(0, len(shard_texts), args.batch_size):
        vectors.append(encode(shard_texts[i:i + args.batch_size]))
        if (i // args.batch_size) % 100 == 0:
            print(f"  shard {shard_id}: {i}/{len(shard_texts)}")

    np.save(shard_path, np.concatenate(vectors))
    print(f"shard {shard_id} written")

shards = sorted(glob.glob(os.path.join(emb_dir, "shard_*.npy")))
embeddings = np.concatenate([np.load(s) for s in shards])

if len(embeddings) != len(docids):
    raise SystemExit(
        f"FATAL: {len(embeddings)} vectors but {len(docids)} docids. "
        f"Delete {emb_dir} and re-encode; a stale shard is present."
    )

if args.index_type == "flat_ip":
    index = faiss.IndexFlatIP(embeddings.shape[1])
else:
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    index = faiss.IndexIVFFlat(
        quantizer, embeddings.shape[1], args.nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.train(embeddings)

index.add(embeddings)
faiss.write_index(index, index_path)

print(f"indexed {index.ntotal} vectors ({args.index_type}) -> {index_path}")
