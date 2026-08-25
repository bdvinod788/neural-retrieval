import os
import shutil
import zipfile
import argparse
import urllib.request

from utils import BEIR_DATASETS


BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
MSMARCO_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking"


parser = argparse.ArgumentParser(description="Download BEIR and MS MARCO")
parser.add_argument("--datasets", type=str, nargs="+", default=BEIR_DATASETS)
parser.add_argument("--msmarco", action="store_true", help="also fetch MS MARCO (week 2)")
parser.add_argument(
    "--output", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)

args = parser.parse_args()

beir_dir = os.path.join(args.output, "beir")
os.makedirs(beir_dir, exist_ok=True)


def progress(count, block_size, total_size):
    if total_size > 0 and count % 200 == 0:
        done = count * block_size / total_size * 100
        print(f"    {min(done, 100):.1f}%", end="\r")


for dataset in args.datasets:
    target = os.path.join(beir_dir, dataset)
    if os.path.exists(os.path.join(target, "corpus.jsonl")):
        print(f"{dataset} already present")
        continue

    archive = os.path.join(beir_dir, f"{dataset}.zip")
    print(f"Downloading {dataset}...")
    urllib.request.urlretrieve(f"{BEIR_URL}/{dataset}.zip", archive, progress)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(beir_dir)
    os.remove(archive)

    corpus = os.path.join(target, "corpus.jsonl")
    qrels = os.path.join(target, "qrels", "test.tsv")
    print(f"  corpus: {sum(1 for _ in open(corpus, encoding='utf-8'))} docs")
    print(f"  qrels:  {'ok' if os.path.exists(qrels) else 'MISSING test.tsv'}")

if not args.msmarco:
    raise SystemExit

msmarco = os.path.join(args.output, "msmarco")
os.makedirs(msmarco, exist_ok=True)

for name in ["collection.tar.gz", "queries.tar.gz", "qrels.train.tsv"]:
    target = os.path.join(msmarco, name)
    if os.path.exists(target):
        print(f"{name} already present")
        continue
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(f"{MSMARCO_URL}/{name}", target, progress)

for name in ["collection.tar.gz", "queries.tar.gz"]:
    shutil.unpack_archive(os.path.join(msmarco, name), msmarco)

print(f"\nMS MARCO extracted to {msmarco}")
