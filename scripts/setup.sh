#!/bin/bash
# One-time environment build on CARC. Run from a login node.

module purge
module load conda

conda create -y -p /scratch1/bandrede/envs/neural-retrieval python=3.11
source activate /scratch1/bandrede/envs/neural-retrieval

# Pyserini wraps lucene, so it needs a JVM.
conda install -y -c conda-forge openjdk=21 maven

# torch and faiss via conda: the pip faiss-gpu wheel is unreliable.
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1
conda install -y -c pytorch faiss-gpu

pip install -r requirements.txt

export NR_PROJECT=/scratch1/bandrede/neural-retrieval
mkdir -p $NR_PROJECT/{logs,data,indexes,runs,embeddings}

echo "Env ready. Add to your .bashrc:"
echo "  export NR_PROJECT=/scratch1/bandrede/neural-retrieval"
echo
echo "Then copy the repo to \$NR_PROJECT and run, in order:"
echo "  sbatch scripts/build_index.job                    # week 1"
echo "  python src/download_data.py --msmarco             # week 2 training data"
echo "  sbatch scripts/train_biencoder.job                # week 2"
echo "  sbatch scripts/encode_corpus.job                  # week 2"
echo "  sbatch --export=ARM=dense scripts/eval.job        # week 2"
