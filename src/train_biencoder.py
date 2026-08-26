import os
import argparse

from transformers import AutoTokenizer, Trainer, TrainingArguments

from models import BiEncoder
from data import load_tsv, load_msmarco_pairs, TripleDataset, make_biencoder_collator


parser = argparse.ArgumentParser(description="Train Bi-Encoder Retriever")
parser.add_argument("--model", type=str, default="bert-base-uncased")
parser.add_argument("--negatives", type=str, choices=["in_batch"], default="in_batch")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--temperature", type=float, default=0.05)
parser.add_argument("--max_query_len", type=int, default=32)
parser.add_argument("--max_passage_len", type=int, default=128)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--data", type=str, default=os.path.join(os.environ.get("NR_PROJECT", "."), "data")
)
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(os.environ.get("NR_PROJECT", "."), "checkpoint"),
)
parser.add_argument("--resume", type=str, default=None)

args = parser.parse_args()

msmarco = os.path.join(args.data, "msmarco")

pairs = load_msmarco_pairs(os.path.join(msmarco, "qrels.train.tsv"))
queries = load_tsv(os.path.join(msmarco, "queries.train.tsv"))
collection = load_tsv(
    os.path.join(msmarco, "collection.tsv"), keep={pid for _, pid in pairs}
)

pairs = [(q, p) for q, p in pairs if q in queries and p in collection]
print(f"{len(pairs)} training pairs, {len(collection)} passages held")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = BiEncoder(args.model, temperature=args.temperature)
model.encoder.gradient_checkpointing_enable()

dataset = TripleDataset(pairs, queries, collection)


class BiEncoderTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model(queries=inputs["queries"], passages=inputs["passages"])
        return (loss, None) if return_outputs else loss


config = TrainingArguments(
    output_dir=f"{args.output}_biencoder_{args.negatives}",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type="linear",
    max_steps=args.max_steps,
    bf16=True,
    seed=args.seed,
    data_seed=args.seed,
    dataloader_num_workers=4,
    logging_steps=50,
    save_steps=5000,
    save_total_limit=3,
    report_to="none",
    remove_unused_columns=False,
    run_name=f"biencoder-{args.negatives}",
)

trainer = BiEncoderTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    data_collator=make_biencoder_collator(
        tokenizer, args.max_query_len, args.max_passage_len
    ),
)

if args.resume:
    trainer.train(resume_from_checkpoint=args.resume)
else:
    trainer.train()

model.encoder.save_pretrained(config.output_dir)
tokenizer.save_pretrained(config.output_dir)

print(f"\nTrained {trainer.state.global_step} steps -> {config.output_dir}")
