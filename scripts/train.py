#!/usr/bin/env python3
"""
Обучение ruT5-small на задачу ITN.

Датасет: data/synthetic.f (или data/real.f, или оба).
Процесс:
  1. Загрузка токенизатора и модели
  2. Подготовка датасета: task_text -> input, ground_truth -> target
  3. Fine-tune на CPU/GPU
  4. Сохранение в models/ruT5-itn

Использование:
  python scripts/train.py [--data data/synthetic.f] [--epochs 3] [--lr 5e-5]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ITNDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            self.targets[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": src["input_ids"][0],
            "attention_mask": src["attention_mask"][0],
            "labels": tgt["input_ids"][0],
        }


def compute_accuracy(eval_pred):
    """Подсчёт accuracy на валидации (точное совпадение токенов)."""
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    mask = labels != -100
    correct = (preds == labels) & mask
    acc = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Train ruT5 for ITN")
    parser.add_argument("--data", default="data/synthetic.f", help="Train data")
    parser.add_argument("--eval-data", default="data/calibration.f", help="Eval data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length")
    parser.add_argument("--output", default="models/ruT5-itn", help="Output dir")
    args = parser.parse_args()

    print(f"Loading ruT5-small...")
    model_name = "cointegrated/ruT5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    print(f"\nLoading data: {args.data}")
    df = pl.read_ipc(args.data)
    print(f"  Rows: {len(df)}")

    # Filter only clean data for training (noisy data is too corrupted)
    train_df = df.filter(pl.col("noise_level") == "clean")
    if train_df.height == 0:
        train_df = df
    print(f"  Clean rows for training: {train_df.height}")

    texts = train_df["task_text"].to_list()
    targets = train_df["ground_truth"].to_list()

    # Split train/val
    split = int(len(texts) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    train_dataset = ITNDataset(train_texts, train_targets, tokenizer, args.max_len)
    val_dataset = ITNDataset(val_texts, val_targets, tokenizer, args.max_len)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # tokenizer removed in newer transformers
    )

    print(f"\nTraining ({args.epochs} epochs)...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\nSaving to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
