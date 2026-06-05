#!/usr/bin/env python3
"""
Обучение ruT5-small на задачу ITN с MLflow трекингом.

Использование:
  python scripts/train.py --epochs 3 --mlflow                    # полное обучение + mlflow
  python scripts/train.py --quick                                # быстрый тест (10 сек)
  python scripts/train.py --epochs 1 --max-samples 500 --mlflow  # 500 сэмплов + mlflow
"""

import argparse
import os
import sys
import time
import json
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sns.set_theme(style='whitegrid')


class ITNDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             truncation=True, padding='max_length', return_tensors='pt')
        tgt = self.tokenizer(self.targets[idx], max_length=self.max_len,
                             truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': src['input_ids'][0],
            'attention_mask': src['attention_mask'][0],
            'labels': tgt['input_ids'][0],
        }


class MLflowCallback(TrainerCallback):
    """Логирует метрики в MLflow на каждом шаге."""
    def __init__(self, mlflow):
        self.mlflow = mlflow

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=state.global_step)


def generate_predictions(model, tokenizer, texts, max_len=128):
    """Генерирует предсказания для списка текстов."""
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
            outputs = model.generate(**inputs, max_length=max_len, num_beams=2)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return preds


def evaluate_and_report(model, tokenizer, test_texts, test_targets, epoch, mlflow, output_dir):
    """Оценка на тестовом наборе: accuracy, classification report, графики."""
    print(f'\n  [Eval epoch {epoch}] Generating predictions...')
    preds = generate_predictions(model, tokenizer, test_texts)

    # Accuracy
    correct = sum(1 for p, t in zip(preds, test_targets) if p == t)
    acc = correct / len(test_texts) * 100
    print(f'  [Eval epoch {epoch}] Accuracy: {correct}/{len(test_texts)} = {acc:.2f}%')

    # Extract numbers from predictions and targets for digit-level analysis
    all_pred_digits = []
    all_true_digits = []
    for p, t in zip(preds, test_targets):
        p_nums = re.findall(r'\d+', p)
        t_nums = re.findall(r'\d+', t)
        for pn, tn in zip(p_nums, t_nums):
            all_pred_digits.append(len(pn))
            all_true_digits.append(len(tn))

    # Classification report (digit length accuracy)
    if all_true_digits and all_pred_digits:
        from sklearn.metrics import classification_report, accuracy_score
        # Map digit lengths to classes
        classes = sorted(set(all_true_digits + all_pred_digits))
        try:
            report = classification_report(
                all_true_digits, all_pred_digits,
                labels=classes,
                target_names=[f'{c}-digit' for c in classes],
                digits=3, zero_division=0
            )
            report_path = output_dir / 'reports' / f'epoch_{epoch}'
            report_path.mkdir(parents=True, exist_ok=True)
            with open(report_path / 'classification_report.txt', 'w') as f:
                f.write(report)
            mlflow.log_text(report, f'reports/epoch_{epoch}/classification_report.txt')
            print(f'  [Eval epoch {epoch}] Report saved')
        except Exception:
            pass

    # Confusion matrix heatmap
    if all_true_digits and all_pred_digits:
        fig, ax = plt.subplots(figsize=(6, 5))
        classes = sorted(set(all_true_digits + all_pred_digits))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(all_true_digits, all_pred_digits):
            if t in classes and p in classes:
                cm[classes.index(t)][classes.index(p)] += 1
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'{c}-d' for c in classes],
                    yticklabels=[f'{c}-d' for c in classes], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (epoch {epoch})')
        plt.tight_layout()
        plot_path = output_dir / 'plots' / f'epoch_{epoch}'
        plot_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path / 'confusion_matrix.png', dpi=120)
        plt.close()
        mlflow.log_figure(fig, f'plots/epoch_{epoch}/confusion_matrix.png')

    # Prediction samples (first 15)
    samples = []
    for i, (text, pred, target) in enumerate(zip(test_texts[:15], preds[:15], test_targets[:15])):
        samples.append(f'{i+1}. IN:  {text}\n   OUT: {pred}\n   GT:  {target}\n')
    samples_text = '\n'.join(samples)
    mlflow.log_text(samples_text, f'reports/epoch_{epoch}/prediction_samples.txt')

    return acc, preds


def main():
    parser = argparse.ArgumentParser(description='Train ruT5 for ITN')
    parser.add_argument('--data', default='data/synthetic.f', help='Train data')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-len', type=int, default=128, help='Max token length')
    parser.add_argument('--max-samples', type=int, default=0, help='Limit samples (0=all)')
    parser.add_argument('--output', default='models/ruT5-itn', help='Output dir')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 200 samples, 1 epoch')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    args = parser.parse_args()

    # ── MLflow setup ──
    mlflow = None
    if args.mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri('http://localhost:5001')
            mlflow.set_experiment('ruT5-itn')
            run_name = f'ruT5_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}'
            if args.max_samples:
                run_name += f'_s{args.max_samples}'
            mlflow.start_run(run_name=run_name)
            mlflow.log_param('model', 'cointegrated/ruT5-small')
            mlflow.log_param('model_params_M', 65)
        except Exception as e:
            print(f'  [WARN] MLflow не подключён ({e}), обучение без трекинга')
            mlflow = None

    output_dir = Path(args.output)

    print('Loading ruT5-small...')
    model_name = 'cointegrated/ruT5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.tie_word_embeddings = False
    print(f'  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')
    print(f'  Device: {"GPU" if torch.cuda.is_available() else "CPU"}')

    if mlflow:
        mlflow.log_param('device', 'GPU' if torch.cuda.is_available() else 'CPU')

    print(f'\nLoading data: {args.data}')
    df = pl.read_ipc(args.data)
    print(f'  Rows: {len(df)}')

    train_df = df.filter(pl.col('noise_level') == 'clean')
    if train_df.height == 0:
        train_df = df
    print(f'  Clean rows: {train_df.height}')

    if args.quick:
        print('  QUICK MODE: 200 samples, 1 epoch, max_len=32')
        args.epochs = 1
        args.max_len = 32
        args.max_samples = 200
        args.lr = 1e-4

    if args.max_samples > 0 and args.max_samples < train_df.height:
        train_df = train_df.head(args.max_samples)
        print(f'  Limited to {args.max_samples} samples')

    texts = train_df['task_text'].to_list()
    targets = train_df['ground_truth'].to_list()

    split = int(len(texts) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]
    train_targets, val_targets = targets[:split], targets[split:]

    if mlflow:
        mlflow.log_params({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'max_len': args.max_len,
            'max_samples': args.max_samples if args.max_samples else 'all',
            'dataset_size': len(train_df),
            'train_size': len(train_texts),
            'test_size': len(val_texts),
        })

    train_dataset = ITNDataset(train_texts, train_targets, tokenizer, args.max_len)
    val_dataset = ITNDataset(val_texts, val_targets, tokenizer, args.max_len)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        report_to='none',
        fp16=False,
    )

    callbacks = []
    if mlflow:
        callbacks.append(MLflowCallback(mlflow))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Metrics tracking
    train_losses = []
    eval_losses = []

    print(f'\nTraining ({args.epochs} epochs)...')
    start = time.time()

    for epoch in range(args.epochs):
        print(f'\n--- Epoch {epoch + 1}/{args.epochs} ---')
        train_result = trainer.train(resume_from_checkpoint=False)
        train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0
        train_losses.append(train_loss)

        # Evaluate
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get('eval_loss', 0)
        eval_losses.append(eval_loss)

        if mlflow:
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('eval_loss', eval_loss, step=epoch)

        # Full evaluation on test set
        acc, preds = evaluate_and_report(
            model, tokenizer, val_texts, val_targets,
            epoch + 1, mlflow, output_dir
        )

        if mlflow:
            mlflow.log_metric('test_accuracy', acc / 100, step=epoch + 1)

    elapsed = time.time() - start

    # Plot loss history
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses, 'o-', label='Train Loss')
        if eval_losses:
            ax.plot(range(1, len(eval_losses) + 1), eval_losses, 's-', label='Eval Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        plt.tight_layout()
        loss_plot = output_dir / 'plots' / 'training_history.png'
        loss_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(loss_plot, dpi=120)
        if mlflow:
            mlflow.log_figure(fig, 'plots/training_history.png')
        plt.close()

    # Save model
    print(f'\nSaving to {output_dir}')
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if mlflow:
        mlflow.log_artifacts(str(output_dir), artifact_path='model')
        mlflow.set_tag('status', 'completed')
        mlflow.set_tag('accuracy', f'{acc:.2f}%' if 'acc' in dir() else 'N/A')
        mlflow.end_run()

    print(f'\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)')


if __name__ == '__main__':
    main()
