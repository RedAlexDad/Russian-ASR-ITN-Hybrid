#!/usr/bin/env python3
"""
Обучение ruT5-small на задачу ITN с MLflow трекингом.

Использование:
  python scripts/train.py --epochs 3                   # полное обучение
  python scripts/train.py --quick                      # быстрый тест (10 сек)
  python scripts/train.py --epochs 1 --max-samples 500 # 500 сэмплов
"""

import argparse
import os
import sys
import time
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
import warnings
warnings.filterwarnings('ignore', message='Creating a tensor from a list of numpy.ndarrays')
sns.set_theme(style='whitegrid')


class ITNDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids = []
        self.attention_masks = []
        self.labels_list = []
        for src, tgt in zip(texts, targets):
            s = tokenizer(src, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
            t = tokenizer(tgt, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
            lbl = t['input_ids'][0].clone()
            lbl[lbl == 0] = -100
            self.input_ids.append(s['input_ids'][0])
            self.attention_masks.append(s['attention_mask'][0])
            self.labels_list.append(lbl)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels_list[idx],
        }


class MLflowCallback(TrainerCallback):
    """Логирует метрики в MLflow на каждом шаге и по эпохам."""
    def __init__(self, mlflow):
        self.mlflow = mlflow

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = int(state.epoch) if state.epoch else 0
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.mlflow.log_metric(k, v, step=step)
                    self.mlflow.log_metric(f'{k}_epoch', v, step=epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        self.mlflow.log_metric('epoch', epoch, step=epoch)
        self.mlflow.log_metric('train_speed_it_per_sec', state.global_step / (time.time() - self._train_start), step=epoch)

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 1:
            model = kwargs.get('model')
            if model is not None:
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.mlflow.log_metric(f'grad_norm/{name}', param.grad.norm().item(), step=state.global_step)


def generate_predictions(model, tokenizer, texts, max_len=128):
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_length=max_len, num_beams=2)
            preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return preds


def evaluate_and_report(model, tokenizer, test_texts, test_targets, epoch, mlflow, output_dir):
    """Оценка на тесте: accuracy, classification report, confusion matrix."""
    preds = generate_predictions(model, tokenizer, test_texts)

    correct = sum(1 for p, t in zip(preds, test_targets) if p == t)
    acc = correct / len(test_texts) * 100

    # ── Character Error Rate (CER) ──
    total_chars, total_errors = 0, 0
    for p, t in zip(preds, test_targets):
        maxlen = max(len(p), len(t))
        if maxlen == 0: continue
        total_chars += maxlen
        total_errors += sum(1 for a, b in zip(p, t) if a != b) + abs(len(p) - len(t))
    cer = total_errors / total_chars if total_chars > 0 else 0

    # ── Number-level accuracy ──
    num_correct, num_total = 0, 0
    for p, t in zip(preds, test_targets):
        pn = re.findall(r'\d+', p)
        tn = re.findall(r'\d+', t)
        num_total += len(tn)
        for a, b in zip(pn, tn):
            if a == b:
                num_correct += 1
    num_acc = num_correct / num_total if num_total > 0 else 0

    # ── Accuracy by input length bucket ──
    buckets = {'short (1-20)': [], 'medium (21-40)': [], 'long (41+)': []}
    for t, p, g in zip(test_texts, preds, test_targets):
        key = 'short (1-20)' if len(t) <= 20 else ('medium (21-40)' if len(t) <= 40 else 'long (41+)')
        buckets[key].append(p == g)
    bucket_repr = ', '.join(f'{k}: {sum(v)/len(v)*100:.0f}%({len(v)})' for k, v in buckets.items() if v)

    all_true, all_pred = [], []
    for p, t in zip(preds, test_targets):
        pn = re.findall(r'\d+', p); tn = re.findall(r'\d+', t)
        for a, b in zip(pn, tn):
            all_true.append(len(a)); all_pred.append(len(b))

    if mlflow:
        mlflow.log_metrics({
            'exact_accuracy': acc / 100,
            'char_error_rate': cer,
            'number_accuracy': num_acc,
        }, step=epoch)

        if all_true:
            from sklearn.metrics import classification_report
            classes = sorted(set(all_true + all_pred))
            try:
                report = classification_report(all_true, all_pred, labels=classes,
                                               target_names=[f'{c}-d' for c in classes],
                                               digits=3, zero_division=0)
                mlflow.log_text(report, f'reports/epoch_{epoch}/classification_report.txt')
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(6, 5))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(all_true, all_pred):
                if t in classes and p in classes:
                    cm[classes.index(t)][classes.index(p)] += 1
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f'{c}-d' for c in classes],
                        yticklabels=[f'{c}-d' for c in classes], ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix (epoch {epoch})')
            plt.tight_layout()
            mlflow.log_figure(fig, f'plots/epoch_{epoch}/confusion_matrix.png')
            plt.close()

        samples = []
        for i, (t, p, g) in enumerate(zip(test_texts[:15], preds[:15], test_targets[:15]), 1):
            samples.append(f'{i}. IN:  {t}\n   OUT: {p}\n   GT:  {g}\n')
        mlflow.log_text('\n'.join(samples), f'reports/epoch_{epoch}/prediction_samples.txt')
        mlflow.log_text(bucket_repr, f'reports/epoch_{epoch}/accuracy_by_length.txt')

    return acc, preds


def main():
    parser = argparse.ArgumentParser(description='Train ruT5 for ITN')
    parser.add_argument('--data', default='data/synthetic.f')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output', default='models/ruT5-itn')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--mlflow', action='store_true')
    parser.add_argument('--lora', action='store_true', default=True,
                        help='Use LoRA (default). Pass --no-lora to disable')
    parser.add_argument('--no-lora', dest='lora', action='store_false')
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='FP16 mixed precision (осторожно: может давать NaN на некоторых GPU)')
    args = parser.parse_args()

    # ── MLflow ──
    mlflow = None
    if args.mlflow:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            mlflow_ok = sock.connect_ex(('localhost', 5001)) == 0
            sock.close()
            if not mlflow_ok:
                raise ConnectionRefusedError('MLflow сервер недоступен')
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri('http://localhost:5001')
            mlflow.set_experiment('ruT5-itn')
            name = f'ruT5_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}'
            if args.max_samples: name += f'_s{args.max_samples}'
            os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'
            os.environ['MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL'] = '10'
            mlflow.start_run(run_name=name)
            mlflow.log_param('model', 'cointegrated/ruT5-small')
            mlflow.log_param('model_params_M', 65)
        except Exception as e:
            print(f'  [WARN] MLflow недоступен: {e}')
            mlflow = None

    out_dir = Path(args.output)

    print('Loading ruT5-small...')
    tokenizer = AutoTokenizer.from_pretrained('cointegrated/ruT5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('cointegrated/ruT5-small')
    model.config.tie_word_embeddings = False
    print(f'  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

    # LoRA — обучаем только адаптеры (2-4M параметров вместо 65M)
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=['q', 'v', 'k', 'o', 'wi', 'wo'],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'  LoRA: r={args.lora_r}, trainable={trainable/1e6:.1f}M/{total/1e6:.0f}M')

    dev = 'GPU' if torch.cuda.is_available() else 'CPU'
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'  Device: {dev} ({dev_name})')

    print(f'\nLoading data: {args.data}')
    df = pl.read_ipc(args.data)
    train_df = df.filter(pl.col('noise_level') == 'clean')
    if train_df.height == 0: train_df = df
    print(f'  Clean rows: {train_df.height}')

    if args.quick:
        args.epochs, args.max_len, args.max_samples, args.lr = 1, 32, 200, 1e-4
        args.fp16 = False

    if args.max_samples and args.max_samples < train_df.height:
        train_df = train_df.head(args.max_samples)
        print(f'  Limited to {args.max_samples}')

    texts, targets = train_df['task_text'].to_list(), train_df['ground_truth'].to_list()
    split = int(len(texts) * 0.9)
    train_texts, train_targets = texts[:split], targets[:split]
    val_texts, val_targets = texts[split:], targets[split:]

    if mlflow:
        params = {
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'max_len': args.max_len,
            'max_samples': args.max_samples or 'all',
            'train_size': len(train_texts), 'test_size': len(val_texts),
            'lora_r': args.lora_r, 'lora_alpha': args.lora_alpha,
            'lora_trainable_M': f'{trainable/1e6:.1f}' if args.lora else 'full',
            'device': dev_name, 'fp16': args.fp16,
        }
        mlflow.log_params(params)
        src_lens = [len(t) for t in train_texts]
        tgt_lens = [len(g) for g in train_targets]
        mlflow.log_metrics({
            'src_len_mean': np.mean(src_lens), 'src_len_std': np.std(src_lens),
            'tgt_len_mean': np.mean(tgt_lens), 'tgt_len_std': np.std(tgt_lens),
        })

    train_ds = ITNDataset(train_texts, train_targets, tokenizer, args.max_len)
    val_ds = ITNDataset(val_texts, val_targets, tokenizer, args.max_len)

    pin = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
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
        dataloader_num_workers=0,
        dataloader_pin_memory=pin,
        fp16=args.fp16,
    )

    callbacks = []
    if mlflow:
        callbacks.append(MLflowCallback(mlflow))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator, callbacks=callbacks,
    )

    print(f'\nTraining ({args.epochs} epochs)...')
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    # Final evaluation
    print(f'\nFinal evaluation...')
    final_acc, _ = evaluate_and_report(
        model, tokenizer, val_texts, val_targets,
        args.epochs, mlflow, out_dir
    )
    if mlflow:
        mlflow.log_metric('test_accuracy', final_acc / 100, step=args.epochs)
    print(f'  Final accuracy: {final_acc:.2f}%')

    # Training loss plot
    if mlflow:
        log_history = train_result.training_loss if hasattr(train_result, 'training_loss') else None
        if isinstance(log_history, (list, tuple)):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(log_history, 'o-', label='Train Loss')
            ax.set_xlabel('Step'); ax.set_ylabel('Loss')
            ax.set_title('Training Loss'); ax.legend()
            plt.tight_layout()
            mlflow.log_figure(fig, 'plots/training_loss.png')
            plt.close()

    # Save model
    print(f'\nSaving to {out_dir}')
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if mlflow:
        mlflow.log_artifacts(str(out_dir), artifact_path='model')
        mlflow.set_tag('status', 'completed')
        mlflow.end_run()

    print(f'\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)')


if __name__ == '__main__':
    main()
