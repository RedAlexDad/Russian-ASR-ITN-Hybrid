# Обучение ruT5-small

Fine-tuning ruT5-small (65M параметров) с LoRA для задачи ITN.
Подробнее о гибридном подходе: [docs/hybrid.md](hybrid.md)

## Требования

- Python 3.12+
- `requirements.txt` — все зависимости включая torch, transformers, peft
- Рекомендуется GPU (на CPU ~1 час на 3 эпохи)

## Датасет

Обучение на `data/synthetic.f` (16 500 строк, из них 7 392 clean).

| Параметр    | Значение                      |
| ----------- | ----------------------------- |
| Clean строк | 7 392                         |
| Train (90%) | ~6 653                        |
| Test (10%)  | ~739                          |
| Шум         | 9 108 строк (55%) — исключены |

Подробнее: [docs/data.md](data.md)

## LoRA

Вместо полного fine-tuning (65M параметров) обучаются только LoRA-адаптеры.

| Параметр         | Значение           |
| ---------------- | ------------------ |
| Ранг (r)         | 8                  |
| Alpha            | 16                 |
| Target modules   | q, v, k, o, wi, wo |
| Trainable params | 0.9M (1.4%)        |
| Dropout          | 0.1                |
| Базовая модель   | frozen             |

## Команды

```bash
# Полное обучение (рекомендуется)
make train-local EPOCHS=3

# Быстрый тест (200 samples, ~10 сек)
make train-quick

# С кастомными параметрами
make train-local EPOCHS=5 BATCH_SIZE=4 MAX_SAMPLES=2000

# Через Docker
make train EPOCHS=3
```

## MLflow

```bash
make mlflow-up        # Запуск UI на :5001
make train-local      # Обучение с авто-логированием
make mlflow-down      # Остановка
```

MLflow логирует:

- Parameters: epochs, batch_size, lr, model_name
- Metrics: train_loss, eval_loss, test_accuracy по эпохам
- Artifacts: confusion matrix, classification report, prediction samples
- Model weights

## Гиперпараметры

| Параметр      | Значение | Описание                       |
| ------------- | -------- | ------------------------------ |
| learning_rate | 5e-5     | Скорость обучения              |
| batch_size    | 8        | Размер батча                   |
| max_length    | 128      | Макс. длина последовательности |
| warmup_steps  | 100      | Разогрев LR scheduler          |
| weight_decay  | 0.01     | L2 регуляризация               |
| LoRA r        | 8        | Ранг адаптера                  |

## Метрики

Текущая accuracy rule-based парсера: **97.6%** на calibration.f.
Ожидаемая accuracy гибрида после обучения: **98-99%**.

Подробнее о парсере: [docs/parser.md](parser.md)
