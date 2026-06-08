# Обучение ruT5-small

Fine-tuning ruT5-small (65M параметров) с LoRA для задачи ITN.
Подробнее о гибридном подходе: [docs/hybrid.md](hybrid.md)

## Требования

- Python 3.12+
- `requirements.txt` — все зависимости включая torch, transformers, peft
- Рекомендуется GPU (на CPU ~1 час на 3 эпохи)

## Датасет

### Clean обучение (основное)

Обучение на `data/synthetic.f` (16 500 строк, clean = 4 853).

| Параметр    | Значение                      |
| ----------- | ----------------------------- |
| Clean строк | 4 853                         |
| Train (90%) | ~4 367                        |
| Test (10%)  | ~486                          |
| Noisy строк | 11 647 — исключены            |

### Noisy обучение (экспериментальное)

Обучение на clean + noisy данных (с фильтрацией noise_level).

```bash
make train-noisy EPOCHS=10 LORA_R=16 NOISE_LEVEL=0.3
```

| Параметр    | Значение                      |
| ----------- | ----------------------------- |
| Всего строк | 16 500                        |
| Clean train | ~4 367                        |
| Noisy train | ~10 483 (после фильтрации)    |
| Split       | 90/10                         |

**Важно:** Текущий чекпоинт noisy обучения повреждён — падает
"element 0 does not require grad". Фикс внесён, требуется
перезапуск: `make train-noisy`.

## LoRA

Вместо полного fine-tuning (65M параметров) обучаются только LoRA-адаптеры.

| Параметр         | Clean         | Noisy         |
| ---------------- | ------------- | ------------- |
| Ранг (r)         | 8             | 16            |
| Alpha            | 16            | 32            |
| Target modules   | q, v, k, o, wi, wo | q, v, k, o, wi, wo |
| Trainable params | 0.9M (1.4%)   | 1.8M (2.8%)   |
| Dropout          | 0.1           | 0.1           |
| Базовая модель   | frozen        | frozen        |

## Команды

```bash
# Clean обучение (рекомендуется)
make train-local EPOCHS=3

# Noisy обучение
make train-noisy-local EPOCHS=10 LORA_R=16 LORA_ALPHA=32

# Быстрый тест (200 samples, ~10 сек)
make train-quick

# С кастомными параметрами
make train-local EPOCHS=5 BATCH_SIZE=4 MAX_SAMPLES=2000

# Через Docker
make train EPOCHS=3
make train-noisy EPOCHS=10
```

## MLflow

```bash
make mlflow-up        # Запуск UI на :5001
make train-local      # Обучение с авто-логированием
make mlflow-down      # Остановка
```

MLflow логирует:

- Parameters: epochs, batch_size, lr, model_name, lora_r, lora_alpha
- Metrics: train_loss, eval_loss, test_accuracy по эпохам
- Artifacts: confusion matrix, classification report, prediction samples
- Model weights

## Гиперпараметры

| Параметр      | Clean     | Noisy     | Описание                       |
| ------------- | --------- | --------- | ------------------------------ |
| learning_rate | 5e-5      | 5e-5      | Скорость обучения              |
| batch_size    | 8         | 8         | Размер батча                   |
| max_length    | 128       | 128       | Макс. длина последовательности |
| warmup_steps  | 100       | 100       | Разогрев LR scheduler          |
| weight_decay  | 0.01      | 0.01      | L2 регуляризация               |
| LoRA r        | 8         | 16        | Ранг адаптера                  |

## Метрики

| Модель               | Accuracy    |
| -------------------- | ----------- |
| Current parser       | 99.80%      |
| Sequence parser      | 99.80%      |
| T5 (clean train)     | ~94%        |
| T5 (noisy train)     | Не завершён |
| Гибрид               | 94.4%       |

Noisy обучение должно поднять accuracy на синтетических ASR-данных.
Подробнее о парсерах: [docs/parser.md](parser.md)
