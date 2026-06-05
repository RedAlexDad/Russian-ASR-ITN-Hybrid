# Датасеты

Проект использует три источника данных для обучения и валидации.

## 1. `data/calibration.f` — эталон (500 строк)

Поставляется с заданием. Содержит `task_text` и `ground_truth`.
Используется для оценки accuracy.

```
Accuracy = 488/500 = 97.6%
```

| Колонка | Описание |
|---------|----------|
| `task_text` | ASR-транскрибация (lowercase, без пунктуации) |
| `ground_truth` | Эталонная нормализация |

## 2. `data/synthetic.f` — синтетический датасет (16 500 строк)

Сгенерирован `scripts/generate_synthetic.py` для обучения ruT5.

| Источник | Строк | Описание |
|----------|-------|----------|
| Шаблоны | ~7 500 | 30 контекстных фраз × случайные числа |
| Группировка | ~6 000 | Сумма + перечисление |
| Порядковые | ~3 000 | 1-31 в 8 шаблонах |
| Новости | ~100 | RSS + Wikipedia |

| Колонка | Описание |
|---------|----------|
| `task_text` | Зашумлённый текст |
| `ground_truth` | Эталонная нормализация |
| `source` | origin / template / grouping / ordinal / news |
| `num_type` | cardinal / ordinal |
| `noise_level` | clean / noisy |

Clean accuracy нормализатора: **96.32%**.
Подробнее: [reports/plan_synthetic_real_data.md](../reports/plan_synthetic_real_data.md)

## 3. `data/real.f` — реальные данные (165 строк)

Собран из Wikipedia API + RSS-лент новостей.

| Разряд | Доля |
|--------|------|
| 1-digit | 29% |
| 2-digit | 26% |
| 3-digit | 11% |
| 4-digit | 34% (годы) |

## Сравнение датасетов

| Датасет | Строк | Clean (для обучения) | Train (90%) | Test (10%) |
|---------|-------|----------------------|-------------|------------|
| calibration.f | 500 | — | — | — |
| synthetic.f | 16 500 | 7 392 | 6 653 | 739 |
| real.f | 165 | 165 | — | — |

## Генерация синтетики

```bash
make synthetic         # через Docker
make synthetic-local  # через .venv
make evaluate-synthetic  # accuracy на синтетике
```

Подробнее: [scripts/generate_synthetic.py](../scripts/generate_synthetic.py)
