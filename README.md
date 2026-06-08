# Russian ASR ITN Hybrid

**Inverse Text Normalization** для числовых выражений в ASR-транскрибациях.

Преобразует словесную запись чисел в цифровую, устойчив к ошибкам распознавания речи.
Два парсера: **current** (legacy mag-based) и **sequence** (state machine + root classifier).

## Документация

| Раздел                                              | Описание                     |
| --------------------------------------------------- | ---------------------------- |
| [Архитектура](docs/architecture.md)                 | Общая схема и модули         |
| [Словари и TokenClass](docs/lexicon.md)             | Словари + классификатор      |
| [Парсер current vs sequence](docs/parser.md)        | Два подхода к парсингу       |
| [Нормализатор](docs/normalizer.md)                  | Обход текста и замена        |
| [Гибрид](docs/hybrid.md)                            | Парсер + ruT5 fallback       |
| [Обучение T5](docs/training.md)                     | Fine-tuning ruT5-small       |
| [Датасеты](docs/data.md)                            | calibration, synthetic, real |
| [CLI и Makefile](docs/cli.md)                       | Команды и переменные         |
| [Результаты обучения](docs/training_results.md)     | Все MLflow запуски           |

## Быстрый старт

### Локально

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make evaluate-local              # 99.80% current parser
make evaluate-sequence-local     # 99.80% sequence parser
make run-local                   # нормализовать test.f → answer.f
```

### Docker

```bash
make deploy               # down + build + up
make evaluate             # accuracy на calibration.f
make evaluate-sequence    # sequence parser accuracy
make run                  # нормализация → answer.f
make down                 # остановить
```

## Результаты

### Rule-based парсеры

| Датасет             | Current parser | Sequence parser |
| ------------------- | -------------- | --------------- |
| calibration.f       | **99.80%**     | **99.80%**      |
| synthetic.f (clean) | 95.87%         | 95.87%          |
| synthetic.f (noisy) | 2.93%          | 3.99%           |
| real.f              | 55.81%         | 55.81%          |

### T5 + LoRA (clean data)

| Модель              | Эпох | Test Acc |
| ------------------- | ---- | -------- |
| ruT5_ep5_lr5e-05    | 5    | 27.4%    |
| ruT5_ep10_lr5e-05   | 10   | 42.3%    |
| ruT5_ep10_r16       | 10   | 43.6%    |
| **ruT5_ep20_lr5e-05** | **20** | **57.3%** |

Подробнее: [docs/training_results.md](docs/training_results.md)
