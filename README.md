# Russian ASR ITN Hybrid

**Inverse Text Normalization** для числовых выражений в ASR-транскрибациях.

Преобразует словесную запись чисел в цифровую, устойчив к ошибкам распознавания речи.
Accuracy на calibration.f: **97.6%**.

## Документация

| Раздел | Описание |
|--------|----------|
| [Архитектура](docs/architecture.md) | Общая схема и модули |
| [Словари и ASR-ошибки](docs/lexicon.md) | Словарные данные |
| [Парсер](docs/parser.md) | Логика сумма/перечисление |
| [Нормализатор](docs/normalizer.md) | Обход текста и замена |
| [Гибрид](docs/hybrid.md) | Парсер + ruT5 fallback |
| [Обучение](docs/training.md) | Fine-tuning ruT5-small |
| [Датасеты](docs/data.md) | calibration, synthetic, real |
| [CLI и Makefile](docs/cli.md) | Команды и переменные |

## Быстрый старт

### Локально

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make evaluate-local       # 97.6% accuracy
make run-local            # нормализовать test.f → answer.f
```

### Docker

```bash
make deploy               # down + build + up
make evaluate             # accuracy на calibration.f
make run                  # нормализация → answer.f
make down                 # остановить
```

## Результаты

| Датасет | Accuracy | Ошибок |
|---------|----------|--------|
| calibration.f | 97.6% | 12/500 |
| synthetic.f (clean) | 96.32% | 5% |
| real.f | 83.0% | 17% |

## Структура проекта

```
├── src/               # Исходный код
│   ├── dicts/         # Словари числительных
│   ├── lexicon.py     # Поиск по словарю
│   ├── parser.py      # Парсер сумма/перечисление
│   ├── normalizer.py  # Нормализатор текста
│   ├── hybrid.py      # Гибрид (парсер + ruT5)
│   ├── cli.py         # CLI run/evaluate/errors
│   └── eda.py         # EDA-статистика
├── scripts/           # Скрипты
│   ├── train.py       # Обучение ruT5
│   ├── eda.py         # EDA с графиками
│   ├── generate_synthetic.py  # Генерация синтетики
│   └── fetch_real_data.py     # Сбор реальных данных
├── makefiles/         # Makefile модули
├── docs/              # Документация
├── data/              # Датасеты (.feather)
├── reports/           # Отчёты и планы
│   └── plots/         # EDA-графики
├── tests/             # Тесты
├── models/            # Обученные модели
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## Технологии

- **Python 3.12** — основной язык
- **Polars** — обработка данных
- **PyTorch + Transformers + PEFT** — DL-обучение (ruT5-small + LoRA)
- **MLflow** — трекинг экспериментов
- **Docker + Compose v2** — контейнеризация
- **Matplotlib + Seaborn** — визуализация EDA
- **pytest** — тестирование

## Лицензия

MIT
