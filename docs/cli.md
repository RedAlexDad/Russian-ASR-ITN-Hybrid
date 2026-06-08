# CLI и Makefile

## Быстрый старт

### Локально (через .venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make evaluate-local              # current parser на calibration.f
make evaluate-sequence-local     # sequence parser на calibration.f
make run-local                   # нормализовать test.f → answer.f
```

### Через Docker

```bash
make deploy                  # down + build + up
make evaluate                # accuracy на calibration.f
make evaluate-sequence       # sequence parser accuracy
make run                     # нормализация test.f → answer.f
make down                    # остановить
```

## Makefile цели

### Docker lifecycle

| Цель                 | Описание                                |
| -------------------- | --------------------------------------- |
| `make` / `make help` | Справка                                 |
| `make build`         | Сборка образа (с кэшем)                 |
| `make up`            | Запуск контейнера (фон)                 |
| `make down`          | Остановка контейнера                    |
| `make deploy`        | down + build + up (рекомендуемый старт) |
| `make clean`         | Очистить Docker-ресурсы                 |

### Команды в контейнере

| Цель                  | Описание                               |
| --------------------- | -------------------------------------- |
| `make run`            | Нормализация test.f → answer.f         |
| `make evaluate`       | Current parser на calibration.f        |
| `make evaluate-sequence` | Sequence parser на calibration.f    |
| `make evaluate-synthetic`| Оценка на синтетическом датасете     |
| `make evaluate-real`  | Оценка на реальных данных              |
| `make errors`         | Показать ошибки (current, N=15)        |
| `make errors-sequence`| Показать ошибки (sequence)             |
| `make test`           | Запустить pytest                       |
| `make synthetic`      | Генерация синтетики                    |
| `make train`          | Обучение ruT5 на clean данных          |
| `make train-noisy`    | Обучение ruT5 на clean + noisy         |
| `make eda`            | EDA с графиками                        |

### Локальный запуск (без Docker)

Те же команды с суффиксом `-local`:

```bash
make evaluate-local
make evaluate-sequence-local
make train-local EPOCHS=3
make train-noisy-local LORA_R=32
make eda-local
```

### MLflow

```bash
make mlflow-up          # Запуск UI на :5001
make train-local        # Обучение с авто-логированием
make mlflow-down        # Остановка
make mlflow-clean       # Очистить БД и артефакты
```

## Переменные

| Переменная    | Умолчание            | Описание                       |
| ------------- | -------------------- | ------------------------------ |
| `INPUT`       | `data/test.f`        | Входной .feather               |
| `CALIB`       | `data/calibration.f` | calibration.f                  |
| `OUTPUT`      | `answer.f`           | Выходной .feather              |
| `EPOCHS`      | `3`                  | Эпох обучения                  |
| `BATCH_SIZE`  | `8`                  | Размер батча                   |
| `LORA_R`      | `16`                 | Ранг LoRA-адаптера             |
| `LORA_ALPHA`  | `32`                 | Alpha LoRA                     |
| `NOISE_LEVEL` | `0.3`                | Доля шума при noisy-обучении   |
| `MAX_SAMPLES` | все                  | Лимит сэмплов для train        |
| `N`           | `15`                 | Количество ошибок для show     |

## Python CLI

```bash
python main.py run data/test.f -o answer.f
python main.py run data/test.f -o answer.f --parser-type sequence
python main.py evaluate data/calibration.f
python main.py evaluate data/calibration.f --parser-type sequence
python main.py errors data/calibration.f -n 15 --parser-type sequence
```

### Флаг `--parser-type`

- `current` — legacy parser (умолчание)
- `sequence` — новый state-machine парсер с root-based классификатором

Подробнее: [docs/parser.md](parser.md), [src/cli.py](../src/cli.py)

### Clean/noisy split

При оценке на synthetic.f (если есть колонка `noise_level`),
accuracy выводится раздельно по clean и noisy подвыборкам.
