# CLI и Makefile

## Быстрый старт

### Локально (через .venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make evaluate-local          # проверить accuracy
make run-local               # нормализовать test.f → answer.f
```

### Через Docker

```bash
make deploy                  # down + build + up
make evaluate                # accuracy на calibration.f
make run                     # нормализация test.f → answer.f
make down                    # остановить
```

## Makefile цели

### Docker lifecycle

| Цель | Описание |
|------|----------|
| `make` / `make help` | Справка |
| `make build` | Сборка образа (с кэшем) |
| `make up` | Запуск контейнера (фон) |
| `make down` | Остановка контейнера |
| `make deploy` | down + build + up (рекомендуемый старт) |
| `make clean` | Очистить Docker-ресурсы |

### Команды в контейнере

| Цель | Описание |
|------|----------|
| `make run` | Нормализация test.f → answer.f |
| `make evaluate` | Оценка accuracy на calibration.f |
| `make errors` | Показать ошибки (N=15) |
| `make test` | Запустить pytest |
| `make synthetic` | Генерация синтетики |
| `make train` | Обучение ruT5 |
| `make eda` | EDA с графиками |

### Локальный запуск (без Docker)

Те же команды с суффиксом `-local`:

```bash
make evaluate-local
make train-local EPOCHS=3
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

| Переменная | Умолчание | Описание |
|------------|-----------|----------|
| `INPUT` | `data/test.f` | Входной .feather |
| `CALIB` | `data/calibration.f` | calibration.f |
| `OUTPUT` | `answer.f` | Выходной .feather |
| `EPOCHS` | `3` | Эпох обучения |
| `BATCH_SIZE` | `8` | Размер батча |
| `MAX_SAMPLES` | все | Лимит сэмплов для train |
| `N` | `15` | Количество ошибок для show |

## Python CLI

```bash
python main.py run data/test.f -o answer.f
python main.py evaluate data/calibration.f
python main.py errors data/calibration.f -n 15
```

Подробнее: [src/cli.py](../src/cli.py)
