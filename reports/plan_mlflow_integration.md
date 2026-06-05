# План: Интеграция MLflow с обучением ruT5

**Branch:** main
**Date:** 2026-06-05 23:55:50 MSK

## Мотивация

Сейчас `make train` запускает обучение вслепую: в консоль сыпятся логи, но нет:
- реального времени графиков loss/accuracy
- сравнения между запусками
- автоматического сохранения лучшей модели
- classification reports на тесте
- истории экспериментов

MLflow решает всё это: дашборд в браузере, логирование каждой метрики, артефакты.

## Архитектура

```
scripts/train.py
  │
  ├── mlflow.start_run()
  │     ├── log_param: model_name, epochs, batch_size, lr, max_samples
  │     ├── log_param: dataset_size, train_size, test_size
  │     │
  │     ├── training loop (via Seq2SeqTrainer callback)
  │     │     └── log_metric: train_loss (каждый step)
  │     │     └── log_metric: eval_loss (каждый epoch)
  │     │     └── log_metric: learning_rate
  │     │
  │     ├── after each epoch:
  │     │     ├── inference на test set → classification report
  │     │     ├── log_text: classification_report.txt
  │     │     ├── generate plots: confusion_matrix.png, accuracy_chart.png
  │     │     └── log_artifact: *.png, *.txt
  │     │
  │     ├── at end:
  │     │     ├── save best model → log_artifact: model/*
  │     │     ├── save tokenizer → log_artifact: tokenizer/*
  │     │     └── set tag: status = completed
  │     │
  │     └── mlflow.end_run()
  │
  └── mlflow ui (отдельный процесс)
        └── http://localhost:5000
```

## Компоненты

### 1. MLflow tracking server

```
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts
```

- `make mlflow-up` — запуск сервера
- `make mlflow-down` — остановка
- `mlflow.db` — SQLite с метаданными экспериментов
- `mlflow-artifacts/` — сохранённые модели, графики, отчёты

### 2. Интеграция в train.py

**Новый модуль `src/trainer.py`** (или расширение `scripts/train.py`):

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ruT5-itn")

with mlflow.start_run(run_name=f"ruT5_ep{epochs}_lr{lr}") as run:
    # params
    mlflow.log_param("model", model_name)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("max_samples", max_samples)
    mlflow.log_param("dataset_size", len(train_df))
    mlflow.log_param("train_size", len(train_texts))
    mlflow.log_param("test_size", len(val_texts))

    # metrics during training — через callback
    class MLflowCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v, step=state.global_step)

    # after each epoch — full eval on test
    for epoch in range(epochs):
        # ... train ...
        # generate predictions on test set
        preds, targets = predict(model, tokenizer, test_texts, test_targets)
        # classification report
        report = classification_report(...)
        mlflow.log_text(report, f"reports/epoch_{epoch}/classification_report.txt")
        # plots
        fig = plot_confusion_matrix(preds, targets)
        mlflow.log_figure(fig, f"plots/epoch_{epoch}/confusion.png")
        # metrics
        acc = accuracy_score(targets, preds)
        mlflow.log_metric("test_accuracy", acc, step=epoch)

    # save model
    mlflow.log_artifacts("models/ruT5-itn", artifact_path="model")
```

### 3. Графики (plots)

| График | Описание | Когда |
|--------|----------|-------|
| `train_loss.png` | Loss по steps | каждый step |
| `eval_loss.png` | Loss по эпохам | каждая эпоха |
| `confusion_matrix.png` | Матрица ошибок | каждая эпоха |
| `accuracy_chart.png` | Accuracy по эпохам | каждая эпоха |
| `prediction_examples.png` | Примеры IN→OUT | последняя эпоха |

### 4. Артефакты

```
mlflow-artifacts/<experiment_id>/<run_id>/
  ├── model/
  │   ├── model.safetensors
  │   ├── config.json
  │   └── tokenizer.json
  ├── reports/
  │   ├── epoch_0/
  │   │   ├── classification_report.txt
  │   │   └── prediction_samples.txt
  │   ├── epoch_1/
  │   └── epoch_2/
  ├── plots/
  │   ├── epoch_0/
  │   │   ├── confusion.png
  │   │   └── accuracy.png
  │   ├── epoch_1/
  │   └── epoch_2/
  └── metrics.json
```

### 5. Дополнительно: compare runs

MLflow UI позволяет:
- Сравнить 2+ запуска side-by-side
- Отфильтровать по параметрам (max_samples, lr)
- Скачать артефакты любого запуска
- Экспорт в CSV

## Чек-лист реализации

### Шаг 1: Установка и настройка
- [x] `pip install mlflow` в requirements.txt
- [x] Создать `makefiles/mlflow.mk`
- [x] `make mlflow-up` — запуск сервера
- [x] `make mlflow-down` — остановка сервера
- [x] `make mlflow-ui` — открыть браузер
- [x] `.gitignore: mlflow.db, mlflow-artifacts/`

### Шаг 2: Интеграция в train.py
- [x] Импорт mlflow, set_tracking_uri, set_experiment
- [x] `start_run` с log_param всех гиперпараметров
- [x] Кастомный MLflowCallback для Seq2SeqTrainer
- [x] log_metric: train_loss, eval_loss, learning_rate

### Шаг 3: Оценка на тесте после эпохи
- [x] Функция `evaluate_and_report(model, tokenizer, ...)`
- [x] Генерация предсказаний на test set
- [x] `classification_report` (sklearn) → log_text
- [x] Примеры IN→OUT (первые 15) → log_text

### Шаг 4: Графики
- [x] Matplotlib: confusion matrix (цифры, не токены)
- [x] Matplotlib: accuracy по эпохам
- [x] Matplotlib: training_loss график
- [x] `mlflow.log_figure()` для каждого графика

### Шаг 5: Сохранение артефактов
- [x] `mlflow.log_artifacts("models/ruT5-itn", "model")`
- [x] `mlflow.log_text(report, "reports/...")`
- [x] `mlflow.log_figure(fig, "plots/...")`
- [x] Установка тегов: status, accuracy, model_name

### Шаг 6: Makefile цели
- [x] `make train / train-local` — mlflow по умолчанию
- [x] `make mlflow-up` — запуск UI
- [x] `make mlflow-down` — остановка
- [x] `make mlflow-clean` — очистка БД + артефактов

### Шаг 7: Тестирование
- [x] `make mlflow-up && make train-local EPOCHS=1 MAX_SAMPLES=200`
- [x] Проверить http://localhost:5001
- [x] Проверить артефакты в mlflow-artifacts/
- [x] Сравнить 2 запуска в UI

## Дерево файлов

```
scripts/
  train.py          ← расширить: +mlflow логирование
makefiles/
  mlflow.mk         ← новый: mlflow-up, mlflow-down, train-mlflow
  help.mk           ← + секция MLflow
requirements.txt    ← + mlflow
reports/
  plan_mlflow_integration.md ← этот план
```

## Критерии готовности

- [x] `make mlflow-up` запускает сервер на :5001
- [x] `make train` логирует все метрики в MLflow
- [x] В UI видны: params, metrics, plots, reports, model artifacts
- [x] `make mlflow-down` чисто останавливает сервер
- [x] classification_report.txt содержит precision/recall/f1
- [x] confusion_matrix.png показывает путаницу цифр
- [x] Можно сравнить 2 запуска в UI
