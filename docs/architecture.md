# Архитектура проекта

**Russian ASR ITN Hybrid** — решение для обратной текстовой нормализации (ITN)
числовых выражений в транскрибациях звонков.

## Общая схема

```
task_text (ASR-транскрибация)
        │
        ▼
┌───────────────────────────────────────────────┐
│                  src/                          │
│                                               │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐  │
│  │ lexicon  │ → │ parser   │ → │normalizer │  │
│  │ (словарь)│   │ (mag-    │   │ (обход    │  │
│  │          │   │  логика) │   │  токенов) │  │
│  └──────────┘   └──────────┘   └─────┬─────┘  │
│        ▲                             │        │
│        │                             ▼        │
│  ┌─────┴──────┐              ┌──────────┐     │
│  │ dicts/     │              │ hybrid   │     │
│  │ (словари)  │              │ (parser  │     │
│  │            │              │  + ruT5) │     │
│  └────────────┘              └──────────┘     │
│                                               │
└───────────────────────────────────────────────┘
        │
        ▼
answer (нормализованный текст с цифрами)
```

## Модули

| Модуль                          | Назначение                       | Документация                        |
| ------------------------------- | -------------------------------- | ----------------------------------- |
| `src/dicts/`                    | Словари числительных всех форм   | [docs/lexicon.md](lexicon.md)       |
| `src/lexicon.py`                | Сборка словаря + lookup-функции  | [docs/lexicon.md](lexicon.md)       |
| `src/parser.py`                 | Парсер суммы vs перечисления     | [docs/parser.md](parser.md)         |
| `src/normalizer.py`             | Обход текста, замена чисел       | [docs/normalizer.md](normalizer.md) |
| `src/hybrid.py`                 | Гибрид: парсер + ruT5            | [docs/hybrid.md](hybrid.md)         |
| `src/cli.py`                    | Точки входа: run/evaluate/errors | [docs/cli.md](cli.md)               |
| `scripts/train.py`              | Обучение ruT5-small              | [docs/training.md](training.md)     |
| `scripts/generate_synthetic.py` | Генерация синтетики              | [docs/data.md](data.md)             |

## Makefile цели

| Цель             | Действие                                |
| ---------------- | --------------------------------------- |
| `make deploy`    | down + build + up (рекомендуемый старт) |
| `make run`       | Нормализация test.f → answer.f          |
| `make evaluate`  | Оценка accuracy на calibration.f        |
| `make synthetic` | Генерация синтетического датасета       |
| `make train`     | Обучение ruT5 (через Docker)            |
| `make mlflow-up` | Запуск MLflow UI                        |
| `make *-local`   | Запуск без Docker (через .venv)         |

Подробнее: [docs/cli.md](cli.md)

## Поток данных

1. **Вход:** сырой текст из ASR (`task_text`)
2. **Токенизация:** разбивка по пробелам
3. **Поиск:** `lookup_word()` — словарь + ASR-ошибки
4. **Группировка:** `parse_number_group()` — сумма vs перечисление
5. **Замена:** числовые токены → цифры
6. **Выход:** нормализованный текст (`answer`)
7. **Опционально:** fallback на ruT5 при низкой уверенности парсера

## Docker

Многостадийная сборка (builder + runtime), 5 слоёв кэширования.
Подробнее: [Dockerfile](../Dockerfile), [docker-compose.yml](../docker-compose.yml).
