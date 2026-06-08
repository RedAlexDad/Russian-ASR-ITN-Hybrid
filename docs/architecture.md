# Архитектура проекта

**Russian ASR ITN Hybrid** — решение для обратной текстовой нормализации (ITN)
числовых выражений в транскрибациях звонков.

## Общая схема

```
task_text (ASR-транскрибация)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                          src/                                 │
│                                                               │
│  ┌──────────┐   ┌─────────────────┐   ┌──────────────────┐    │
│  │dicts/    │ → │token_classifier │ → │sequence_parser   │    │
│  │(словари) │   │(root-based      │   │(state machine    │    │
│  │          │   │ classify +      │   │ lookahead heur.) │    │
│  └──────────┘   │ TokenClass)     │   └──────┬───────────┘    │
│       │         └────────┬────────┘          │                │
│       │                  │                   │                │
│      \|/                \|/                 \|/               │
│  ┌──────────────────────────────────────────────┐             │
│  │               normalizer.py                  │             │
│  │  normalize_text()  (legacy)                  │             │
│  │  normalize_text_sequence()  (sequence path)  │             │
│  │  _asr_preprocess()  (ASR regex preprocessor) │             │
│  └─────────────────────┬────────────────────────┘             │
│                        │                                      │
│              ┌─────────┴──────────┐                           │
│              │     hybrid.py      │                           │
│              │  parser + ruT5     │                           │
│              └────────────────────┘                           │
│                                                               │
│  ┌──────────────────────────────────────────────────┐         │
│  │  lexicon.py  (legacy lookup_word, delegating)    │         │
│  └──────────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
answer (нормализованный текст с цифрами)
```

## Два пути парсинга

| Параметр          | Current (legacy)                     | Sequence (новый)                      |
| ----------------- | ------------------------------------ | ------------------------------------- |
| Классификация     | `lookup_word()` — точное совпадение  | `classify()` — root regex + ASR + Fused |
| Типы токенов      | val, mag, is_mult, is_ordinal (tuple)| TokenClass (value, mag, subtype, raw)  |
| Группировка       | `parse_number_group()` — ad-hoc mag  | `parse_sequence()` — state machine     |
| ASR-препроцессинг | нет                                  | `_asr_preprocess()` — regex-замены     |
| Неизвестные слова | None → пропуск                      | root regex + fused compound split      |
| Accuracy калибр.  | 99.80% (499/500)                    | 99.80% (499/500)                       |

Оба пути работают параллельно. Выбор через `--parser-type {current,sequence}`.

## Модули

| Модуль                          | Назначение                            | Документация                              |
| ------------------------------- | ------------------------------------- | ----------------------------------------- |
| `src/dicts/`                    | Словари числительных всех форм        | [docs/lexicon.md](lexicon.md)             |
| `src/token_classifier.py`       | Root-based классификация токенов      | [docs/lexicon.md](lexicon.md)             |
| `src/sequence_parser.py`        | State machine над TokenClass          | [docs/parser.md](parser.md)               |
| `src/lexicon.py`                | Legacy lookup + делегирование         | [docs/lexicon.md](lexicon.md)             |
| `src/parser.py`                 | Legacy парсер (mag-logic)             | [docs/parser.md](parser.md)               |
| `src/normalizer.py`             | Обход текста, замена чисел (оба пути) | [docs/normalizer.md](normalizer.md)       |
| `src/hybrid.py`                 | Гибрид: парсер + ruT5                | [docs/hybrid.md](hybrid.md)               |
| `src/cli.py`                    | Точки входа: run/evaluate/errors      | [docs/cli.md](cli.md)                     |
| `scripts/train.py`              | Обучение ruT5-small + LoRA            | [docs/training.md](training.md)           |
| `scripts/generate_synthetic.py` | Генерация синтетики                   | [docs/data.md](data.md)                   |

## Makefile цели

| Цель                    | Действие                                   |
| ----------------------- | ------------------------------------------ |
| `make deploy`           | down + build + up (рекомендуемый старт)    |
| `make run`              | Нормализация test.f → answer.f             |
| `make evaluate`         | Оценка accuracy на calibration.f           |
| `make evaluate-sequence`| Sequence parser accuracy                   |
| `make evaluate-synthetic`| Оценка на синтетике                       |
| `make evaluate-real`    | Оценка на реальных данных                  |
| `make synthetic`        | Генерация синтетического датасета           |
| `make train`            | Обучение ruT5 (clean)                      |
| `make train-noisy`      | Обучение ruT5 (clean + noisy)              |
| `make mlflow-up`        | Запуск MLflow UI                           |
| `make *-local`          | Все команды без Docker (через .venv)        |

Подробнее: [docs/cli.md](cli.md)

## Поток данных (sequence path)

1. **Вход:** сырой текст из ASR (`task_text`)
2. **ASR-препроцессинг:** regex-замены (двеси → двести, склейки)
3. **Токенизация:** разбивка по пробелам
4. **Классификация:** `classify()` — dict match → ASR errors → root regex → fused
5. **State machine:** `parse_sequence()` — сумма/умножение/перечисление через TokenClass
6. **Замена:** числовые токены → цифры
7. **Выход:** нормализованный текст (`answer`)
8. **Опционально:** fallback на ruT5 при низкой уверенности парсера

## Docker

Многостадийная сборка (builder + runtime), 5 слоёв кэширования.
Подробнее: [Dockerfile](../Dockerfile), [docker-compose.yml](../docker-compose.yml).
