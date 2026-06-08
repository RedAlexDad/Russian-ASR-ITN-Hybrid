# Нормализатор текста

Содержит два параллельных пути нормализации: **current** (legacy) и **sequence** (новый).

---

# Current: `normalize_text()`

Обход токенов, выделение числовых групп по `lookup_word()`, замена на цифры.

```
Исходный текст → разбивка по пробелам → обход токенов

Для каждого токена:
  ├── Если числительное:
  │     └── Собрать все следующие числительные в группу
  │     └── Передать группу в parse_number_group()
  │     └── Вставить цифры вместо слов
  └── Если не числительное:
        └── Оставить как есть
```

Accuracy: **99.80%** (499/500) на calibration.f.

---

# Sequence: `normalize_text_sequence()`

Новый путь через `token_classifier.classify()` + `sequence_parser.parse_sequence()`.
Включает ASR-препроцессинг. Старый код не затрагивается.

```
normalize_text_sequence(text)
    │
    ├── _asr_preprocess(text)      # regex-замены
    ├── classify(token, context)   # TokenClass или None
    ├── parse_sequence(group)      # state machine → list[str]
    └── " ".join(result_tokens)
```

Accuracy: **99.80%** (499/500) на calibration.f.

## `_asr_preprocess()`

Regex-замены до парсинга для коррекции известных ASR-искажений:

| Паттерн | Замена | Пример |
| ------- | ------ | ------ |
| `\bдвеси\b` | двести | "двеси пятьдесят" → 250 |
| Склейки (две+тысячи) | Разделение пробелом | "дветысячи" → "две тысячи" |
| `(?<=\d)\s+(?=\d{3}\b)` | Удалить пробел | "200 000" → "200000" |
| ASR hundred+ten+thousand | Нормализация | "двесипятьдесяттысяч" → "двести пятьдесят тысяч" |

## Требование метрики

Accuracy требует **полного совпадения** строки с эталоном.
Оба нормализатора **никогда не трогают** слова, не распознанные как числительные.

## Связи

```
normalize_text()
    ├── lookup_word()          → src/lexicon.py
    ├── is_ordinal_word()      → src/lexicon.py
    └── parse_number_group()   → src/parser.py

normalize_text_sequence()
    ├── _asr_preprocess()      → src/normalizer.py
    ├── classify()             → src/token_classifier.py
    └── parse_sequence()       → src/sequence_parser.py
```

## Clean/noisy split

При запуске `make evaluate-synthetic` (если в данных есть колонка `noise_level`),
accuracy выводится отдельно по clean и noisy подвыборкам.

| Подвыборка | Доля   | Current    | Sequence   |
| ---------- | ------ | ---------- | ---------- |
| Clean      | 29.4%  | 95.87%     | 95.87%     |
| Noisy      | 70.6%  | 2.93%      | 3.99%      |

Подробнее о парсерах: [docs/parser.md](parser.md)  
Подробнее о гибриде: [docs/hybrid.md](hybrid.md)
