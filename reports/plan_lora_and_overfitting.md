# План: LoRA и борьба с переобучением при DL-обучении ITN

**Branch:** main
**Date:** 2026-06-06 00:39:43 MSK (обновлён: 2026-06-07)
**Status:** LoRA работает, модель обучена (57.3%), гибрид реализован. Осталось: early stopping, save best, hybrid в main.py

## Проблема

При обучении ruT5-small (65M параметров) на малом количестве данных (<500 samples)
наблюдается:

1. **eval_loss падает** (24 → 0.2), но **accuracy = 0%**
2. Модель выдаёт мусорные токены (`<0x57>`) на тесте
3. Полное fine-tuning на 65M параметров при 150 samples = 100% переобучение
4. Одна эпоха на полном датасете (7392 строк) на CPU занимает ~20 минут
5. Нет механизма early stopping и сохранения лучшей модели

## Почему модель выдавала `<0x57>` (мусорные токены)

**Коренная причина: ошибка в формировании `labels`.**

В `ITNDataset.__getitem__` метки (target sequences) создавались через
`tokenizer(..., padding='max_length')`, что заполняло все пустые позиции
токеном `<pad>` (id=0). Эти метки напрямую передавались в `loss`.

```python
# БЫЛО (неправильно):
tgt = tokenizer(target, padding='max_length', return_tensors='pt')
labels = tgt['input_ids'][0]    # содержит <pad> (id=0)

# СТАЛО (правильно):
tgt = tokenizer(target, padding='max_length', return_tensors='pt')
labels = tgt['input_ids'][0].clone()
labels[labels == 0] = -100      # loss игнорирует -100
```

**Почему это приводило к `<0x57>`:**

1. В каждом батче ~59 из 64 токенов были `<pad>` (id=0)
2. Модель видела `<pad>` как валидный токен и училась его предсказывать
3. На генерации модель начинала с `<pad>`, затем переходила на байтовые
   токены (`<0x57>`), потому что они статистически связаны с `<pad>` в BPE
4. Визуально: `['<pad>', '<pad>', ..., '<0x57>', 'относительно', 'скольки', 'дней']`

**Как проверили:**

- До обучения: модель повторяла вход (нормально для untrained T5)
- После обучения на 150 samples: добавились `<pad>` + `<0x57>` в начале выхода
- После фикса labels: `<0x57>` исчез, модель вернулась к повторению входа
  (но это уже проблема количества данных, не качества labels)

## Почему это происходит

| Фактор            | Значение            | Проблема                               |
| ----------------- | ------------------- | -------------------------------------- |
| Параметров модели | 65 000 000          | Огромное пространство для переобучения |
| Train samples     | 135 (при 150 total) | Меньше, чем параметров в 500 000 раз   |
| Test samples      | 15 (при 150 total)  | Статистически ничтожно                 |
| Токены на выходе  | 32-128              | Каждый токен — 65K вариантов           |
| Обучение          | Все слои            | Модель запоминает, а не обобщает       |

## Решение: LoRA (Low-Rank Adaptation)

### Что это

Вместо обучения всех 65M параметров, LoRA добавляет маленькие адаптеры (ранг r=8)
в слои attention. Обучается только **2.1M параметров** (3.2% от оригинала).

```
Полное fine-tuning:     ████████████████████████████ 65M параметров ← переобучение
LoRA (r=8):             ██▎                          2.1M параметров ← обобщение
```

### Компоненты

1. **LoraConfig** — конфигурация адаптеров:
   - `r=8` — ранг (размер адаптера)
   - `lora_alpha=16` — масштаб обновлений
   - `target_modules=['q', 'v', 'k', 'o', 'wi', 'wo']` — слои attention
   - `lora_dropout=0.1` — dropout для регуляризации

2. **Обучаемые модули** — только LoRA-адаптеры, базовая модель frozen.

3. **Совместимость** — работает с HF Trainer, Seq2SeqTrainer, save/load как обычно.

### Результаты (150 samples, 3 epochs) — начальные

| Метрика                | Полный fine-tune | LoRA (r=8)     |
| ---------------------- | ---------------- | -------------- |
| Trainable params       | 65M (100%)       | 2.1M (3.2%)    |
| Время эпохи            | ~25s             | ~25s           |
| eval_loss              | 0.20 (overfit)   | —              |
| Тест                   | мусор (`<0x57>`) | —              |
| Размер модели на диске | 250MB            | 6MB (адаптеры) |

### Результаты (2026-06-07) — итоговые

| Конфигурация | Accuracy | Number Acc | CER | Время | eval_loss |
|---|---|---|---|---|---|
| 3ep, 2000 samples (r=8) | 2.00% | 1.80% | 0.704 | ~3 мин | — |
| 5ep, 4890 clean (r=8) | 27.40% | 26.70% | 0.334 | 15.5 мин | 1.08 |
| 10ep, 4890 clean (r=8) | 42.33% | 46.52% | 0.237 | 31 мин | 0.83 |
| **10ep, 4890 clean (r=16)** | **43.56%** | — | — | **31.3 мин** (GPU) | **0.71** |
| **20ep, 4890 clean (r=8)** | **57.26%** | **60.07%** | **0.152** | **61.6 мин** | **0.48** |

**Ключевые выводы:**
- Мусорных токенов нет — метки исправлены (labels=-100 для pad)
- Потеря регистра (заглавная буква) исправлена к 20 эпохе — 0 ошибок из 489
- eval_loss: 1.33 → 0.48 (плато с 16 эпохи)
- Модель не переобучается: eval_loss монотонно падает все 20 эпох
- **r=16 vs r=8 при 10ep**: accuracy 43.56% vs 42.33% (+1.2%), eval_loss 0.71 vs 0.83 — прирост есть, но небольшой
- **r=16 (1.8M)** vs **r=8 (0.9M)** — удвоение параметров не даёт пропорционального прироста на clean-данных

## Чек-лист реализации LoRA

### Шаг 1: Установка

- [x] `pip install peft` — библиотека
- [x] Добавить `peft` в requirements.txt
- [ ] Добавить `peft` в Dockerfile

### Шаг 2: Интеграция в train.py

- [x] Импорт: `from peft import LoraConfig, get_peft_model, TaskType`
- [x] LoraConfig с r=8, alpha=16, target_modules
- [x] `model = get_peft_model(model, lora_config)` после загрузки
- [x] Вывод числа trainable params
- [x] Аргументы: `--lora` (default), `--no-lora`, `--lora-r`, `--lora-alpha`

### Шаг 3: Борьба с переобучением

- [x] **Early Stopping** — `EarlyStoppingCallback(patience=5)` добавлен
- [x] **Save best model** — `load_best_model_at_end=True` + `metric_for_best_model='eval_loss'`
- [x] **Learning rate scheduler** — линейный decay (по умолчанию HF Trainer)
- [x] **Gradient clipping** — `max_grad_norm=1.0` (по умолчанию HF Trainer)
- [ ] **Weight decay** — `weight_decay=0.01` (не задан)
- [x] **Dropout** — `lora_dropout=0.1` уже в LoRA

### Шаг 4: Аугментация данных

- [x] Resume: `--model-path` загружает LoRA адаптер, `make train-local MODEL_PATH=...`
- [x] Noisy: `--noise-level` (clean/noisy/all), `make train-noisy`
- [x] LoRA rank: `--lora-r`, `make train-local LORA_R=16`
- [ ] Добавить разнообразие шаблонов
- [ ] Добавить реальные данные (Wikipedia + RSS) в train (make fetch-real-local)

### Шаг 5: Метрики

- [ ] Exact match accuracy (текущая)
- [ ] Digit-level accuracy (совпадение цифр, не строк)
- [ ] eval_loss для early stopping

### Шаг 6: Полный тест

- [x] `make train-local EPOCHS=5` — accuracy 27.4% (> 0% ✓)
- [x] `make train-local EPOCHS=10` — accuracy 42.3%
- [x] `make train-local BATCH_SIZE=16 EPOCHS=10 LORA_R=16` — r=16: 43.56%
- [x] `make train-local EPOCHS=20` — accuracy 57.3%
- [x] Проверить MLflow: eval_loss по эпохам — логируется per-epoch ✓
- [x] Проверить inference: генерация без мусорных токенов — чисто ✓

## Архитектура LoRA

```
Input
  │
  ▼
┌─────────────────────┐
│  ruT5 Encoder       │  ← frozen (не обучается)
│  (12 layers)        │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│  ruT5 Decoder       │  ← frozen (не обучается)
│  (12 layers)        │
│  ┌───────────────┐  │
│  │ Attention QKV │  │  ← LoRA адаптеры (r=8)
│  │ + LoRA (r=8)  │  │     Q, K, V, O — 4 адаптера
│  └───────────────┘  │     WI, WO — 2 адаптера
│  ┌───────────────┐  │     Всего: 6 адаптеров × 12 слоёв
│  │ FeedForward   │  │
│  │ + LoRA (r=8)  │  │
│  └───────────────┘  │
└─────────────────────┘
  │
  ▼
Output tokens
```

## Критерии готовности

- [x] LoRA работает: trainable = 0.9M/66M
- [x] eval_loss не растёт к концу обучения (0.48, плато без переобучения)
- [x] Accuracy > 0% на 500 samples (27%+ на чистых)
- [x] Accuracy > 50% на полном датасете (57.3%)
- [x] MLflow логирует train_loss, eval_loss, lr по эпохам
- [x] Нет мусорных токенов (`<0x57>`) в генерации
- [x] Early stopping останавливает обучение при плато
- [x] Сохраняется лучшая модель (не последняя)
- [x] `make train-local` работает без дополнительных флагов

## Гибридный подход: Парсер + ruT5 (fallback)

### Архитектура гибрида

```
Входной текст
     │
     ▼
┌──────────────┐
│ Парсер       │ → normalize_text() → pred_rule + confidence
│ (словари +   │
│  magnitude)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Оценка       │ → confidence ≥ 0.95 → сразу ответ
│ уверенности  │
│ парсера      │ → confidence < 0.95 → fallback на ruT5
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ruT5-small   │ → генерация через model.generate()
│ (LoRA r=8)   │
│ 0.9M/66M     │
└──────┬───────┘
       │
       ▼
    Ответ
```

### Когда парсер сомневается (confidence < 0.95)

| Признак             | Вес   | Описание                                    |
| ------------------- | ----- | ------------------------------------------- |
| `has_asr_error`     | -0.3  | В тексте найдены ASR-искажения (двеси, тыщ) |
| `has_merged_tokens` | -0.2  | Склейки слов (дветысячи, тристатысяч)       |
| `has_unknown_words` | -0.2  | Слова вне словаря                           |
| `numeral_density`   | -0.1  | >3 числительных подряд                      |
| `magnitude_gap`     | -0.15 | Скачок magnitude (ед → тыс без сотен)       |

Стартовая уверенность = 1.0. Каждый признак уменьшает confidence.
Если confidence < 0.75 → fallback на ruT5.

### Код: `src/hybrid.py`

```python
def hybrid_normalize(text):
    pred_rule = normalize_text(text)
    confidence = parser_confidence(text, pred_rule)

    if confidence >= 0.75 or not _t5_available:
        return pred_rule

    pred_t5 = t5_generate(text)
    return pred_t5
```

### Фактический результат

| Мера                    | Парсер          | Гибрид                  |
| ----------------------- | --------------- | ----------------------- |
| Accuracy на calibration | 97.6% (488/500) | 94.4% (472/500)         |
| Время на 500 строк      | ~0.2 сек        | ~30-60 сек (из-за ruT5) |

**Вывод:** гибрид уступает парсеру, т.к. ruT5 обучена на синтетике и не
справляется с реальными ASR-паттернами. Флаг `--hybrid` доступен для
экспериментов, но не рекомендуется по умолчанию.

### Чек-лист гибрида

- [x] Создать `src/hybrid.py` (normalize + t5_fallback)
- [x] Функция parser_confidence() — оценка уверенности
- [x] fallback на ruT5 при низкой уверенности
- [x] graceful fallback: если модели нет → только парсер
- [x] Интеграция в main.py (run, evaluate) — флаг `--hybrid`
- [x] Тест на calibration.f — 94.4% гибрид vs 97.6% парсер (модель не исправляет ошибки синтетики)

## Следующие шаги

1. ✅ LoRA интегрирован — 0.9M параметров вместо 65M
2. ✅ Добавить early stopping (patience=5)
3. ✅ Добавить save best model (load_best_model_at_end)
4. ✅ Протестировано: 3ep→2%, 5ep→27%, 10ep→42%, 20ep→57%, r=16→43.5%
5. ✅ Полное обучение на 4890 clean: 20 эпох, 57.3% accuracy (r=8)
6. ✅ Реализовать гибрид: парсер + ruT5 fallback
7. ✅ Подключить hybrid_normalize в main.py — флаг `--hybrid`
8. ✅ `make evaluate` — гибрид 94.4% vs парсер 97.6%
9. ✅ Resume: `--model-path` загружает LoRA адаптер, дообучает дальше
10. ✅ Noisy: `--noise-level` (clean/noisy/all), `make train-noisy`  
11. ✅ LoRA rank: `--lora-r N`, `make train-local LORA_R=16`
12. 📝 Запустить `make train-noisy MODEL_PATH=models/ruT5-itn EPOCHS=10`
13. ❌ n-gram char LM для confidence — НЕ РАБОТАЕТ (длина mismatch: train 11.7 символов vs inference ASR)
14. 📝 n-gram char correction model (лёгкая альтернатива ruT5)

### Эксперимент: Char n-gram LM для confidence

| Подход | Ошибок поймано | FP |
|---|---|---|
| Хардкодные правила (базовый) | **11/12** | **13** |
| N-gram на parser output (order=4, p5) | 1/12 | 25 |
| N-gram на parser output (order=4, p20) | 1/12 | 99 |
| N-gram на input text (order=4, p20) | 1/12 | 98 |
| N-gram на всех текстах synthetic.f (p20) | 1/12 | 98 |

**Вывод:** n-gram не работает для confidence из-за:
- Длина ground_truth в train: 3-69 (средняя 11.7)
- Длина calibration.f: 60-250+ символов
- Распределение символов разное (короткие числа vs длинные ASR-транскрипты)
- Хардкодные правила детектят конкретные паттерны (тыщ, двеси) — n-gram их "размазывает"
- `src/ngram_lm.py` и `scripts/train_ngram_lm.py` оставлены как инструмент анализа данных
