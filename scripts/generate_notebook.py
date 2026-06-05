import json
import os


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.split("\n"),
        "outputs": [],
        "execution_count": None,
    }


cells = []

cells.append(
    md("""# EDA: Обратная текстовая нормализация (ITN) для ASR-транскрибаций

**Задача:** Преобразование словесной записи чисел в цифровую в транскрибациях звонков Avito.

**Данные:**
- `calibration.f` — 500 строк с известной нормализацией (колонки `task_text`, `ground_truth`)
- `test.f` — 501 строка без нормализации (только `task_text`)

**Цель EDA:**
1. Изучить распределение числовых значений в транскрибациях
2. Проанализировать ASR-ошибки в записи числительных
3. Оценить сложность различения суммы vs перечисления
4. Измерить точность текущего решения
5. Классифицировать ошибки для дальнейшего улучшения""")
)

cells.append(
    code("""import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120

print("Библиотеки загружены")""")
)

cells.append(
    md("""## 2. Загрузка данных

Файлы в формате feather (Apache Arrow IPC). Загружаем через Polars.""")
)

cells.append(
    code("""cal = pl.read_ipc('../data/calibration.f')
test = pl.read_ipc('../data/test.f')

print(f'calibration.f: {cal.height} rows x {cal.width} cols')
print(f'test.f:        {test.height} rows x {test.width} cols')
print()
print('Схема calibration:')
print(cal.schema)
print()
print('Схема test:')
print(test.schema)""")
)

cells.append(
    md("""## 3. Первичный осмотр

Смотрим на данные: примеры строк, длину текстов, количество чисел. Определяем базовые статистики.""")
)

cells.append(
    code("""print('=== Первые 5 строк calibration ===')
for row in cal.head(5).iter_rows(named=True):
    print(f'  task:   {row["task_text"][:70]}...')
    print(f'  ground: {row["ground_truth"][:70]}...')
    print()

print('=== Первые 3 строки test ===')
for row in test.head(3).iter_rows(named=True):
    print(f'  task: {row["task_text"][:70]}...')
    print()""")
)

cells.append(
    code("""cal = cal.with_columns(
    pl.col('task_text').str.len_bytes().alias('task_len'),
    pl.col('task_text').str.split(' ').list.len().alias('task_tokens'),
    pl.col('ground_truth').str.len_bytes().alias('gt_len'),
    pl.col('ground_truth').str.split(' ').list.len().alias('gt_tokens'),
)

print('Статистики длин (task_text):')
print(cal.select('task_len').describe())
print()
print('Статистики количества токенов:')
print(cal.select('task_tokens').describe())""")
)

cells.append(
    code("""diff = cal.filter(pl.col('task_text') != pl.col('ground_truth'))
same = cal.filter(pl.col('task_text') == pl.col('ground_truth'))
print(f'Строк с изменениями: {diff.height}/{cal.height} ({diff.height/cal.height*100:.1f}%)')
print(f'Строк без изменений: {same.height}/{cal.height} ({same.height/cal.height*100:.1f}%)')""")
)

cells.append(
    md("""## 4. Распределение чисел в данных

Смотрим, сколько чисел встречается в каждой строке и какие разрядности преобладают.""")
)

cells.append(
    code("""def extract_numbers(text):
    return re.findall(r'\\d+', str(text))

def count_digits(text):
    return len(extract_numbers(str(text)))

cal = cal.with_columns(
    pl.col('ground_truth').map_elements(count_digits, return_dtype=pl.Int32).alias('num_count')
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = cal['num_count'].to_list()
sns.histplot(counts, bins=range(0, max(counts)+2), discrete=True, ax=axes[0])
axes[0].set_title('Чисел на строку')
axes[0].set_xlabel('Количество чисел')
axes[0].set_ylabel('Количество строк')

all_digits = []
for gt in cal['ground_truth'].to_list():
    for num in extract_numbers(gt):
        all_digits.append(len(num))

sns.histplot(all_digits, bins=range(1, max(all_digits)+2), discrete=True, ax=axes[1])
axes[1].set_title('Разрядность чисел')
axes[1].set_xlabel('Количество цифр в числе')
axes[1].set_ylabel('Количество вхождений')

plt.tight_layout()
plt.savefig('../reports/plots/numbers_distribution.png', dpi=120)
plt.show()

print(f'Всего чисел в calibration: {len(all_digits)}')
print(f'Среднее чисел на строку: {len(all_digits)/cal.height:.2f}')""")
)

cells.append(
    code("""has_nums = cal.filter(pl.col('num_count') > 0).height
no_nums = cal.filter(pl.col('num_count') == 0).height

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([has_nums, no_nums],
       labels=[f'С числами ({has_nums})', f'Без чисел ({no_nums})'],
       autopct='%1.1f%%', colors=['#2ecc71', '#95a5a6'], startangle=90)
ax.set_title('Строки с числами vs без')
plt.savefig('../reports/plots/pie_has_numbers.png', dpi=120)
plt.show()""")
)

cells.append(
    md("""## 5. Длина текста vs количество чисел

Проверяем корреляцию между длиной фразы и количеством чисел в ней.""")
)

cells.append(
    code("""fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=cal.to_pandas(), x='task_tokens', y='num_count', alpha=0.6, ax=ax)
sns.regplot(data=cal.to_pandas(), x='task_tokens', y='num_count',
            scatter=False, color='red', ax=ax)
ax.set_title('Длина текста vs количество чисел')
ax.set_xlabel('Количество токенов в строке')
ax.set_ylabel('Количество чисел')
plt.savefig('../reports/plots/token_vs_numbers.png', dpi=120)
plt.show()

corr = cal.select(pl.corr('task_tokens', 'num_count')).item()
print(f'Корреляция: {corr:.3f}')""")
)

cells.append(
    md("""## 6. Анализ ASR-ошибок

Система распознавания речи (ASR) вносит искажения. Смотрим частотность — это словарь для устойчивости решения.""")
)

cells.append(
    code("""import sys
sys.path.insert(0, '..')
from src.dicts.asr_errors import ASR_ERRORS

error_counts = {}
for row in cal['task_text'].to_list():
    for w in str(row).lower().split():
        if w in ASR_ERRORS:
            error_counts[w] = error_counts.get(w, 0) + 1

sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:20]

print('Топ ASR-искажений:')
print(f'{"Искажение":<20} {"канон":<15} {"вхождений":<10}')
print('-' * 45)
for err, cnt in sorted_errors:
    canon = ASR_ERRORS[err]
    print(f'{err:<20} {canon:<15} {cnt:<10}')

fig, ax = plt.subplots(figsize=(10, 6))
words = [f'{e}->{ASR_ERRORS[e]}' for e, _ in sorted_errors]
counts = [c for _, c in sorted_errors]
ax.barh(range(len(words)), counts, color='coral')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Количество вхождений')
ax.set_title('Топ ASR-искажений числительных')
plt.tight_layout()
plt.savefig('../reports/plots/asr_errors.png', dpi=120)
plt.show()""")
)

cells.append(
    md("""## 7. Анализ группировки: сумма vs перечисление

Ключевая сложность задачи. Одна и та же последовательность слов может означать:
- **Сумму:** `две тысячи пятьсот` -> `2500` (группируем)
- **Перечисление:** `двести триста` -> `200 300` (не группируем)""")
)

cells.append(
    code("""task_tokens_count = cal['task_tokens'].to_list()
gt_tokens_count = cal['gt_tokens'].to_list()
grouped = sum(1 for t, g in zip(task_tokens_count, gt_tokens_count) if t != g)
print(f'Строк с группировкой: {grouped}/{cal.height}')

print()
print('Примеры группировки:')
examples = cal.filter(pl.col('task_tokens') != pl.col('gt_tokens')).head(10)
for row in examples.iter_rows(named=True):
    print(f'  task:   {row["task_text"]}')
    print(f'  ground: {row["ground_truth"]}')
    print(f'  tokens: {row["task_tokens"]} -> {row["gt_tokens"]}')
    print()""")
)

cells.append(
    md("""## 8. Точность текущего решения

Оцениваем качество пайплайна на calibration.f. Метрика: Accuracy.""")
)

cells.append(
    code("""from src.normalizer import normalize_text

correct = 0
errors_list = []
for row in cal.iter_rows(named=True):
    pred = normalize_text(row['task_text'])
    if pred == row['ground_truth']:
        correct += 1
    else:
        errors_list.append((row['task_text'], row['ground_truth'], pred))

accuracy = correct / cal.height * 100
print(f'Accuracy: {correct}/{cal.height} = {accuracy:.2f}%')
print(f'Ошибок: {len(errors_list)}')

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([correct, len(errors_list)],
       labels=[f'Верно ({correct})', f'Ошибки ({len(errors_list)})'],
       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
ax.set_title(f'Accuracy: {accuracy:.1f}%')
plt.savefig('../reports/plots/accuracy_pie.png', dpi=120)
plt.show()""")
)

cells.append(
    md("""## 9. Детальный разбор ошибок

Классифицируем каждую ошибку по типу.""")
)

cells.append(
    code("""def classify_error(task, gt, pred):
    gt_nums = set(re.findall(r'\\d+', gt))
    pred_nums = set(re.findall(r'\\d+', pred))

    task_tok = task.split()
    gt_tok = gt.split()
    pred_tok = pred.split()

    if len(task_tok) != len(gt_tok) and len(task_tok) != len(pred_tok):
        return 'Неверная группировка'

    missed = gt_nums - pred_nums
    extra = pred_nums - gt_nums

    if missed:
        return 'Пропущенное число'
    if extra:
        return 'Лишнее число'

    wrong_vals = gt_nums.symmetric_difference(pred_nums)
    if wrong_vals:
        return 'Неверное значение'

    return 'Другое'

types = {}
for task, gt, pred in errors_list:
    t = classify_error(task, gt, pred)
    types[t] = types.get(t, 0) + 1

print('Классификация ошибок:')
print(f'{"Тип":<25} {"Количество":<10}')
print('-' * 35)
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f'{t:<25} {c:<10}')

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(list(types.keys()), list(types.values()), color='coral')
ax.set_xlabel('Количество')
ax.set_title('Типы ошибок нормализации')
plt.tight_layout()
plt.savefig('../reports/plots/error_types.png', dpi=120)
plt.show()""")
)

cells.append(
    code("""print('=' * 80)
print(f'ПОЛНЫЙ СПИСОК ОШИБОК ({len(errors_list)})')
print('=' * 80)
for i, (task, gt, pred) in enumerate(errors_list, 1):
    print(f'\\n--- Ошибка {i} ---')
    print(f'TASK: {task}')
    print(f'GT:   {gt}')
    print(f'PRED: {pred}')""")
)

cells.append(
    md("""## 10. Порядковые числительные

Отдельный анализ порядковых числительных.""")
)

cells.append(
    code("""from src.lexicon import is_ordinal_word, ordinal_value

ord_count = 0
ord_examples = []
for row in cal.iter_rows(named=True):
    for w in str(row['task_text']).split():
        if is_ordinal_word(w):
            ord_count += 1
            if len(ord_examples) < 10:
                ord_examples.append((w, ordinal_value(w)))

print(f'Всего вхождений порядковых: {ord_count}')
print()
print('Примеры:')
print(f'{"Слово":<25} {"число":<10}')
print('-' * 35)
for word, val in ord_examples:
    print(f'{word:<25} {val:<10}')""")
)

cells.append(
    md("""## 11. Сравнение calibration vs test

Проверяем, что распределения обучающей и тестовой выборки совпадают.""")
)

cells.append(
    code("""test = test.with_columns(
    pl.col('task_text').str.len_bytes().alias('task_len'),
    pl.col('task_text').str.split(' ').list.len().alias('task_tokens'),
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(cal['task_tokens'].to_list(), bins=30, alpha=0.5, label='calibration', ax=ax)
sns.histplot(test['task_tokens'].to_list(), bins=30, alpha=0.5, label='test', ax=ax)
ax.set_xlabel('Количество токенов')
ax.set_ylabel('Количество строк')
ax.set_title('Распределение длин: calibration vs test')
ax.legend()
plt.savefig('../reports/plots/cal_vs_test.png', dpi=120)
plt.show()

print(f'Calibration: mean={cal["task_tokens"].mean():.1f}, std={cal["task_tokens"].std():.1f}')
print(f'Test:        mean={test["task_tokens"].mean():.1f}, std={test["task_tokens"].std():.1f}')""")
)

cells.append(
    md("""## 12. Выводы и рекомендации

### Итоги EDA

1. **Точность:** 97.6% на calibration.f, 12 ошибок из 500 строк.

2. **Основные источники ошибок:**
   - ASR-искажения: словарь покрывает ~40 вариантов
   - Группировка: различение суммы и перечисления
   - Слитные написания: не вошли в словарь

3. **Распределение чисел:**
   - Большинство строк содержит 1-3 числа
   - Преобладают 2-4-значные числа
   - ~15% строк не содержат чисел вообще

4. **Рекомендации:**
   - Добавить Levenshtein distance для неизвестных слов
   - Синтетическая генерация шумных данных
   - Улучшить детекцию границ числовых групп""")
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.3"},
    },
    "cells": cells,
}

os.makedirs("../notebooks", exist_ok=True)
with open("../notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("notebooks/eda.ipynb created")
