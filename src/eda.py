"""
EDA — анализ результатов нормализации.

Два режима:
  report_run(df)      — после run: статистика найденных чисел
  report_evaluate(df) — после evaluate: accuracy, пропуски, примеры ошибок

Использует эти же данные в src/eda.py для расширенного анализа
с графиками (запускается через make eda).
"""

import re

from src.lexicon import is_ordinal_word, lookup_word


def count_numbers(text):
    """Считает цифровые токены в тексте (последовательности цифр)."""
    return len(re.findall(r'\d+', text))


def count_numeral_words(text):
    """Считает слова, которые являются числительными (по словарю)."""
    return sum(1 for w in text.split() if lookup_word(w) is not None or is_ordinal_word(w))


def text_stats(text):
    """Базовая статистика одного текста: длина, токены, числа."""
    return {
        'length': len(text),
        'tokens': len(text.split()),
        'digit_tokens': count_numbers(text),
        'numeral_words': count_numeral_words(text),
    }


def report_run(df):
    """Выводит статистику после нормализации: сколько чисел найдено."""
    stats = df['answer'].apply(text_stats)
    total_digits = sum(s['digit_tokens'] for s in stats)
    total_numeral = sum(s['numeral_words'] for s in stats)
    rows_with_digits = sum(1 for s in stats if s['digit_tokens'] > 0)

    print(f'  · строк с числами:      {rows_with_digits}/{len(df)}')
    print(f'  · всего цифровых чисел:  {total_digits}')
    print(f'  · всего слов-чисел:      {total_numeral}')
    print(f'  · среднее чисел/строку:  {total_digits / len(df):.2f}')

    lens = [s['tokens'] for s in stats]
    print(f'  · токенов: min={min(lens)}, avg={sum(lens)/len(lens):.1f}, max={max(lens)}')


def report_evaluate(df):
    """Выводит статистику после оценки: accuracy, пропуски, лишние, ошибки."""

    # Динамический импорт, чтобы избежать циклических зависимостей
    from src.normalizer import normalize_text

    correct = 0
    wrong_examples = []
    total_gold = 0
    total_pred = 0

    for _, row in df.iterrows():
        pred = normalize_text(row['task_text'])
        if pred == row['ground_truth']:
            correct += 1
        else:
            wrong_examples.append((row['task_text'], row['ground_truth'], pred))
        total_gold += count_numbers(row['ground_truth'])
        total_pred += count_numbers(pred)

    acc = correct / len(df) * 100
    print(f'  · accuracy:               {correct}/{len(df)} = {acc:.2f}%')
    print(f'  · чисел в GT:             {total_gold}')
    print(f'  · чисел в PRED:           {total_pred}')

    missed = max(0, total_gold - total_pred)
    extra = max(0, total_pred - total_gold)
    print(f'  · пропущено чисел:        {missed}')
    print(f'  · лишних чисел:           {extra}')

    if wrong_examples:
        print(f'  · ошибочных строк:        {len(wrong_examples)}')
        print(f'\n  Первые 3 ошибки:')
        for task, gt, pred in wrong_examples[:3]:
            print(f'    TASK: {task[:70]}')
            print(f'    GT:   {gt[:70]}')
            print(f'    PRED: {pred[:70]}')
            print()

    return acc
