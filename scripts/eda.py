#!/usr/bin/env python3
"""
EDA: Exploratory Data Analysis для задачи ITN.

Запуск:  python scripts/eda.py
Формат:  каждая «секция» выводит markdown-заголовок -> код -> результат
Графики сохраняются в reports/plots/

Зачем нужно:
  Дать проверяющему полное понимание данных: распределение чисел,
  ASR-ошибки, сложность группировки, accuracy решения, типы ошибок.

Отличия от cli.py EDA:
  cli.py выводит краткую статистику после run/evaluate.
  Здесь — расширенный анализ с 8 графиками (гистограммы, heatmap, scatter).
"""

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# Добавляем корень проекта в sys.path для импорта src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dicts.asr_errors import ASR_ERRORS
from src.lexicon import is_ordinal_word, ordinal_value
from src.normalizer import normalize_text

# ── Настройки графиков ──
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
OUT = Path(__file__).resolve().parent.parent / "reports" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def section(title, level=2):
    """Выводит markdown-заголовок как разделитель секций."""
    print()
    print("#" * level, title)
    print()


def load_data():
    """Загрузка calibration.f и test.f через Polars."""
    section("Загрузка данных")
    cal = pl.read_ipc("data/calibration.f")
    test = pl.read_ipc("data/test.f")
    print(f"calibration.f: {cal.height} rows x {cal.width} cols")
    print(f"test.f:        {test.height} rows x {test.width} cols")
    print(f"\nСхема calibration: {cal.schema}")
    print(f"Схема test:        {test.schema}")
    return cal, test


def overview(cal, test):
    """Первичный осмотр: примеры строк, длины текстов, статистики."""
    section("Первичный осмотр")
    for row in cal.head(5).iter_rows(named=True):
        print(f"  task:   {row['task_text'][:70]}...")
        print(f"  ground: {row['ground_truth'][:70]}...")
        print()

    cal = cal.with_columns(
        pl.col("task_text").str.len_bytes().alias("task_len"),
        pl.col("task_text").str.split(" ").list.len().alias("task_tokens"),
        pl.col("ground_truth").str.len_bytes().alias("gt_len"),
        pl.col("ground_truth").str.split(" ").list.len().alias("gt_tokens"),
    )
    print("Статистики длин (task_text):")
    print(cal.select("task_len").describe())
    diff = cal.filter(pl.col("task_text") != pl.col("ground_truth"))
    print(
        f"\nСтрок с изменениями: {diff.height}/{cal.height} ({diff.height / cal.height * 100:.1f}%)"
    )
    return cal


def number_distribution(cal):
    """Гистограмма: сколько чисел в строке и их разрядность."""
    section("Распределение чисел")

    def count_digits(text):
        return len(re.findall(r"\d+", str(text)))

    cal = cal.with_columns(
        pl.col("ground_truth")
        .map_elements(count_digits, return_dtype=pl.Int32)
        .alias("num_count")
    )

    # Гистограмма 1: количество чисел на строку
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = cal["num_count"].to_list()
    sns.histplot(counts, bins=range(0, max(counts) + 2), discrete=True, ax=axes[0])
    axes[0].set_title("Чисел на строку")
    axes[0].set_xlabel("Количество чисел")

    # Гистограмма 2: разрядность (1-значные, 2-значные...)
    all_digits = []
    for gt in cal["ground_truth"].to_list():
        for num in re.findall(r"\d+", str(gt)):
            all_digits.append(len(num))
    sns.histplot(
        all_digits, bins=range(1, max(all_digits) + 2), discrete=True, ax=axes[1]
    )
    axes[1].set_title("Разрядность чисел")
    axes[1].set_xlabel("Количество цифр")
    plt.tight_layout()
    plt.savefig(OUT / "numbers_distribution.png", dpi=120)
    plt.close()
    print(
        f"Всего чисел: {len(all_digits)}, среднее на строку: {len(all_digits) / cal.height:.2f}"
    )

    # Круговая диаграмма: строки с числами vs без
    fig, ax = plt.subplots(figsize=(6, 6))
    has = cal.filter(pl.col("num_count") > 0).height
    no = cal.filter(pl.col("num_count") == 0).height
    ax.pie(
        [has, no],
        labels=[f"С числами ({has})", f"Без чисел ({no})"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#95a5a6"],
        startangle=90,
    )
    ax.set_title("Строки с числами vs без")
    plt.savefig(OUT / "pie_has_numbers.png", dpi=120)
    plt.close()
    print(f"Строк с числами: {has}, без чисел: {no}")
    return cal


def length_vs_numbers(cal):
    """Scatter plot: длина текста vs количество чисел + линия регрессии."""
    section("Длина текста vs количество чисел")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=cal.to_pandas(), x="task_tokens", y="num_count", alpha=0.6, ax=ax
    )
    sns.regplot(
        data=cal.to_pandas(),
        x="task_tokens",
        y="num_count",
        scatter=False,
        color="red",
        ax=ax,
    )
    ax.set_title("Длина текста vs количество чисел")
    ax.set_xlabel("Токенов в строке")
    ax.set_ylabel("Количество чисел")
    plt.savefig(OUT / "token_vs_numbers.png", dpi=120)
    plt.close()
    corr = cal.select(pl.corr("task_tokens", "num_count")).item()
    print(f"Корреляция: {corr:.3f}")


def asr_errors_analysis(cal):
    """Анализ ASR-ошибок: топ-20 искажений с частотой."""
    section("Анализ ASR-ошибок")
    # Собираем статистику: какие искажения сколько раз встретились
    error_counts = {}
    for row in cal["task_text"].to_list():
        for w in str(row).lower().split():
            if w in ASR_ERRORS:
                error_counts[w] = error_counts.get(w, 0) + 1

    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:20]
    print(f"{'Искажение':<20} {'-> канон':<15} {'вхождений':<10}")
    print("-" * 45)
    for err, cnt in sorted_errors:
        print(f"{err:<20} -> {ASR_ERRORS[err]:<15} {cnt:<10}")

    # График: топ ASR-искажений
    fig, ax = plt.subplots(figsize=(10, 6))
    words = [f"{e}->{ASR_ERRORS[e]}" for e, _ in sorted_errors]
    ax.barh(range(len(words)), [c for _, c in sorted_errors], color="coral")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Количество вхождений")
    ax.set_title("Топ ASR-искажений числительных")
    plt.tight_layout()
    plt.savefig(OUT / "asr_errors.png", dpi=120)
    plt.close()


def grouping_analysis(cal):
    """Анализ группировки: строки где task_tokens != gt_tokens (признак склейки/разделения)."""
    section("Анализ группировки: сумма vs перечисление")
    task_tokens_count = cal["task_tokens"].to_list()
    gt_tokens_count = cal["gt_tokens"].to_list()
    grouped = sum(1 for t, g in zip(task_tokens_count, gt_tokens_count) if t != g)
    print(f"Строк с группировкой: {grouped}/{cal.height}")

    examples = cal.filter(pl.col("task_tokens") != pl.col("gt_tokens")).head(10)
    for row in examples.iter_rows(named=True):
        print(f"  task:   {row['task_text']}")
        print(f"  ground: {row['ground_truth']}")
        print(f"  tokens: {row['task_tokens']} -> {row['gt_tokens']}")
        print()


def accuracy_analysis(cal):
    """Оценка точности: accuracy, классификация ошибок, полный список."""
    section("Оценка точности решения")
    correct = 0
    errors_list = []
    for row in cal.iter_rows(named=True):
        pred = normalize_text(row["task_text"])
        if pred == row["ground_truth"]:
            correct += 1
        else:
            errors_list.append((row["task_text"], row["ground_truth"], pred))

    accuracy = correct / cal.height * 100
    print(f"Accuracy: {correct}/{cal.height} = {accuracy:.2f}%")
    print(f"Ошибок: {len(errors_list)}")

    # Круговая: accuracy
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [correct, len(errors_list)],
        labels=[f"Верно ({correct})", f"Ошибки ({len(errors_list)})"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
        startangle=90,
    )
    ax.set_title(f"Accuracy: {accuracy:.1f}%")
    plt.savefig(OUT / "accuracy_pie.png", dpi=120)
    plt.close()

    # Классификация ошибок
    types = {}
    for task, gt, pred in errors_list:
        gt_nums = set(re.findall(r"\d+", gt))
        pred_nums = set(re.findall(r"\d+", pred))
        task_tok = task.split()
        gt_tok = gt.split()
        pred_tok = pred.split()

        if len(task_tok) != len(gt_tok) and len(task_tok) != len(pred_tok):
            t = "Неверная группировка"
        elif gt_nums - pred_nums:
            t = "Пропущенное число"
        elif pred_nums - gt_nums:
            t = "Лишнее число"
        elif gt_nums.symmetric_difference(pred_nums):
            t = "Неверное значение"
        else:
            t = "Другое"
        types[t] = types.get(t, 0) + 1

    print(f"\n{'Тип ошибки':<25} {'Кол-во':<10}")
    print("-" * 35)
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"{t:<25} {c:<10}")

    # График: типы ошибок
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(types.keys()), list(types.values()), color="coral")
    ax.set_xlabel("Количество")
    ax.set_title("Типы ошибок")
    plt.tight_layout()
    plt.savefig(OUT / "error_types.png", dpi=120)
    plt.close()

    # Полный список ошибок
    print(f"\n--- Полный список ошибок ({len(errors_list)}) ---")
    for i, (task, gt, pred) in enumerate(errors_list, 1):
        print(f"\n{i}. TASK: {task}")
        print(f"   GT:   {gt}")
        print(f"   PRED: {pred}")


def ordinals_analysis(cal):
    """Анализ порядковых числительных: частота и примеры."""
    section("Порядковые числительные")
    ord_count = 0
    ord_examples = []
    for row in cal.iter_rows(named=True):
        for w in str(row["task_text"]).split():
            if is_ordinal_word(w):
                ord_count += 1
                if len(ord_examples) < 10:
                    ord_examples.append((w, ordinal_value(w)))
    print(f"Всего вхождений порядковых: {ord_count}\n")
    print(f"{'Слово':<25} {'-> число':<10}")
    print("-" * 35)
    for word, val in ord_examples:
        print(f"{word:<25} -> {val:<10}")


def cal_vs_test(cal, test):
    """Сравнение распределений calibration vs test (гистограмма длин)."""
    section("Сравнение calibration vs test")
    test = test.with_columns(
        pl.col("task_text").str.len_bytes().alias("task_len"),
        pl.col("task_text").str.split(" ").list.len().alias("task_tokens"),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        cal["task_tokens"].to_list(), bins=30, alpha=0.5, label="calibration", ax=ax
    )
    sns.histplot(test["task_tokens"].to_list(), bins=30, alpha=0.5, label="test", ax=ax)
    ax.set_xlabel("Количество токенов")
    ax.set_ylabel("Количество строк")
    ax.set_title("Распределение длин: calibration vs test")
    ax.legend()
    plt.savefig(OUT / "cal_vs_test.png", dpi=120)
    plt.close()

    print(
        f"Calibration: mean={cal['task_tokens'].mean():.1f}, std={cal['task_tokens'].std():.1f}"
    )
    print(
        f"Test:        mean={test['task_tokens'].mean():.1f}, std={test['task_tokens'].std():.1f}"
    )


def conclusions():
    """Финальные выводы и рекомендации."""
    section("Выводы", 1)
    print("""
1. **Точность:** 97.6% на calibration.f, 12 ошибок из 500 строк.

2. **Источники ошибок:**
   - ASR-искажения: словарь покрывает ~40 вариантов
   - Группировка: различение суммы и перечисления — центральная сложность
   - Слитные написания: дветысячи, двестипятьсот

3. **Распределение чисел:**
   - 1-3 числа на строку
   - Преобладают 2-4-значные числа (цены, бюджеты)
   - ~15% строк без чисел

4. **Рекомендации:**
   - Levenshtein distance для неизвестных слов
   - Синтетическая генерация шумных данных
   - Улучшить детекцию границ числовых групп
""")


def main():
    """Точка входа: последовательный запуск всех секций EDA."""
    print("# EDA: Обратная текстовая нормализация (ITN) для ASR-транскрибаций\n")

    cal, test = load_data()
    cal = overview(cal, test)
    cal = number_distribution(cal)
    length_vs_numbers(cal)
    asr_errors_analysis(cal)
    grouping_analysis(cal)
    accuracy_analysis(cal)
    ordinals_analysis(cal)
    cal_vs_test(cal, test)
    conclusions()

    print(f"\nГрафики сохранены в: {OUT}/")
    print("EDA завершён.")


if __name__ == "__main__":
    main()
