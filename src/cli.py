"""
CLI — три точки входа: run, evaluate, errors.

Использование:
  python main.py run data/test.f -o answer.f
  python main.py evaluate data/calibration.f
  python main.py errors data/calibration.f -n 15

Каждая команда:
  1. Читает .feather файл через pandas
  2. Применяет normalize_text() к task_text
  3. Выводит результат + EDA-статистику
"""

import argparse
import sys

import pandas as pd

from src import eda
from src.normalizer import normalize_text


def cmd_run(args):
    """Нормализовать test.f и сохранить answer.f."""
    df = pd.read_feather(args.input)
    if "task_text" not in df.columns:
        print("[ERROR] Колонка task_text не найдена")
        sys.exit(1)
    # Применяем нормализацию ко всем строкам
    df["answer"] = df["task_text"].apply(normalize_text)
    output = args.output or args.input.rsplit(".", 1)[0] + "_answer.f"
    df.to_feather(output)
    print(f"[OK] Сохранено {len(df)} строк в {output}")
    print()
    print("═══ EDA ═══")
    eda.report_run(df)


def cmd_evaluate(args):
    """Оценить accuracy на calibration.f.

    Сравнивает результат normalize_text() с ground_truth для каждой строки.
    Выводит Accuracy = доля полностью совпавших строк.
    """
    df = pd.read_feather(args.input)
    if "task_text" not in df.columns or "ground_truth" not in df.columns:
        print("[ERROR] Требуются колонки task_text и ground_truth")
        sys.exit(1)

    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        pred = normalize_text(row["task_text"])
        if pred == row["ground_truth"]:
            correct += 1

    pct = correct / total * 100
    print(f"Accuracy: {correct}/{total} = {pct:.2f}%")
    print()
    print("═══ EDA ═══")
    eda.report_evaluate(df)
    return correct / total


def cmd_errors(args):
    """Показать первые N ошибок на calibration.f."""
    df = pd.read_feather(args.input)
    if "task_text" not in df.columns or "ground_truth" not in df.columns:
        print("[ERROR] Требуются колонки task_text и ground_truth")
        sys.exit(1)

    n = args.n
    errors = []
    for i, row in df.iterrows():
        pred = normalize_text(row["task_text"])
        if pred != row["ground_truth"]:
            errors.append((i, row["task_text"], row["ground_truth"], pred))
            if len(errors) >= n:
                break

    print(f"Найдено ошибок: {len(errors)} (показано {n})")
    for idx, (row_i, task, gt, pred) in enumerate(errors):
        print(f"\n--- {idx + 1}. Row {row_i} ---")
        print(f"TASK: {task}")
        print(f"GT:   {gt}")
        print(f"PRED: {pred}")


def build_parser():
    """Строит argparse с субкомандами."""
    p = argparse.ArgumentParser(
        description="ITN: нормализация чисел в ASR-транскрибациях"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Нормализовать файл и сохранить результат")
    run_p.add_argument("input", help="Путь к .feather файлу")
    run_p.add_argument("-o", "--output", help="Выходной .feather файл")
    run_p.set_defaults(func=cmd_run)

    eval_p = sub.add_parser("evaluate", help="Оценить accuracy на calibration")
    eval_p.add_argument("input", help="Путь к calibration.f")
    eval_p.set_defaults(func=cmd_evaluate)

    err_p = sub.add_parser("errors", help="Показать ошибки на calibration")
    err_p.add_argument("input", help="Путь к calibration.f")
    err_p.add_argument("-n", type=int, default=15, help="Количество ошибок")
    err_p.set_defaults(func=cmd_errors)

    return p


def main():
    """Точка входа: парсит аргументы и вызывает нужную команду."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
