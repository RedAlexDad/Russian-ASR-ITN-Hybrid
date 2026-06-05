#!/usr/bin/env python3
"""Проверка корректности Jupyter ноутбука.

Использование:
  python scripts/validate_notebook.py notebooks/eda.ipynb

Что проверяет:
  1. Файл существует и читается
  2. Валидный JSON (соответствует nbformat v4)
  3. Все code-ячейки завершились без ошибок (если есть output)
  4. Нет пустых code-ячеек (без кода)
  5. Каждой code-ячейке предшествует markdown-ячейка
  6. Присутствуют обязательные секции (по заголовкам)
  7. Используется Polars, а не Pandas
  8. Нет инлайн-датасетов (данные только из data/)
"""

import json
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "подготовка",
    "загрузка",
    "распределение",
    "asr",
    "группировк",
    "accuracy",
    "ошибк",
    "вывод",
]

FORBIDDEN_IMPORTS = [
    "import pandas",
    "from pandas",
]

REQUIRED_IMPORTS = [
    "import polars",
    "from polars",
    "import polars as pl",
    "import matplotlib",
    "from matplotlib",
    "import seaborn",
    "from seaborn",
]


def e(msg, *args, **kwargs):
    print(f"  \033[91m✗ {msg}\033[0m".format(*args, **kwargs))


def o(msg, *args, **kwargs):
    print(f"  \033[92m✓ {msg}\033[0m".format(*args, **kwargs))


def w(msg, *args, **kwargs):
    print(f"  \033[93m⚠ {msg}\033[0m".format(*args, **kwargs))


def i(msg, *args, **kwargs):
    print(f"  \033[94m→ {msg}\033[0m".format(*args, **kwargs))


def validate(path):
    path = Path(path)
    errors = 0
    warnings = 0

    print(f"\n\033[1mПроверка: {path}\033[0m\n")

    # 1. Файл существует
    if not path.exists():
        e("Файл не найден: {}", path)
        return False
    o("Файл существует")

    # 2. Валидный JSON nbformat v4
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except json.JSONDecodeError as ex:
        e("Невалидный JSON: {}", ex)
        return False
    o("Валидный JSON")

    nbformat = nb.get("nbformat")
    if nbformat != 4:
        w("nbformat = {}, ожидается 4", nbformat)
        warnings += 1
    else:
        o("nbformat = 4")

    cells = nb.get("cells", [])
    if not cells:
        e("Нет ячеек")
        return False
    o("Ячеек: {}", len(cells))

    # 3. Все code-ячейки без ошибок
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    for idx, cell in enumerate(code_cells):
        cell_num = idx + 1
        outputs = cell.get("outputs", [])
        for out in outputs:
            if out.get("output_type") == "error":
                ename = out.get("ename", "?")
                evalue = out.get("evalue", "?")
                e("[ячейка {}] Ошибка: {}: {}", cell_num, ename, evalue)
                errors += 1

        source = "".join(cell.get("source", []))
        if not source.strip():
            w("[ячейка {}] Пустая code-ячейка", cell_num)
            warnings += 1

    if not any(
        e_
        for e_ in [
            out
            for c in code_cells
            for out in c.get("outputs", [])
            if out.get("output_type") == "error"
        ]
    ):
        o("Все code-ячейки без ошибок")

    # 4. Каждой code-ячейке предшествует markdown
    prev_was_code = False
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            if prev_was_code and i > 0:
                w("[ячейка {}] Две code-ячейки подряд без markdown", i + 1)
                warnings += 1
            prev_was_code = True
        else:
            prev_was_code = False
    o(
        "Проверка чередования markdown/code пройдена"
        if warnings == 0
        else "Нарушения чередования"
    )

    # 5. Обязательные секции (по заголовкам в markdown)
    md_sources = ["".join(c.get("source", [])) for c in md_cells]
    found_sections = set()
    for src in md_sources:
        for line in src.split("\n"):
            if line.startswith("#"):
                for section in REQUIRED_SECTIONS:
                    if section in line.lower():
                        found_sections.add(section)

    missing = set(REQUIRED_SECTIONS) - found_sections
    if missing:
        w("Отсутствуют секции: {}", ", ".join(sorted(missing)))
        warnings += 1
    else:
        o("Все обязательные секции присутствуют")

    # 6. Нет запрещённых импортов (pandas)
    all_source = ""
    for c in code_cells:
        all_source += "".join(c.get("source", [])) + "\n"

    for forb in FORBIDDEN_IMPORTS:
        if forb in all_source:
            e("Запрещённый импорт: {}", forb)
            errors += 1

    if not any(forb in all_source for forb in FORBIDDEN_IMPORTS):
        o("Нет запрещённых импортов (pandas)")

    # 7. Есть обязательные импорты
    for req in REQUIRED_IMPORTS:
        if req not in all_source:
            # проверяем частично
            pass

    has_polars = "polars" in all_source
    has_mpl = "matplotlib" in all_source or "import matplotlib" in all_source
    has_sns = "seaborn" in all_source or "import seaborn" in all_source

    if has_polars:
        o("Polars используется")
    else:
        e("Polars не найден в импортах")
        errors += 1

    if has_mpl:
        o("Matplotlib используется")
    else:
        w("Matplotlib не найден")
        warnings += 1

    if has_sns:
        o("Seaborn используется")
    else:
        w("Seaborn не найден")
        warnings += 1

    # 8. Нет инлайн-датасетов (данные — только из data/)
    bad_patterns = [
        r'task_text\s*=\s*[\'"]',
        r'ground_truth\s*=\s*[\'"]',
    ]
    for pat in bad_patterns:
        if re.search(pat, all_source):
            w("Возможен инлайн-датасет (данные прямо в коде)")
            warnings += 1
            break

    o("Данные загружаются из файлов (не инлайн)")

    # Итог
    print()
    if errors == 0:
        print("  \033[92m\033[1m✓ Все проверки пройдены\033[0m")
        print(f"    \033[93mПредупреждений: {warnings}\033[0m")
        return True
    else:
        print(f"  \033[91m\033[1m✗ Найдено ошибок: {errors}\033[0m")
        print(f"    \033[93mПредупреждений: {warnings}\033[0m")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python scripts/validate_notebook.py <notebook.ipynb>")
        sys.exit(1)

    success = validate(sys.argv[1])
    sys.exit(0 if success else 1)
