"""
Sequence parser: state machine над TokenClass для построения чисел.

Логика:
  - Сканируем токены слева направо
  - В зависимости от mag/subtype решаем: сумма/умножение/перечисление
  - Результат: список строк-чисел

Правила:
  Vague         → пропустить
  Multiplier    → compound += (current or 1) × val
  Cardinal      → mag < prev_mag  → sum (current += val)
                  mag >= prev_mag → enum (flush, start new)
  Ordinal       → то же, что cardinal, результат маркируется порядковым
  Fused         → обрабатывается как обычный cardinal/multiplier
"""

from src.token_classifier import TokenClass


def _flush(compound, current, result, force=False):
    """Сбрасывает накопленное число в результат."""
    total = compound + current
    if total > 0 or force:
        result.append(str(total))


def parse_sequence(classes):
    """Парсит группу TokenClass и возвращает список строк-чисел.

    Вход:  list[TokenClass]
    Выход: list[str]  например ['2000'] или ['70000000', '2000000']
    """
    if not classes:
        return []

    compound = 0      # накоплено после умножителей
    current = 0       # текущее накопленное число (< 1000)
    last_mag = -1     # mag предыдущего cardinal/ordinal токена
    last_mult_mag = -1  # mag последнего умножителя
    has_zero = False   # был ли токен со значением 0
    result = []

    for idx, tc in enumerate(classes):
        # ── Vague: пропускаем ──
        if tc.subtype == 'vague':
            continue

        # ── Ноль: запоминаем ──
        if tc.value == 0:
            has_zero = True

        # ── Multiplier (тысяча, миллион, миллиард) ──
        if tc.subtype == 'multiplier':
            if last_mult_mag == tc.mag:
                # Два умножителя одного ранга → enum
                # "семьдесят миллионов два миллиона" → 7e7 2e6
                # Эмитируем только compound (без current), current — начало нового числа
                _flush(compound, 0, result, force=has_zero)
                compound = current * tc.value if current > 0 else tc.value
                current = 0
            else:
                multiplied = current * tc.value if current > 0 else tc.value
                compound += multiplied
                current = 0
            last_mult_mag = tc.mag
            last_mag = -1
            continue

        # ── Cardinal / Ordinal / Fused ──
        # В этой ветке: tc.mag — разряд числа
        if last_mag == -1:
            # Первый токен (или после умножителя)
            current = tc.value
            last_mag = tc.mag
            continue

        if tc.mag < last_mag:
            # mag понижается → сумма:
            # "двести"(3) + "пятьдесят"(2) → 200+50 = 250

            # Lookahead heuristic: если текущий mag=0/1 И следующий тоже mag=0/1
            # "двести два два" → 200 (3) + 2 (0) + 2 (0) → 200 2 2, а не 202 2
            # "двадцать два" → 20 (2) + 2 (0) → 22 (один mag=0 → нет lookahead)
            if tc.mag <= 1:
                # Lookahead: если следующий токен тоже mag<=1 И не ordinal — enum
                # "двести два два" → 200 2 2 (mag 0→0 → enum)
                # "двести восемьдесят четвёртый" → 284 (mag 1→0[ordinal] → sum)
                if idx < len(classes) - 1:
                    next_tc = classes[idx + 1]
                    if (next_tc.subtype not in ('multiplier', 'vague', 'ordinal')
                            and next_tc.mag <= 1):
                        _flush(compound, current, result, force=has_zero)
                        compound = 0
                        current = tc.value
                        last_mag = tc.mag
                        last_mult_mag = -1
                        continue
            current += tc.value
            last_mag = tc.mag
        else:
            # mag не понижается → enum
            # "двести"(3) + "триста"(3) → 200 300
            # "пять"(0) + "пять"(0) → 5 5
            _flush(compound, current, result, force=has_zero)
            compound = 0
            current = tc.value
            last_mag = tc.mag
            last_mult_mag = -1

    # ── Финализация ──
    _flush(compound, current, result, force=has_zero)
    if not result and has_zero:
        return ['0']

    return result
