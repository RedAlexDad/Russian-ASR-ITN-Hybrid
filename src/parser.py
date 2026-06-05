"""Парсер числовых последовательностей.

Различает суммирование (две тысячи пятьсот → 2500)
от перечисления (двести триста → 200 300).
"""


def parse_number_group(tokens_data):
    """Парсит группу числовых токенов и возвращает список чисел.

    Каждый token_data: (val, mag, is_mult, is_ordinal)

    Алгоритм:
    - Строго убывающая magnitude → добавляем к текущему числу (сумма)
    - Неубывающая magnitude → завершаем число, начинаем новое (перечисление)
    - Умножитель (тысяча/миллион) → умножает накопленное значение
    - Если после умножителя идёт другой умножитель того же уровня → новое число

    Примеры:
      (200,3)+(50,2)+(1000,4)        → ['250000']
      (200,3)+(300,3)                → ['200', '300']
      (2,0)+(1000,4)+(80,1)+(5,0)    → ['2085']
      (70,2)+(5e6,5)+(2,0)+(1e6,5)   → ['70000000', '2000000']
    """
    if not tokens_data:
        return []

    is_standalone_mult = (len(tokens_data) == 1 and tokens_data[0][2])

    compound = 0
    current = 0
    last_mag = -1
    last_mult_mag = -1
    result = []

    for val, mag, is_mult, _ in tokens_data:
        if is_mult:
            if last_mult_mag == mag and current == 0:
                result.append(str(int(compound)))
                compound = 0
                compound = val
            else:
                multiplied = current * val if current > 0 else val
                compound += multiplied
                current = 0
            last_mult_mag = mag
            last_mag = -1
        else:
            if last_mag == -1:
                current = val
                last_mag = mag
            elif mag < last_mag:
                current += val
                last_mag = mag
            else:
                if compound > 0 or current > 0:
                    result.append(str(int(compound + current)))
                compound = 0
                current = val
                last_mag = mag
                last_mult_mag = -1

    if is_standalone_mult:
        v = tokens_data[0][0]
        return [str(int(v))] if v in (1000, 1000000, 1000000000) else []

    total = compound + current
    if total > 0 or (len(tokens_data) >= 1 and any(td[0] == 0 for td in tokens_data)):
        result.append(str(int(total)))

    return result
