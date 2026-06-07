"""
Парсер числовых последовательностей.

Ядро решения. Различает два случая по разрядам (magnitude) соседних слов:

1. Сумма — разряд понижается:
   "двести пятьдесят"   сотни(3) → десятки(2)  = 200 + 50  = 250
   "триста шестьдесят"  сотни(3) → десятки(2)  = 300 + 60  = 360

2. Перечисление — разряд не меняется:
   "двести триста"      сотни(3) → сотни(3)    = 200   300 (два числа)
   "двадцать тридцать"  дес(2)   → дес(2)      = 20    30

Умножители (тысяча, миллион):
   "пять тысяч"   = 5 × 1000 = 5000
   После умножителя накопление продолжается:
   "пять тысяч двести" = (5 × 1000) + 200 = 5200
"""


def parse_number_group(tokens_data):
    """Парсит группу числовых токенов и возвращает список чисел-строк.

    Вход: список кортежей (val, mag, is_mult, is_ordinal)
      val       — числовое значение токена (1, 2, 10, 100, 1000...)
      mag       — разряд числа (0=единицы, 1=10-19, 2=десятки, 3=сотни, 4=тысячи...)
      is_mult   — True если это умножитель (тысяча, миллион)
      is_ordinal — True если это порядковое числительное

    Выход: список строк ['2500'] или ['200', '300']

    Правила:
      Разряд понижается (mag 3 → 2)  → СУММА:     current += val
      Разряд не меняется (mag 3 → 3) → ПЕРЕЧИСЛЕНИЕ: сохраняем compound+current, начинаем новое
      Умножитель (тысяча/миллион)    → УМНОЖЕНИЕ:  current × val → compound

    Примеры:
      двести(200,3) + пятьдесят(50,2)        → 3 > 2 → сумма   → 250
      двести(200,3) + триста(300,3)          → 3 = 3 → перечисл → 200 300
      две(2,0) + тысячи(1000,4)              → умножитель      → 2000
      пять(5,0) + тысяч(1000,4) + двести(200,3) → ×1000 + 200 → 5200
    """
    if not tokens_data:
        return []

    # Флаг: группа состоит из одного слова-умножителя (например, просто "тысяча")
    is_standalone_mult = len(tokens_data) == 1 and tokens_data[0][2]

    compound = 0  # накопленная сумма после умножений
    current = 0  # текущее накапливаемое число (< 1000)
    last_mag = -1  # mag предыдущего токена
    last_mult_mag = -1  # mag последнего умножителя
    result = []

    for j, (val, mag, is_mult, _) in enumerate(tokens_data):
        if is_mult:
            if last_mult_mag == mag:
                # Два умножителя одного ранга — разные числа
                # "семьдесят миллионов два миллиона" → 70M 2M
                result.append(str(int(compound)))
                compound = current * val if current > 0 else val
                current = 0
            else:
                multiplied = current * val if current > 0 else val
                compound += multiplied
                current = 0
            last_mult_mag = mag
            last_mag = -1
        else:
            if last_mag == -1:
                # Первый токен (или после умножителя)
                current = val
                last_mag = mag
            elif mag < last_mag:
                # Два токена подряд с одинаковым mag=0 → перечисление
                # Ловит "два два", "два три" как отдельные числа
                if (
                    mag <= 1
                    and j + 1 < len(tokens_data)
                    and tokens_data[j + 1][1] == mag
                ):
                    result.append(str(int(compound + current)))
                    compound = 0
                    current = val
                    last_mag = mag
                    last_mult_mag = -1
                else:
                    # Строго убывающая magnitude -> сумма
                    current += val
                    last_mag = mag
            else:
                # Не убывает -> перечисление: завершаем текущее число
                result.append(str(int(compound + current)))
                compound = 0
                current = val
                last_mag = mag
                last_mult_mag = -1

    # Обработка остатка после цикла
    if is_standalone_mult:
        v = tokens_data[0][0]
        return [str(int(v))] if v in (1000, 1000000, 1000000000) else []

    total = compound + current
    if total > 0 or (len(tokens_data) >= 1 and any(td[0] == 0 for td in tokens_data)):
        result.append(str(int(total)))

    return result
