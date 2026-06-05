"""
Парсер числовых последовательностей.

Ядро решения. Различает два принципиально разных случая:

1. Суммирование (группировка): строго убывающий порядок величины
   "две тысячи пятьсот" = 2000 + 500 = 2500
   "двести пятьдесят"   = 200  + 50  = 250
   Mag: 3 > 2 — убывает -> сумма

2. Перечисление (надиктовка): одинаковый или возрастающий порядок
   "двести триста"      = 200    300    (два числа подряд)
   "двадцать тридцать"   = 20     30
   Mag: 3 == 3 — не убывает -> перечисление

Умножители (тысяча, миллион): особый случай.
  "пять тысяч" = 5 * 1000 = 5000
  После умножителя накопление продолжается: "пять тысяч двести" = 5000 + 200 = 5200
"""


def parse_number_group(tokens_data):
    """Парсит группу числовых токенов и возвращает список чисел-строк.

    Вход: список кортежей (val, mag, is_mult, is_ordinal)
      val       — числовое значение токена (1, 2, 10, 100, 1000...)
      mag       — порядок величины (0=ед, 1=10-19, 2=десятки, 3=сотни, 4=тысячи...)
      is_mult   — True если это умножитель (тысяча, миллион)
      is_ordinal — True если это порядковое числительное

    Выход: список строк ['2500'] или ['200', '300']

    Алгоритм:
      compound — накопленная сумма после умножений (тысячи/миллионы)
      current  — текущее накапливаемое число (единицы+десятки+сотни)
      last_mag — mag последнего обработанного токена

      Для каждого токена:
        Если умножитель:
          current * val -> compound += результат
        Иначе если mag строго убывает:
          current += val (сумма)
        Иначе:
          compound + current -> сохраняем в результат, начинаем новое число
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

    for val, mag, is_mult, _ in tokens_data:
        if is_mult:
            # Умножитель: умножаем накопленное current * val
            # Если current == 0, значит умножитель без предшествующего числа
            if last_mult_mag == mag and current == 0:
                # Два умножителя одного уровня подряд — новое число
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
                # Первый токен (или после умножителя)
                current = val
                last_mag = mag
            elif mag < last_mag:
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
