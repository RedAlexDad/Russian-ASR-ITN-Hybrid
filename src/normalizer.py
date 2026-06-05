"""
Нормализатор текста: обход токенов, выделение числовых групп, замена.

Это главная функция пайплайна normalize_text().
Она принимает сырой текст из ASR и возвращает текст с числами в цифровом виде.

Алгоритм:
  1. Разбиваем текст на токены (по пробелам)
  2. Идём по токенам слева направо
  3. Если токен — числительное (количественное или порядковое):
     - Собираем все следующие числовые токены в группу
     - Передаём группу в parser.parse_number_group()
     - Вставляем результат (цифры) вместо слов
  4. Если токен — не числительное: оставляем как есть
  5. Склеиваем токены обратно в строку

Важно: мы не меняем не-числовые токены вообще.
Это требование метрики Accuracy — любое изменение не-числа считается ошибкой.
"""

from src.lexicon import is_ordinal_word, lookup_word, ordinal_value
from src.parser import parse_number_group


def normalize_text(text):
    """Преобразует словесную запись чисел в цифровую.

    Находит все последовательности числовых токенов (включая порядковые),
    группирует их по правилам суммы/перечисления и заменяет на цифры.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    tokens = text.split()
    result_tokens = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        lookup = lookup_word(token)
        is_ord = is_ordinal_word(token)

        if lookup is not None or is_ord:
            # Нашли числовой токен — собираем группу
            group_start = i
            while i < len(tokens):
                t = tokens[i]
                lk = lookup_word(t)
                io = is_ordinal_word(t)
                if lk is not None or io:
                    i += 1
                else:
                    break

            # Строим данные для парсера
            group_data = []
            for j in range(group_start, i):
                t = tokens[j]
                lk = lookup_word(t)
                io = is_ordinal_word(t)
                if io:
                    val_str = ordinal_value(t)
                    assert val_str is not None
                    group_data.append((int(val_str), 0, False, True))
                elif lk is not None:
                    val, mag, is_mult = lk
                    group_data.append((val, mag, is_mult, False))

            # Парсим и заменяем
            parsed = parse_number_group(group_data)
            result_tokens.extend(parsed)
        else:
            # Не числовой токен — оставляем как есть
            result_tokens.append(token)
            i += 1

    return " ".join(result_tokens)
