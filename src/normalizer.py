"""Нормализатор текста: обход токенов, выделение числовых групп, замена."""

from src.lexicon import lookup_word, is_ordinal_word, ordinal_value
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
            group_start = i
            while i < len(tokens):
                t = tokens[i]
                lk = lookup_word(t)
                io = is_ordinal_word(t)
                if lk is not None or io:
                    i += 1
                else:
                    break
            group_end = i

            group_data = []
            for j in range(group_start, group_end):
                t = tokens[j]
                lk = lookup_word(t)
                io = is_ordinal_word(t)
                if io:
                    val_str = ordinal_value(t)
                    group_data.append((int(val_str), 0, False, True))
                elif lk is not None:
                    val, mag, is_mult = lk
                    group_data.append((val, mag, is_mult, False))

            parsed = parse_number_group(group_data)
            result_tokens.extend(parsed)
        else:
            result_tokens.append(token)
            i += 1

    return ' '.join(result_tokens)
