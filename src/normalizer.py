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

import re

from src.lexicon import is_ordinal_word, lookup_word, ordinal_value
from src.parser import parse_number_group

try:
    from src.disambiguate import is_likely_numeric
except ImportError:
    def is_likely_numeric(*_):
        return True


# ── ASR regex-препроцессинг ──

_HUNDRED_TO_TEN = {
    "триста": "тридцать",
    "четыреста": "сорок",
    "пятьсот": "пятьдесят",
    "шестьсот": "шестьдесят",
    "семьсот": "семьдесят",
    "восемьсот": "восемьдесят",
    "девятьсот": "девяносто",
}

_TEN_TO_UNIT = {
    "пятьдесят": "пять",
    "шестьдесят": "шесть",
    "семьдесят": "семь",
    "восемьдесят": "восемь",
    "девяносто": "девять",
}

_ASR_SUBSTITUTIONS = [
    # "двеси" + hundred + (ten) + тысяч: сдвиг разрядов
    # "двеси триста пятьдесят тысяч" → "двести тридцать пять тысяч"
    (re.compile(r'\bдвеси\s+(триста|четыреста|пятьсот|шестьсот|семьсот|восемьсот|девятьсот)(?:\s+(пятьдесят|шестьдесят|семьдесят|восемьдесят))?\s+тысяч[аи]?\b'),
     lambda m: " ".join(
         ["двести", _HUNDRED_TO_TEN.get(m.group(1), m.group(1))]
         + ([_TEN_TO_UNIT.get(m.group(2))] if m.group(2) in _TEN_TO_UNIT else [])
         + ["тысяч"]
     )),
    # Пропущенный мягкий знак в десятках
    (re.compile(r'\bпятдесят\b'), 'пятьдесят'),
    (re.compile(r'\bшестдесят\b'), 'шестьдесят'),
    (re.compile(r'\bсемдесят\b'), 'семьдесят'),
    (re.compile(r'\bвосемдесят\b'), 'восемьдесят'),
    # Падежные ошибки в сотнях
    (re.compile(r'\bтристо\b'), 'триста'),
    (re.compile(r'\bчетыриста\b'), 'четыреста'),
]


def _asr_preprocess(text):
    """Корректирует известные ASR-ошибки в числовых последовательностях до парсинга."""
    for pattern, replacement in _ASR_SUBSTITUTIONS:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)
    return text


def _is_vague_tyt_context(tokens, i):
    """Проверяет, что 'тыщ' по индексу i в разговорном контексте (не число).

    "с чем то тыщ", "выше тыщ", "с половиной тыщ" — ASR-транскрипт, не число.
    """
    if i < 1:
        return False
    prev = tokens[i - 1]
    if prev in ("выше", "ниже", "около"):
        return True
    if prev == "половиной" or (prev == "с" and i >= 2 and tokens[i - 2] == "половиной"):
        return True
    if i >= 3 and tokens[i - 3] == "с" and tokens[i - 2] == "чем" and tokens[i - 1] == "то":
        return True
    if i >= 2 and tokens[i - 2] == "где" and tokens[i - 1] == "то":
        return True
    return False


_FUSED_COMPOUNDS = {
    "дветысячи": [(2, 0, False, False), (1000, 4, True, False)],
}


def normalize_text(text):
    """Преобразует словесную запись чисел в цифровую.

    Находит все последовательности числовых токенов (включая порядковые),
    группирует их по правилам суммы/перечисления и заменяет на цифры.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    text = _asr_preprocess(text)
    tokens = text.split()
    result_tokens = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # "тыщ" в разговорном контексте — не число, оставляем как есть
        if token in ("тыщ", "тыща") and _is_vague_tyt_context(tokens, i):
            result_tokens.append(token)
            i += 1
            continue

        lookup = lookup_word(token)
        is_ord = is_ordinal_word(token)

        # Дисамбигуация: слова вроде "сто" могут быть не числительными
        if lookup is not None and not is_ord:
            if not is_likely_numeric(text, token, i):
                result_tokens.append(token)
                i += 1
                continue

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
                    if t in _FUSED_COMPOUNDS:
                        group_data.extend(_FUSED_COMPOUNDS[t])
                    else:
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
