"""
Дисамбигуация: определение части речи для слов, которые могут быть
как числительными, так и другими частями речи.

Использует pymorphy2 для POS-теггинга контекста.
Если pymorphy2 не установлен — работает через эвристики.
"""

import re

# ── Эвристики для известных омонимичных слов ──

# "сто" — 100 (числительное) vs "стоять" (глагол)
#   Числительное: перед существительным в род.п. ("сто рублей")
#   Глагол: после "я" ("я сто"), перед "на/в/у" ("сто на")
_STO_VERB_PATTERNS = [
    (r"\bя\s+сто\b", False),  # "я сто" → verb
    (r"\bсто\s+(на|в|у|за|перед)\b", False),  # "сто на" → verb
    (r"\bсто\s+и\s+", False),  # "сто и" → verb listing
]
_STO_NUMERAL_PATTERNS = [
    (r"\bсто\s+(рубл|доллар|евро|тысяч|миллион|процент|раз|человек|штук)", True),
    (r"\b(до|около|более|менее|свыше|почти)\s+сто\b", True),
]

# "три" — 3 vs "тереть" (глагол)
#   Числительное: с существительным
#   Глагол: в контексте "три/тереть" с возвратными частицами
_TRI_VERB_PATTERNS = [
    (r"\bтри\s+ся\b", False),
    (r"\bтри\s+те\b", False),
]

# "сорок" — 40 vs "сорока" (сорока)
#   Числительное: с единицами измерения
#   Птица: редкий контекст в e-commerce
_SOROK_NOUN_PATTERNS = [
    (r"\bсорок\s+(летит|птиц|синиц|вороб)", False),
]


def _context_match(text, patterns):
    """Проверяет текст на совпадение с паттернами, возвращает первое совпадение."""
    for pattern, is_num in patterns:
        if re.search(pattern, text):
            return is_num
    return None


def is_likely_numeric(text, word, pos):
    """Определяет, является ли слово в данном контексте числительным.

    Args:
        text: Полный исходный текст
        word: Слово для проверки
        pos: Позиция слова в токенах (индекс)

    Returns:
        True если слово скорее числительное, False если нет
    """
    w = word.lower()

    if w == "сто":
        # Сначала проверяем специфические паттерны
        result = _context_match(text, _STO_NUMERAL_PATTERNS + _STO_VERB_PATTERNS)
        if result is not None:
            return result
        # Если нет явных маркеров — проверяем, является ли "сто" числительным
        # по pymorphy2
        try:
            return _pymorphy_check(text, word, pos)
        except Exception:
            return True  # по умолчанию — числительное

    if w == "сорок":
        result = _context_match(text, _SOROK_NOUN_PATTERNS)
        if result is not None:
            return result
        return True

    if w == "три" or w == "тру":
        result = _context_match(text, _TRI_VERB_PATTERNS)
        if result is not None:
            return result
        return True

    return True  # По умолчанию — числительное


def _pymorphy_check(text, word, pos):
    """Использует pymorphy2 для проверки части речи."""
    # Monkey-patch для Python 3.12
    import collections
    import inspect

    if not hasattr(inspect, "getargspec"):
        ArgSpec = collections.namedtuple(
            "ArgSpec", ["args", "varargs", "keywords", "defaults"]
        )

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(
                args=spec.args,
                varargs=spec.varargs,
                keywords=spec.varkw,
                defaults=spec.defaults,
            )

        inspect.getargspec = getargspec

    import pymorphy2

    try:
        morph = pymorphy2.MorphAnalyzer()
    except Exception:
        return True

    parses = morph.parse(word.lower())
    has_numr = any("NUMR" in str(p.tag) for p in parses)
    has_non_numr = any("NUMR" not in str(p.tag) and p.score > 0.1 for p in parses)

    if has_numr and not has_non_numr:
        return True  # только числительное
    if not has_numr and has_non_numr:
        return False  # не числительное
    # амбивалентно — полагаемся на контекст из эвристик
    return True
