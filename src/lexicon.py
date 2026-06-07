"""
Сборка общего словаря числительных и функции поиска.

Адаптер: делегирует классификацию в token_classifier, сохраняя
обратную совместимость с lookup_word, is_ordinal_word, ordinal_value.

Новый код должен использовать token_classifier.classify() напрямую.
"""

# Re-export for backward compatibility
from src.dicts.hundreds import HUNDREDS
from src.dicts.ordinals import ORDINAL_SET, ORDINALS
from src.dicts.thousands import BILLIONS, MILLIONS, MULTIPLIERS, THOUSANDS
from src.dicts.units import TEENS, TENS, UNITS
from src.token_classifier import TokenClass
from src.token_classifier import classify as _classify

# Полный словарь числительных (быстрый путь для старого кода)
NUMERAL_DICT = {}
for d in (UNITS, TEENS, TENS, HUNDREDS, THOUSANDS, MILLIONS, BILLIONS):
    NUMERAL_DICT.update(d)

# Дополнительные слитные формы
NUMERAL_DICT.update(
    {
        "дветысячи": (2000, 0),
        "двесте": (200, 3),
        "двестипятьсот": (2500, 0),
    }
)

from src.dicts.asr_errors import ASR_ERRORS


def lookup_word(word):
    """Ищет слово в словаре числительных.

    Возвращает (значение, порядок_величины, is_multiplier) или None.
    Делегирует в token_classifier.classify().
    """
    w = word.lower()
    # Fast path: прямой поиск в NUMERAL_DICT
    if w in NUMERAL_DICT:
        val, mag = NUMERAL_DICT[w]
        return (val, mag, w in MULTIPLIERS)
    if w in ASR_ERRORS:
        canonical = ASR_ERRORS[w]
        if canonical in NUMERAL_DICT:
            val, mag = NUMERAL_DICT[canonical]
            return (val, mag, canonical in MULTIPLIERS)

    # Root-based классификация
    result = _classify(word)
    if result and not any(t.subtype == "vague" for t in result):
        tc = result[0]
        return (tc.value, tc.mag, tc.subtype == "multiplier")
    return None


def is_ordinal_word(word):
    """Проверяет, является ли слово порядковым числительным."""
    w = word.lower()
    if w in ORDINAL_SET:
        return True
    if w in NUMERAL_DICT or w in ASR_ERRORS:
        return False
    result = _classify(word)
    if result:
        return result[0].subtype == "ordinal"
    return False


def ordinal_value(word):
    """Возвращает цифровое значение порядкового числительного."""
    w = word.lower()
    if w in ORDINALS:
        return ORDINALS[w]
    if w in NUMERAL_DICT or w in ASR_ERRORS:
        return None
    result = _classify(word)
    if result and result[0].subtype == "ordinal":
        return str(result[0].value)
    return None
