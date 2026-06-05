"""Сборка общего словаря числительных и функции поиска."""

from src.dicts.asr_errors import ASR_ERRORS
from src.dicts.hundreds import HUNDREDS
from src.dicts.ordinals import ORDINAL_SET, ORDINALS
from src.dicts.thousands import BILLIONS, MILLIONS, MULTIPLIERS, THOUSANDS
from src.dicts.units import TEENS, TENS, UNITS

# ──────────────────────────────────────────────
# Полный словарь числительных
# ──────────────────────────────────────────────

NUMERAL_DICT = {}
for d in (UNITS, TEENS, TENS, HUNDREDS, THOUSANDS, MILLIONS, BILLIONS):
    NUMERAL_DICT.update(d)

# Дополнительные слитные формы (ASR-склейки)
NUMERAL_DICT.update(
    {
        "дветысячи": (2000, 0),
        "двесте": (200, 3),
        "двестипятьсот": (2500, 0),
    }
)


def lookup_word(word):
    """Ищет слово в словаре числительных, порядковых и ASR-ошибок.
    Возвращает (значение, порядок_величины, is_multiplier) или None.
    """
    w = word.lower()
    if w in NUMERAL_DICT:
        val, mag = NUMERAL_DICT[w]
        return (val, mag, w in MULTIPLIERS)
    if w in ASR_ERRORS:
        canonical = ASR_ERRORS[w]
        if canonical in NUMERAL_DICT:
            val, mag = NUMERAL_DICT[canonical]
            return (val, mag, canonical in MULTIPLIERS)
    return None


def is_ordinal_word(word):
    """Проверяет, является ли слово порядковым числительным."""
    return word.lower() in ORDINAL_SET


def ordinal_value(word):
    """Возвращает цифровое значение порядкового числительного."""
    return ORDINALS.get(word.lower())
