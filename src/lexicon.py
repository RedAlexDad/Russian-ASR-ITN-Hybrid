"""
Сборка общего словаря числительных и функции поиска.

Этот модуль — единая точка входа для всех словарных данных.
Вместо того чтобы импортировать 5 разрозненных словарей,
потребители (parser, normalizer) работают через 3 функции:
  lookup_word(word) -> (val, mag, is_mult) | None
  is_ordinal_word(word) -> bool
  ordinal_value(word) -> str | None

Это изолирует логику поиска от структуры хранения.
Если мы захотим перейти на Trie или Levenshtein — меняем только этот модуль.
"""

from src.dicts.asr_errors import ASR_ERRORS
from src.dicts.hundreds import HUNDREDS
from src.dicts.ordinals import ORDINAL_SET, ORDINALS
from src.dicts.thousands import BILLIONS, MILLIONS, MULTIPLIERS, THOUSANDS
from src.dicts.units import TEENS, TENS, UNITS

# ──────────────────────────────────────────────
# Полный словарь числительных: объединяем все словари в один
# ──────────────────────────────────────────────

NUMERAL_DICT = {}
for d in (UNITS, TEENS, TENS, HUNDREDS, THOUSANDS, MILLIONS, BILLIONS):
    NUMERAL_DICT.update(d)

# Дополнительные слитные формы — результат ASR-склеек
# "дветысячи" вместо "две тысячи", "двестипятьсот" вместо "двести пятьсот"
NUMERAL_DICT.update(
    {
        "дветысячи": (2000, 0),
        "двесте": (200, 3),
        "двестипятьсот": (2500, 0),
    }
)


def lookup_word(word):
    """Ищет слово в словаре числительных.

    Возвращает (значение, порядок_величины, is_multiplier) или None.

    Алгоритм поиска:
    1. Прямой поиск в NUMERAL_DICT (O(1) по хешу)
    2. Если не найдено — поиск в ASR_ERRORS (O(1))
       и повторный поиск канонической формы
    3. Если ничего не найдено — None

    is_multiplier=True означает, что это слово-умножитель
    (тысяча, миллион), который в парсере умножает накопленное значение.
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
    """Проверяет, является ли слово порядковым числительным.

    Использует ORDINAL_SET (множество) для O(1) проверки.
    """
    return word.lower() in ORDINAL_SET


def ordinal_value(word):
    """Возвращает цифровое значение порядкового числительного.

    Примеры:
      ordinal_value("первого") -> "1"
      ordinal_value("двадцать") -> "20"
    """
    return ORDINALS.get(word.lower())
