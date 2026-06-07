"""
Token classifier: определяет числовой тип слова через корни и regex.

This is the core of the new sequence-based parser approach.
Replaces the ad-hoc lookup_word + is_ordinal_word + _cardinal_from_root pipeline
with a systematic token classification system.

The classifier returns TokenClass objects that can be fed into sequence_parser.
"""

import re
from dataclasses import dataclass

from src.dicts.hundreds import HUNDREDS
from src.dicts.ordinals import ORDINAL_SET, ORDINALS
from src.dicts.thousands import BILLIONS, MILLIONS, THOUSANDS
from src.dicts.units import TEENS, TENS, UNITS

# ── Token class ──


@dataclass
class TokenClass:
    value: int
    mag: int
    subtype: str  # 'cardinal', 'ordinal', 'multiplier', 'fused', 'vague'
    raw: str
    confidence: float = 1.0


# ── Exact dictionary (fast path) ──

_NUMERAL_DICT = {}
for d in (UNITS, TEENS, TENS, HUNDREDS, THOUSANDS, MILLIONS, BILLIONS):
    _NUMERAL_DICT.update(d)

# Дополнительные ASR-формы, отсутствующие в стандартных словарях
_NUMERAL_DICT.update(
    {
        "двесте": (200, 3),
        "двестипятьсот": (2500, 0),
    }
)

# Known ASR errors — map to canonical form
from src.dicts.asr_errors import ASR_ERRORS

# Multiplier set
_MULTIPLIER_SET = set()
for d in (THOUSANDS, MILLIONS, BILLIONS):
    _MULTIPLIER_SET.update(d.keys())

# Ordinal set for fast lookup
ORDINAL_SET = set(ORDINALS.keys())


# ── Root-based patterns (sorted by length descending) ──

# Each: (root_substring, value, mag, is_mult)
# "десять" has mag=1 (teens), "двадцать"/"сорок"/etc have mag=2 (tens)
# Longest roots first to avoid partial matches (e.g., "пятидесят" before "десят")
_NUMERIC_ROOTS = [
    ("миллиард", 1000000000, 6, True),
    ("миллион", 1000000, 5, True),
    ("тысяч", 1000, 4, True),
    ("тыщ", 1000, 4, True),
    # tens compound forms
    ("пятидесят", 50, 2, False),
    ("пятьдесят", 50, 2, False),
    ("шестидесят", 60, 2, False),
    ("шестьдесят", 60, 2, False),
    ("семидесят", 70, 2, False),
    ("семьдесят", 70, 2, False),
    ("восьмидесят", 80, 2, False),
    ("восемьдесят", 80, 2, False),
    # tens simple
    ("двадцат", 20, 2, False),
    ("тридцат", 30, 2, False),
    ("девяност", 90, 2, False),
    ("десят", 10, 1, False),
    ("сорок", 40, 2, False),
    # hundreds
    ("девятисот", 900, 3, False),
    ("девятьсот", 900, 3, False),
    ("восьмисот", 800, 3, False),
    ("восемьсот", 800, 3, False),
    ("семисот", 700, 3, False),
    ("семьсот", 700, 3, False),
    ("шестисот", 600, 3, False),
    ("шестьсот", 600, 3, False),
    ("пятисот", 500, 3, False),
    ("пятьсот", 500, 3, False),
    ("четырехсот", 400, 3, False),
    ("четырёхсот", 400, 3, False),
    ("четыреста", 400, 3, False),
    ("трехсот", 300, 3, False),
    ("трёхсот", 300, 3, False),
    ("триста", 300, 3, False),
    ("двухсот", 200, 3, False),
    ("двумстам", 200, 3, False),
    ("двумястами", 200, 3, False),
    ("двухстах", 200, 3, False),
    ("двести", 200, 3, False),
    ("двеси", 200, 3, False),
    ("дваста", 200, 3, False),
    ("девятисот", 900, 3, False),
    ("сот", 100, 3, False),
    ("ста", 100, 3, False),
    ("сто", 100, 3, False),
    # units and collectives
    ("четыр", 4, 0, False),
    ("трое", 3, 0, False),
    ("трёх", 3, 0, False),
    ("трех", 3, 0, False),
    ("трем", 3, 0, False),
    ("тремя", 3, 0, False),
    ("три", 3, 0, False),
    ("двое", 2, 0, False),
    ("двух", 2, 0, False),
    ("двум", 2, 0, False),
    ("двумя", 2, 0, False),
    ("две", 2, 0, False),
    ("два", 2, 0, False),
    ("один", 1, 0, False),
    ("одна", 1, 0, False),
    ("одно", 1, 0, False),
    ("одного", 1, 0, False),
    ("одной", 1, 0, False),
    ("одному", 1, 0, False),
    ("одним", 1, 0, False),
    ("одном", 1, 0, False),
    ("одни", 1, 0, False),
    ("одних", 1, 0, False),
    ("пят", 5, 0, False),
    ("шест", 6, 0, False),
    ("сем", 7, 0, False),
    ("восемь", 8, 0, False),
    ("восьм", 8, 0, False),
    ("восем", 8, 0, False),
    ("девят", 9, 0, False),
    ("нол", 0, 0, False),
    ("нул", 0, 0, False),
    # teens
    ("одиннадцат", 11, 1, False),
    ("двенадцат", 12, 1, False),
    ("тринадцат", 13, 1, False),
    ("четырнадцат", 14, 1, False),
    ("пятнадцат", 15, 1, False),
    ("шестнадцат", 16, 1, False),
    ("семнадцат", 17, 1, False),
    ("восемнадцат", 18, 1, False),
    ("девятнадцат", 19, 1, False),
]

# Valid inflection suffixes for Russian numerals
_INFLECTION_SUFFIXES = {
    "",
    "а",
    "у",
    "е",
    "и",
    "ой",
    "ых",
    "ым",
    "ыми",
    "ом",
    "ам",
    "ами",
    "ах",
    "ей",
    "ю",
    "я",
    "ь",
    "ый",
    "ий",
    "ая",
    "ое",
    "ые",
    "ого",
    "его",
    "ому",
    "ему",
    "ую",
    "ья",
    "ье",
    "ьи",
}

_ORDINAL_SUFFIXES = {
    "ый",
    "ий",
    "ой",
    "ая",
    "ья",
    "ое",
    "ье",
    "ые",
    "ьи",
    "ого",
    "его",
    "ому",
    "ему",
    "ым",
    "ими",
    "ыми",
    "ом",
    "ем",
    "ых",
    "ых",
    "их",
    "ую",
    "ю",
    "ей",
}

# Words that look like ordinals but aren't
_NON_ORDINALS = {
    "какой",
    "такой",
    "другой",
    "любой",
    "каждой",
    "самой",
    "самый",
    "самое",
    "новая",
    "новый",
    "простой",
    "главный",
    "большой",
    "хороший",
    "плохой",
    "маленький",
    "маленькая",
    "высокий",
    "низкий",
    "нужный",
    "последний",
    "ближайший",
    "разный",
    "целый",
    "полный",
    "важный",
    "точный",
    "активный",
    "интересный",
    "обычный",
    "подобный",
    "отдельный",
    "значительный",
    "собственный",
    "человеческий",
    "прежний",
    "дополнительный",
    "практический",
    "технический",
    "экономический",
    "политический",
    "исторический",
    "физический",
    "юридический",
    "медицинский",
    "социальный",
    "культурный",
    "научный",
    "международный",
    "современный",
    "второй",
}


# ── Vague context detection for "тыщ"/"тыща" ──

_VAGUE_MARKERS = {"выше", "ниже", "около", "почти"}


def _is_vague_context(prev_tokens):
    """Проверяет, что 'тыщ' в разговорном контексте (не число)."""
    if len(prev_tokens) < 1:
        return False
    prev = prev_tokens[-1]
    if prev in _VAGUE_MARKERS:
        return True
    # "с чем то тыщ"
    if len(prev_tokens) >= 3 and prev_tokens[-3:] == ["с", "чем", "то"]:
        return True
    # "где то тыщ"
    if len(prev_tokens) >= 2 and prev_tokens[-2:] == ["где", "то"]:
        return True
    # "с половиной тыщ"
    if len(prev_tokens) >= 2 and prev_tokens[-2:] == ["с", "половиной"]:
        return True
    return False


# ── Root matching ──


def _match_root(word):
    """Пытается найти числовой корень в слове.

    Сортировка по убыванию длины корня, чтобы "пятидесят" (8)
    матчился раньше, чем "десят" (5).

    Returns:
        (value, mag, is_mult, suffix) or None
    """
    w = word.lower()
    for root, val, mag, is_mult in _NUMERIC_ROOTS:
        idx = w.find(root)
        if idx == -1:
            continue
        # Must match at start of the word
        if idx != 0:
            continue
        suffix = w[len(root) :]
        if suffix in _INFLECTION_SUFFIXES:
            return val, mag, is_mult, suffix
    return None


# ── Ordinal detection ──


def _is_ordinal_suffix(suffix):
    """Проверяет, является ли суффикс порядковым окончанием."""
    return suffix in _ORDINAL_SUFFIXES


def _classify_ordinal(word, val, mag, suffix):
    """Пытается классифицировать слово как ordinal.

    Проверяет:
    1. Слово не в списке исключений
    2. Суффикс — порядковое окончание
    3. Слово не является известным кардинальным числительным
    """
    w = word.lower()
    if w in _NON_ORDINALS:
        return None
    if w in _NUMERAL_DICT:
        return None  # already cardinal, don't reclassify as ordinal
    if _is_ordinal_suffix(suffix):
        return TokenClass(val, mag, "ordinal", word)
    return None


# ── Fused compound detection ──


def _find_fused_compound(word):
    """Пытается разбить слитное слово на два числовых токена.

    "дветысячи" → "две"(2) + "тысячи"(1000)
    "двестипятьсот" → "двести"(200) + "пятьсот"(500) — не числовая конструкция

    Returns:
        list[TokenClass] or None
    """
    w = word.lower()
    for i in range(2, len(w)):
        part1 = w[:i]
        part2 = w[i:]
        r1 = _match_root(part1)
        r2 = _match_root(part2)
        if r1 and r2:
            v1, m1, mult1, _ = r1
            v2, m2, mult2, _ = r2
            # Valid fused compound: unit/digit + multiplier
            if mult2 and m1 <= 3 and m2 >= 4:
                return [
                    TokenClass(v1, m1, "cardinal", word),
                    TokenClass(v2, m2, "multiplier", word),
                ]
    return None


# ── Main classify API ──


def classify(word, prev_tokens=None):
    """Классифицирует слово как числительное.

    Пайплайн:
    1. Точное совпадение в словаре (NUMERAL_DICT) — быстрый путь
    2. ASR-ошибки → каноническая форма → словарь
    3. Root-based regex match
    4. Ordinal suffix check
    5. Fused compound split

    Args:
        word: Слово для классификации
        prev_tokens: Предыдущие токены (для vague контекста)

    Returns:
        list[TokenClass] | None: список из 1..N токенов или None
    """
    if not word or len(word) < 2:
        return None

    w = word.lower()

    # 1. Vague "тыщ"/"тыща" context (check BEFORE dict match)
    if w in ("тыщ", "тыща") and prev_tokens is not None:
        if _is_vague_context(prev_tokens):
            return [TokenClass(1000, 4, "vague", word)]

    # 2. Exact dict match
    if w in _NUMERAL_DICT:
        val, mag = _NUMERAL_DICT[w]
        subtype = "multiplier" if w in _MULTIPLIER_SET else "cardinal"
        return [TokenClass(val, mag, subtype, word)]

    # 3. Ordinal exact match
    if w in ORDINAL_SET:
        val = int(ORDINALS[w])
        return [TokenClass(val, 0, "ordinal", word)]

    # 4. ASR error correction
    if w in ASR_ERRORS:
        canonical = ASR_ERRORS[w]
        if canonical in _NUMERAL_DICT:
            val, mag = _NUMERAL_DICT[canonical]
            subtype = "multiplier" if canonical in _MULTIPLIER_SET else "cardinal"
            return [TokenClass(val, mag, subtype, word)]
        if canonical in ORDINAL_SET:
            val = int(ORDINALS[canonical])
            return [TokenClass(val, 0, "ordinal", word)]

    # 5. Root-based matching
    match = _match_root(w)
    if match:
        val, mag, is_mult, suffix = match
        # Check ordinal first
        ordinal = _classify_ordinal(word, val, mag, suffix)
        if ordinal:
            return [ordinal]
        # Custom "сбор" forms (collective, like "пятеро")
        if suffix in ("о", "е"):
            pass  # Same as cardinal
        subtype = "multiplier" if is_mult else "cardinal"
        return [TokenClass(val, mag, subtype, word)]

    # 6. Fused compound
    fused = _find_fused_compound(word)
    if fused:
        return fused

    return None


def classify_tokens(tokens):
    """Классифицирует все токены строки.

    Args:
        tokens: list[str] — токены строки

    Returns:
        list[list[TokenClass] | None]
    """
    result = []
    for i, token in enumerate(tokens):
        prev = tokens[:i]
        result.append(classify(token, prev))
    return result
