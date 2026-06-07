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


# ── Динамическое расширение словаря через корни ──

_NUMERIC_ROOTS = [
    ("миллиард", 1000000000, 6, True),
    ("миллион", 1000000, 5, True),
    ("тысяч", 1000, 4, True),
    ("пятидесят", 50, 2, False),
    ("шестидесят", 60, 2, False),
    ("семидесят", 70, 2, False),
    ("восьмидесят", 80, 2, False),
    ("двадцат", 20, 2, False),
    ("тридцат", 30, 2, False),
    ("девяност", 90, 2, False),
    ("десят", 10, 1, False),
    ("сорок", 40, 2, False),
    ("девятисот", 900, 3, False),
    ("восьмисот", 800, 3, False),
    ("семисот", 700, 3, False),
    ("шестисот", 600, 3, False),
    ("пятисот", 500, 3, False),
    ("четырехсот", 400, 3, False),
    ("трехсот", 300, 3, False),
    ("двухсот", 200, 3, False),
    ("сот", 100, 3, False),
]

_INFLECTION_SUFFIXES = {"", "а", "у", "е", "и", "ой", "ых", "ым", "ыми",
                         "ом", "ам", "ами", "ах", "ей", "ю", "я", "ь"}


def _cardinal_from_root(word):
    """Пытается определить числительное по корню.

    Сортировка по убыванию длины корня, чтобы "пятидесят" (7)
    матчился раньше, чем "десят" (5).
    """
    w = word.lower()
    for root, val, mag, is_mult in sorted(_NUMERIC_ROOTS, key=lambda x: -len(x[0])):
        if root not in w:
            continue
        idx = w.index(root)
        suffix = w[idx + len(root):]
        if suffix in _INFLECTION_SUFFIXES:
            return (val, mag, is_mult)
    return None


def expand_dictionaries(texts):
    """Сканирует тексты и добавляет неизвестные числительные в словарь.

    Вызвать один раз при запуске на всех доступных данных.
    """
    seen = set()
    added = []
    for text in texts:
        for w in text.split():
            wc = w.strip(".,!?;:()[]{}«»\"").lower()
            if len(wc) <= 3 or wc in seen or wc in NUMERAL_DICT or wc in ASR_ERRORS:
                continue
            seen.add(wc)
            result = _cardinal_from_root(wc)
            if result:
                NUMERAL_DICT[wc] = (result[0], result[1])
                added.append(wc)
    return added


def lookup_word(word):
    """Ищет слово в словаре числительных.

    Возвращает (значение, порядок_величины, is_multiplier) или None.

    Алгоритм поиска:
     1. Прямой поиск в NUMERAL_DICT (O(1) по хешу)
     2. Если не найдено — поиск в ASR_ERRORS (O(1))
        и повторный поиск канонической формы
     3. Если не найдено — root-based детекция через известные корни

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
    return _cardinal_from_root(w)


_ORDINAL_FUZZY_CACHE = {}


def _levenshtein(s1, s2):
    """Расстояние Левенштейна между двумя строками."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[len(s2)]


def _fuzzy_ordinal_match(word):
    """Нечёткий поиск слова в ORDINAL_SET.

    Возвращает (каноническая_форма, значение) или None.
    Используется для слов с опечатками, ё/е, пропущенными буквами,
    отсутствующими падежными формами.

    Алгоритм:
     1. Levenshtein distance против всех ключей ORDINAL_SET
     2. Отсев по длине (разница ≤ 3)
     3. Префиксный фильтр (первые 5 символов не более 1 ошибки)
     4. Порог расстояния: max(1, len(w) // 6 + 1)
    """
    w = word.lower()
    # Минимальная длина 9 — короткие слова дают ложные срабатывания
    # ("потому"→"пятому", "период"→"первое")
    if len(w) <= 8:
        return None
    if w in _ORDINAL_FUZZY_CACHE:
        return _ORDINAL_FUZZY_CACHE[w]

    best_key = None
    best_dist = 99
    for key in ORDINAL_SET:
        if abs(len(key) - len(w)) > 3:
            continue

        # Префиксный фильтр: первые 5 символов — не более 1 отличия
        prefix_n = min(5, len(w), len(key))
        prefix_diffs = sum(1 for a, b in zip(w[:prefix_n], key[:prefix_n]) if a != b)
        if prefix_diffs > 1:
            continue

        dist = _levenshtein(w, key)
        if dist < best_dist:
            best_dist = dist
            best_key = key

    threshold = max(1, len(w) // 6 + 1)
    if best_key is not None and best_dist <= threshold:
        result = (best_key, ORDINALS[best_key])
    else:
        result = None
    _ORDINAL_FUZZY_CACHE[w] = result
    return result


def is_ordinal_word(word):
    """Проверяет, является ли слово порядковым числительным.

    Сначала точное совпадение по ORDINAL_SET (O(1)),
    затем нечёткое через Levenshtein.

    Если слово уже найдено в NUMERAL_DICT (кардинальное) или ASR_ERRORS,
    fuzzy-матчинг не применяется — это предотвращает ложное определение
    кардинальных чисел как порядковых ("девяносто"→"девяностого").
    """
    w = word.lower()
    if w in ORDINAL_SET:
        return True
    if w in NUMERAL_DICT or w in ASR_ERRORS:
        return False
    return _fuzzy_ordinal_match(w) is not None


def ordinal_value(word):
    """Возвращает цифровое значение порядкового числительного.

    Примеры:
      ordinal_value("первого") -> "1"
      ordinal_value("двадцать") -> "20"
    """
    w = word.lower()
    if w in ORDINALS:
        return ORDINALS[w]
    if w in NUMERAL_DICT or w in ASR_ERRORS:
        return None
    match = _fuzzy_ordinal_match(w)
    if match is not None:
        return match[1]
    return None
