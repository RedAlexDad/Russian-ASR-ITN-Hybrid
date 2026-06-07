# Sequence-based Token Classifier + Parser

План декомпозиции: замена текущего ad-hoc парсера на систематический token classifier + sequence parser.

---

## Текущие проблемы

1. **`parser.py` — единая функция**, mag-based, нет явного state machine → сложно добавлять новые правила
2. **`lexicon.py` — разрозненные fallback'и**: Levenshtein (FP на коротких словах), root-based (только кардиналы), ordinals отдельно
3. **`normalizer.py` — fused compounds хардкодом**: `_FUSED_COMPOUNDS` словарь, `_asr_preprocess()` regex-заплатки
4. **Ordinals и cardinals в одной группе** — нет унифицированного типа токена
5. **Parser accuracy stuck at 99.8%** — оставшаяся ошибка `дваста→200` (GT содержит ASR-ошибку, parser прав для ITN). Новый подход может решить пограничные случаи за счёт лучшей sequence logic

---

## Phase 1: `src/token_classifier.py` (NEW)

Единый классификатор вместо `lookup_word` + `is_ordinal_word` + `ordinal_value` + `_cardinal_from_root` + `expand_dictionaries` + `_fuzzy_ordinal_match`.

### Data class

```python
@dataclass
class TokenClass:
    value: int          # 5, 10, 100, 1000...
    mag: int            # 0=unit, 1=teen, 2=ten, 3=hundred, 4=thousand, 5=million, 6=billion
    subtype: str        # 'cardinal', 'ordinal', 'multiplier', 'fused', 'vague'
    raw: str            # исходное слово
    confidence: float   # 0.0–1.0
```

### Root patterns (`_ROOT_PATTERNS`)

Сортировка по убыванию длины корня (длинные корни матчатся раньше коротких, чтобы избежать "десят" ⊂ "пятидесят"). Каждый корень — regex + action:

| # | Regex | val | mag | subtype | Примеры |
|---|-------|-----|-----|---------|--------|
| 1 | `миллиард` | 1e9 | 6 | mult | миллиарда, миллиардов, миллиардам |
| 2 | `миллион` | 1e6 | 5 | mult | миллиона, миллионам, миллионах |
| 3 | `тысяч` | 1000 | 4 | mult | тысячи, тысячам, тысячами |
| 4 | `тыщ` | 1000 | 4 | mult/special | тыща, тыщи, тыщ |
| 5 | `пятидесят|пятьдесят` | 50 | 2 | card | пятидесяти, пятьдесят |
| 6 | `шестидесят|шестьдесят` | 60 | 2 | card | шестидесяти, шестьдесят |
| 7 | `семидесят|семьдесят` | 70 | 2 | card | семидесяти, семьдесят |
| 8 | `восьмидесят|восемьдесят` | 80 | 2 | card | восьмидесяти, восемьдесят |
| 9 | `двадцат` | 20 | 2 | card | двадцать, двадцати |
| 10 | `тридцат` | 30 | 2 | card | тридцать, тридцати |
| 11 | `девяност` | 90 | 2 | card | девяносто, девяноста |
| 12 | `десят` | 10 | 1 | card | десять, десяти, десятью |
| 13 | `сорок` | 40 | 2 | card | сорок, сорока |
| 14 | `девятисот|девятьсот` | 900 | 3 | card | девятисот, девятьсот |
| 15 | `восьмисот|восемьсот` | 800 | 3 | card | восьмисот, восемьсот |
| 16 | `семисот|семьсот` | 700 | 3 | card | семисот, семьсот |
| 17 | `шестисот|шестьсот` | 600 | 3 | card | шестисот, шестьсот |
| 18 | `пятисот|пятьсот` | 500 | 3 | card | пятисот, пятьсот |
| 19 | `четырехсот|четырёхсот` | 400 | 3 | card | четырехсот, четырёхсот |
| 20 | `трехсот|трёхсот` | 300 | 3 | card | трехсот, трёхсот |
| 21 | `двухсот|двумстам|двумястам|двухстах` | 200 | 3 | card | двухсот, двумстам |
| 22 | `двест|двес[ия]` | 200 | 3 | card | двести, двеси, дваста |
| 23 | `сот|ста\b|сто\b` | 100 | 3 | card | сто, ста, сот |
| 24 | `четыр` | 4 | 0 | card | четыре, четырёх, четырех |
| 25 | `тро[её]|тр[её]х|трем|тремя` | 3 | 0 | card | трое, трёх, трем |
| 26 | `дв[оеуя]` | 2 | 0 | card | двое, двух, двумя |
| 27 | `од[инн]|одн[аео]|единиц` | 1 | 0 | card | один, одна, одно |
| 28 | `пят[еёьию]` | 5 | 0 | card | пятеро, пять, пяти |
| 29 | `шест[еёьию]` | 6 | 0 | card | шестеро, шесть, шести |
| 30 | `сем[еёьию]` | 7 | 0 | card | семеро, семь, семи |
| 31 | `вос[ье]м[еёию]|восем` | 8 | 0 | card | восемь, восьми, восемью |
| 32 | `девят[еёию]` | 9 | 0 | card | девятеро, девять, девяти |
| 33 | `нол[ьяю]|нул[ьяю]` | 0 | 0 | card | ноль, нуль |
| 34 | `одиннадцат` | 11 | 1 | card | одиннадцать, одиннадцати |
| 35 | `двенадцат` | 12 | 1 | card | двенадцать, двенадцати |
| 36 | `тринадцат` | 13 | 1 | card | тринадцать, тринадцати |
| 37 | `четырнадцат` | 14 | 1 | card | четырнадцать, четырнадцати |
| 38 | `пятнадцат` | 15 | 1 | card | пятнадцать, пятнадцати |
| 39 | `шестнадцат` | 16 | 1 | card | шестнадцать, шестнадцати |
| 40 | `семнадцат` | 17 | 1 | card | семнадцать, семнадцати |
| 41 | `восемнадцат` | 18 | 1 | card | восемнадцать, восемнадцати |
| 42 | `девятнадцат` | 19 | 1 | card | девятнадцать, девятнадцати |

### Ordinal detection

Если слово matched как cardinal — проверяем суффикс после корня:

```python
_ORDINAL_SUFFIXES = {
    "ый", "ий", "ой", "ая", "ое", "ые",
    "ых", "ым", "ыми", "ого", "ому", "ом",
    "ую", "ей", "его", "ему", "ем",
    "ья", "ье", "ьи",
}

_ORDINAL_EXCEPTIONS = {
    "какой", "такой", "другой", "любой", "каждой",
    "самой", "самый", "самое", "новая", "новый",
    "простой", "главный", "большой", "хороший", "плохой",
    "маленький", "маленькая", "высокий", "низкий", "нужный",
    "последний", "ближайший", "разный", "целый", "полный",
    "важный", "точный", "активный", "интересный",
    "обычный", "подобный", "отдельный", "значительный",
    "собственный", "человеческий", "прежний",
    "дополнительный", "практический", "технический",
    "экономический", "политический", "исторический",
    "физический", "юридический", "медицинский",
    "социальный", "культурный", "научный",
    "международный", "современный",
}
```

Алгоритм:
1. Если слово в `_ORDINAL_EXCEPTIONS` → не ordinal
2. Если слово заканчивается на суффикс из `_ORDINAL_SUFFIXES` → subtype='ordinal'
3. Значение ordinal = значение cardinal корня

### Fused compound detection

Если слово не нашлось как один токен — пробуем split по корням:

```
"дветысячи" → match "две"(val=2,mag=0) + остаток "тысячи" → match "тысячи"(1000,mag=4,mult)
Результат: [TokenClass(2,0,'fused'), TokenClass(1000,4,'fused')]
```

Это заменяет `_FUSED_COMPOUNDS` словарь и `_asr_preprocess()` правило "двеси триста пятьдесят тысяч".

### Vague "тыщ" heuristic

Перенести `_is_vague_tyt_context()` в classifier:

```python
_VAGUE_MARKERS = {
    "выше", "ниже", "около", "почти",
    "половиной",     # "с половиной тыщ"
}

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
```

Если True → `TokenClass(1000,4,'vague')` — sequence parser проигнорирует.

### Main API

```python
def classify(word: str, prev_tokens: list[str] | None = None) -> list[TokenClass] | None:
    """
    Пайплайн:
    1. Exact dict match (NUMERAL_DICT + ASR_ERRORS) — быстрый путь O(1)
    2. Root-based regex match
    3. Ordinal suffix check
    4. Fused compound split
    5. Vague context check (для тыщ/тыща)
    Возвращает список из 0..N токенов (для fused — >1).
    None если не числовое слово.
    """
```

```python
def classify_tokens(tokens: list[str]) -> list[list[TokenClass] | None]:
    """Классифицировать все токены строки."""
    result = []
    for i, token in enumerate(tokens):
        prev = tokens[:i]
        result.append(classify(token, prev))
    return result
```

---

## Phase 2: `src/sequence_parser.py` (NEW)

State-machine над `list[TokenClass]` вместо `parse_number_group()`.

### State diagram

```
START ──→ ACCUM ──→ MULTIPLY ──→ ACCUM
  │          │          │          │
  └──→ ENUM ──→ FLUSH ──→ START   │
       └──→ ORDINAL ──→ FLUSH ────┘
```

### Scan rules (left→right)

```
Token[i]    vs Token[i-1]     → Action
───────     ─────────────     ──────
subtype=vague                   → skip token (оставить как есть)
subtype=multiplier              → current ×= val, record mult_mag
subtype=ordinal                 → finalize current with ordinal suffix
mag < prev_mag                  → current += val (sum mode)
mag == prev_mag (оба < 3)      → flush group, start new (enum mode)
mag >= prev_mag (оба == 3)     → flush group, start new (enum: "двести триста")
same mult_mag as last           → flush, start new ("миллионов…миллиона" → 7e7 2e6)
```

### Construction logic

```python
def parse_sequence(classes: list[TokenClass]) -> list[str]:
    compound = 0    # накоплено после умножителей
    current = 0     # текущее число (< 1000)
    last_mag = -1
    last_mult_mag = -1
    is_ordinal = False
    result = []

    for i, tc in enumerate(classes):
        # ── skip vague ──
        if tc.subtype == 'vague':
            continue

        # ── multiplier ──
        if tc.subtype == 'multiplier':
            if last_mult_mag == tc.mag:
                # два mult одного ранга → enum
                flush(compound, current, is_ordinal, result)
                compound = 0; current = 0
            compound += (current or 1) * tc.value
            current = 0
            last_mult_mag = tc.mag
            last_mag = -1
            continue

        # ── ordinal ──
        if tc.subtype == 'ordinal':
            if is_ordinal and last_mag == tc.mag:
                # "первый второй" → enum
                flush(compound, current, True, result)
                current = tc.value
            elif last_mag == -1 or tc.mag < last_mag:
                current = (current or 0) + tc.value
            else:
                flush(compound, current, is_ordinal, result)
                current = tc.value
            is_ordinal = True
            last_mag = tc.mag
            continue

        # ── cardinal ──
        if last_mag == -1:
            current = tc.value
        elif tc.mag < last_mag:
            current += tc.value
        else:
            # mag не убывает → enum
            flush(compound, current, is_ordinal, result)
            compound = 0; current = tc.value
            is_ordinal = False
        last_mag = tc.mag

    # flush остаток
    flush(compound, current, is_ordinal, result)
    return result if result else (["0"] if any zero tokens else [])


def flush(compound, current, is_ordinal, result):
    total = compound + current
    if total > 0:
        result.append(str(total))
```

### Comparison with current `parse_number_group()`

| Аспект | Current | New |
|--------|---------|-----|
| Вход | `list[(val, mag, is_mult, is_ordinal)]` | `list[TokenClass]` |
| Fused compounds | Хардкод в normalizer | Авто в classifier |
| Vague | Pre-check в normalizer | subtype='vague' |
| State machine | Неявный (if-elif) | Явная state machine |
| Ordinals | Флаг в кортеже | subtype='ordinal' |
| Extensibility | Менять parse_number_group | Добавлять правила в state machine |

---

## Phase 3: `src/lexicon.py` — adapt

### Изменения

```diff
- def _cardinal_from_root(word):
- def expand_dictionaries(texts):
- def _levenshtein(s1, s2):
- def _fuzzy_ordinal_match(word):
- _NUMERIC_ROOTS
- _INFLECTION_SUFFIXES
- _ORDINAL_FUZZY_CACHE
+ from src.token_classifier import classify

def lookup_word(word):
    """Adapter: вызывает classify(), возвращает (val, mag, is_mult) или None."""
+   result = classify(word)
+   if result:
+       tc = result[0]  # берём первый токен
+       return (tc.value, tc.mag, tc.subtype == 'multiplier')
    return None

def is_ordinal_word(word):
+   result = classify(word)
+   if result:
+       return result[0].subtype == 'ordinal'
    return False

def ordinal_value(word):
+   result = classify(word)
+   if result and result[0].subtype == 'ordinal':
+       return str(result[0].value)
    return None
```

`NUMERAL_DICT`, `ASR_ERRORS`, `ORDINAL_SET`, `ORDINALS` остаются как источник для exact match в classify().

---

## Phase 4: `src/normalizer.py` — new path

### normalize_text_sequence()

```python
def normalize_text_sequence(text):
    """Sequence-based нормализация: token classifier + sequence parser."""
    if not isinstance(text, str) or not text.strip():
        return text

    text = _asr_preprocess(text)  # только мягкий знак и тристо→триста (остальное убрать)
    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        classes = classify(token, tokens[:i])

        if classes and not all(c.subtype == 'vague' for c in classes):
            # Собираем группу числовых токенов
            group = []
            while i < len(tokens):
                cs = classify(tokens[i], tokens[:i])
                if cs and not all(c.subtype == 'vague' for c in cs):
                    group.extend(cs)
                    i += 1
                else:
                    break
            parsed = parse_sequence(group)
            result.extend(parsed)
        else:
            result.append(token)
            i += 1

    return " ".join(result)
```

### Changes to `_asr_preprocess()`

```diff
_ASR_SUBSTITUTIONS = [
-   # "двеси" + hundred + (ten) + тысяч → remove (classifier handles)
-   (re.compile(r'\bдвеси\s+...'), ...)
    # Мягкий знак — оставить
    (re.compile(r'\bпятдесят\b'), 'пятьдесят'),
    (re.compile(r'\bшестдесят\b'), 'шестьдесят'),
    (re.compile(r'\bсемдесят\b'), 'семьдесят'),
    (re.compile(r'\bвосемдесят\b'), 'восемьдесят'),
    # Падежные ошибки — оставить
    (re.compile(r'\bтристо\b'), 'триста'),
    (re.compile(r'\bчетыриста\b'), 'четыреста'),
]
```

```diff
- _FUSED_COMPOUNDS = { ... }  → удалить
```

### Disambiguate integration

```python
# В classify(), после root match:
from src.disambiguate import is_likely_numeric
if tc.mag == 3 and tc.value == 100:  # "сто"
    if not is_likely_numeric(full_text, word, pos):
        return None  # не числительное
```

---

## Phase 5: `src/cli.py` + Makefile — --parser-type

### cli.py

```diff
+ run_p.add_argument("--parser-type", choices=['current', 'sequence'], default='current')
+ eval_p.add_argument("--parser-type", choices=['current', 'sequence'], default='current')
+ err_p.add_argument("--parser-type", choices=['current', 'sequence'], default='current')
```

```diff
def cmd_evaluate(args):
-   normalize = hybrid_normalize if args.hybrid else normalize_text
+   if args.parser_type == 'sequence':
+       from src.normalizer import normalize_text_sequence
+       normalize = normalize_text_sequence
+   else:
+       normalize = hybrid_normalize if args.hybrid else normalize_text
```

### local.mk

```makefile
evaluate-sequence-local:
	@printf "$(YELLOW)${BOLD}[EVAL-SEQ]$(NC)  Sequence parser accuracy на: $(CALIB)\n"
	$(VENV_PY) main.py evaluate "$(CALIB)" --parser-type sequence

errors-sequence-local:
	@printf "$(RED)${BOLD}[ERR-SEQ]$(NC)    Первые $(or $(N),15) ошибок (sequence) на: $(CALIB)\n"
	$(VENV_PY) main.py errors "$(CALIB)" -n $(or $(N),15) --parser-type sequence
```

### help.mk

Добавить в секцию локального запуска:

```
  @printf "$(YELLOW)  evaluate-seq$(NC)      Sequence parser:      $(YELLOW)make evaluate-sequence-local$(NC)\n"
  @printf "$(RED)  errors-seq$(NC)         Sequence parser errors: $(YELLOW)make errors-sequence-local N=10$(NC)\n"
```

---

## Phase 6: `src/hybrid.py` — update

### _parser_confidence()

```diff
def _parser_confidence(text, pred):
    score = 1.0
    tokens = text.lower().split()

-   # 2. "двеси" + compound — classifier сам разберёт
-   if "двеси" in tokens:
-       ...

-   # 5. Слово с префиксом числа + "тысяч" — classifier сам split'ит
-   for w in tokens:
-       if "тысяч" in w:
-           for pfx in _number_prefixes:
-               ...
- 
    return max(0.1, min(1.0, score))
```

### hybrid_normalize()

```python
def hybrid_normalize(text, parser_type='current'):
    pred_rule = normalize_text_sequence(text) if parser_type == 'sequence' else normalize_text(text)
    confidence = _parser_confidence(text, pred_rule)
    if confidence < 0.75 and os.environ.get("HYBRID_USE_T5") == "1":
        if _load_model():
            pred_t5 = _t5_generate(text)
            if pred_t5 and pred_t5 != text:
                return pred_t5
    return pred_rule
```

---

## Phase 7: Tests

### `tests/test_sequence_parser.py` (NEW)

Все 9 тестов из `test_parser.py` (должны проходить без изменений) + новые:

```python
# Current tests (adapt to new API)
def test_sum_hundreds_tens():
    assert parse_sequence([TokenClass(200,3,'cardinal'), TokenClass(50,2,'cardinal')]) == ['250']

def test_enumeration_same_magnitude():
    assert parse_sequence([TokenClass(200,3,'cardinal'), TokenClass(300,3,'cardinal')]) == ['200', '300']

def test_sum_with_thousand_multiplier():
    assert parse_sequence([
        TokenClass(2,0,'cardinal'), TokenClass(1000,4,'multiplier'),
        TokenClass(800,3,'cardinal'), TokenClass(40,2,'cardinal'), TokenClass(3,0,'cardinal')
    ]) == ['2843']

def test_two_multiplier_blocks():
    assert parse_sequence([
        TokenClass(70,2,'cardinal'), TokenClass(1000000,5,'multiplier'),
        TokenClass(2,0,'cardinal'), TokenClass(1000000,5,'multiplier')
    ]) == ['70000000', '2000000']

def test_standalone_thousand():
    assert parse_sequence([TokenClass(1000,4,'multiplier')]) == ['1000']

def test_zero():
    assert parse_sequence([TokenClass(0,0,'cardinal')]) == ['0']

def test_zero_and_seven():
    assert parse_sequence([TokenClass(0,0,'cardinal'), TokenClass(7,0,'cardinal')]) == ['0', '7']

def test_ordinal_in_compound():
    assert parse_sequence([
        TokenClass(200,3,'cardinal'), TokenClass(80,1,'cardinal'), TokenClass(5,0,'ordinal')
    ]) == ['285']

def test_empty():
    assert parse_sequence([]) == []

# ── New tests ──

def test_fused_compound():
    # "дветысячи" → classify split'ит на два токена
    assert parse_sequence([
        TokenClass(2,0,'fused'), TokenClass(1000,4,'fused')
    ]) == ['2000']

def test_asr_dvesi_hundred_tycyach():
    # "двеси триста пятьдесят тысяч" → 235000
    assert parse_sequence([
        TokenClass(200,3,'cardinal'), TokenClass(300,3,'cardinal'),
        TokenClass(50,2,'cardinal'), TokenClass(1000,4,'multiplier')
    ]) == ['235000']
    # важно: "двеси"(200) + "триста"(300) не должны enum'иться — группируем в 235000

def test_vague_tyt_skipped():
    assert parse_sequence([TokenClass(1000,4,'vague')]) == []

def test_standalone_ordinal():
    assert parse_sequence([TokenClass(20,2,'ordinal')]) == ['20']

def test_multiple_ordinals():
    assert parse_sequence([
        TokenClass(1,0,'ordinal'), TokenClass(2,0,'ordinal'), TokenClass(3,0,'ordinal')
    ]) == ['1', '2', '3']

def test_ordinal_with_cardinal_prefix():
    assert parse_sequence([
        TokenClass(200,3,'cardinal'), TokenClass(80,1,'cardinal'), TokenClass(4,0,'ordinal')
    ]) == ['284']

def test_large_number():
    assert parse_sequence([
        TokenClass(5,0,'cardinal'), TokenClass(1000,4,'multiplier'),
        TokenClass(200,3,'cardinal'),
        TokenClass(30,2,'cardinal'),
        TokenClass(5,0,'cardinal')
    ]) == ['5235']
```

### `tests/test_classifier.py` (NEW)

```python
def test_classify_unit():
    tc = classify("пять")
    assert tc and tc[0].value == 5 and tc[0].mag == 0

def test_classify_teen():
    tc = classify("четырнадцать")
    assert tc and tc[0].value == 14 and tc[0].mag == 1

def test_classify_ten():
    tc = classify("восемьдесят")
    assert tc and tc[0].value == 80

def test_classify_hundred():
    tc = classify("триста")
    assert tc and tc[0].value == 300

def test_classify_ordinal():
    tc = classify("двадцатое")
    assert tc and tc[0].value == 20 and tc[0].subtype == 'ordinal'

def test_classify_fused():
    tc = classify("дветысячи")
    assert tc and len(tc) == 2
    assert tc[0].value == 2 and tc[1].value == 1000

def test_classify_non_numeral():
    assert classify("компьютер") is None
    assert classify("программа") is None

def test_vague_context():
    tc = classify("тыщ", prev_tokens=["с", "чем", "то"])
    assert tc and tc[0].subtype == 'vague'

def test_classify_asr_error():
    tc = classify("двеси")
    assert tc and tc[0].value == 200
```

---

## Timeline

| Phase | File | Что делаем | Изменения | Эстимейт |
|-------|------|-----------|-----------|----------|
| 1 | `src/token_classifier.py` | NEW: TokenClass, root patterns, classify(), classify_tokens(), vague heuristic, fused split | ~200 строк | 2-3 ч |
| 2 | `src/sequence_parser.py` | NEW: parse_sequence() state machine | ~130 строк | 1-2 ч |
| 3 | `src/lexicon.py` | Adapt: lookup_word→classify, удалить dead code | ~-110 строк | 30 мин |
| 4 | `src/normalizer.py` | Добавить normalize_text_sequence(), упростить _asr_preprocess(), удалить FUSED | ~30 строк +- | 30 мин |
| 5 | `src/cli.py` + local.mk | argparse --parser-type, evaluate-sequence-local, errors-sequence-local | ~20 строк | 15 мин |
| 6 | `src/hybrid.py` | Упростить _parser_confidence() | ~-20 строк | 15 мин |
| 7 | `tests/test_sequence_parser.py` + `test_classifier.py` | NEW: все тесты | ~150 строк | 30 мин |
| | **Итого** | | | **~6 часов** |

---

## Migration strategy

1. Новый код живёт параллельно: `normalize_text_sequence()` → `parse_sequence()` → `classify()`
2. Старый код untouched: `normalize_text()` → `parse_number_group()` → `lookup_word()`
3. `--parser-type sequence` флаг выбирает путь
4. После валидации на всех датасетах (calibration.f + synthetic.f + real.f) и достижения ≥99.8%:
   - `normalize_text()` переключается на sequence parser
   - Старый `parse_number_group()` удаляется
   - `from src.lexicon import lookup_word, is_ordinal_word, ordinal_value` реэкспортируют из classifier

---

## Эвристика для «тыщ» — детали

Текущая `_is_vague_tyt_context()` проверяет триггеры за 3-4 токена назад:

```
"с чем то тыщ"     → tokens[i-3]=с, tokens[i-2]=чем, tokens[i-1]=то
"выше тыщ"         → tokens[i-1]=выше
"с половиной тыщ"  → tokens[i-2]=с, tokens[i-1]=половиной
"где то тыщ"       → tokens[i-2]=где, tokens[i-1]=то
```

В новом классификаторе:

1. Если слово `тыщ`/`тыща` → вызываем `_is_vague_context(prev_tokens)`
2. Если True → возвращаем `TokenClass(1000,4,'vague')`
3. Sequence parser видит subtype='vague' → пропускает токен (не заменяет)

Расширение: можно добавить `около тыщ`, `почти тыщ`, `где-то тыщ`. А также: если перед `тыщ` НЕТ числа (одиночное "тыщ" без digit→hundred→thousand цепочки) → vague.
