#!/usr/bin/env python3
"""
Генератор синтетического датасета для ITN.

Источники:
  1. Новостные RSS-ленты (ТАСС, РИА, Интерфакс)
  2. Wikipedia API — случайные статьи с числами
  3. Шаблонные фразы из calibration.f
  4. Контролируемая генерация суммы/перечисления
  5. Порядковые числительные

Процесс:
  Парсинг -> извлечение предложений с числами -> число->слова -> ASR-шум -> .feather
"""

import json
import os
import random
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dicts.asr_errors import ASR_ERRORS

random.seed(42)

# ──────────────────────────────────────────────
# 1. Контекстные шаблоны (извлечены из calibration.f)
# ──────────────────────────────────────────────

CONTEXT_TEMPLATES = [
    "бюджет в размере {num} рублей",
    "примерно {num} показов",
    "около {num} процентов",
    "на {num} дней",
    "суммарно от {num1} до {num2}",
    "в районе {num} тысяч",
    "порядка {num} рублей",
    "ставка {num} процентов",
    "скидка {num} процентов",
    "лимит {num} рублей",
    "в среднем {num}",
    "где то {num}",
    "минимальный {num}",
    "около {num}",
    "до {num}",
    "от {num}",
    "по {num}",
    "за {num}",
    "более {num}",
    "меньше {num}",
    "свыше {num}",
    "всего {num}",
    "уже {num}",
    "почти {num}",
    "на {num}",
    "в размере {num}",
    "объемом {num}",
    "в количестве {num}",
]

FILLERS = [
    "ну",
    "вот",
    "типа",
    "как бы",
    "то есть",
    "да",
    "в общем",
    "короче",
    "понимаете",
    "смотрите",
    "скажем",
    "допустим",
    "где то",
    "примеро",
    "в принципе",
    "вообще",
]

# Хвосты для тысяч/миллионов в зависимости от числа
THOUSAND_TAIL = {
    1: "тысяча",
    2: "тысячи",
    3: "тысячи",
    4: "тысячи",
    5: "тысяч",
    6: "тысяч",
    7: "тысяч",
    8: "тысяч",
    9: "тысяч",
    0: "тысяч",
}
MILLION_TAIL = {
    1: "миллион",
    2: "миллиона",
    3: "миллиона",
    4: "миллиона",
    5: "миллионов",
    6: "миллионов",
    7: "миллионов",
    8: "миллионов",
    9: "миллионов",
    0: "миллионов",
}

# ──────────────────────────────────────────────
# 2. Число -> словесная форма
# ──────────────────────────────────────────────

ONES = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
ONES_FEM = [
    "",
    "одна",
    "две",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
]
TEENS_WORDS = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
]
TENS_WORDS = [
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
]
HUNDREDS_WORDS = [
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
]


def _hundreds(n, fem=False):
    """Преобразует число 0-999 в слова."""
    if n == 0:
        return []
    parts = []
    parts.append(HUNDREDS_WORDS[n // 100])
    n %= 100
    if 10 <= n <= 19:
        parts.append(TEENS_WORDS[n - 10])
        return parts
    if n >= 20:
        parts.append(TENS_WORDS[n // 10])
        n %= 10
    if n > 0:
        parts.append(ONES_FEM[n] if fem else ONES[n])
    return [p for p in parts if p]


def number_to_words(n):
    """Преобразует число 0-999999999 в словесную форму."""
    if n == 0:
        return "ноль"
    parts = []
    if n >= 1000000:
        m = n // 1000000
        parts.extend(_hundreds(m))
        parts.append(MILLION_TAIL.get(m, "миллионов"))
        n %= 1000000
    if n >= 1000:
        t = n // 1000
        parts.extend(_hundreds(t, fem=True))
        parts.append(THOUSAND_TAIL.get(t, "тысяч"))
        n %= 1000
    if n > 0:
        parts.extend(_hundreds(n))
    return " ".join(parts)


def number_to_ordinal_words(n):
    """Преобразует число 1-31 в порядковое (именительный падеж)."""
    ordinals_map = {
        1: "первый",
        2: "второй",
        3: "третий",
        4: "четвертый",
        5: "пятый",
        6: "шестой",
        7: "седьмой",
        8: "восьмой",
        9: "девятый",
        10: "десятый",
        11: "одиннадцатый",
        12: "двенадцатый",
        13: "тринадцатый",
        14: "четырнадцатый",
        15: "пятнадцатый",
        16: "шестнадцатый",
        17: "семнадцатый",
        18: "восемнадцатый",
        19: "девятнадцатый",
        20: "двадцатый",
        21: "двадцать первый",
        22: "двадцать второй",
        23: "двадцать третий",
        24: "двадцать четвертый",
        25: "двадцать пятый",
        26: "двадцать шестой",
        27: "двадцать седьмой",
        28: "двадцать восьмой",
        29: "двадцать девятый",
        30: "тридцатый",
        31: "тридцать первый",
    }
    return ordinals_map.get(n, str(n))


# ──────────────────────────────────────────────
# 3. ASR-шум
# ──────────────────────────────────────────────


def _is_numeral_word(w):
    """Проверяет, является ли слово числительным (из словаря)."""
    from src.lexicon import lookup_word
    return lookup_word(w) is not None


# Список слов-числительных для быстрой проверки слитных написаний
_NUMERAL_TOKENS = {
    'один','одна','одно','два','две','три','четыре','пять','шесть','семь',
    'восемь','девять','десять','одиннадцать','двенадцать','тринадцать',
    'четырнадцать','пятнадцать','шестнадцать','семнадцать','восемнадцать',
    'девятнадцать','двадцать','тридцать','сорок','пятьдесят','шестьдесят',
    'семьдесят','восемьдесят','девяносто','сто','двести','триста','четыреста',
    'пятьсот','шестьсот','семьсот','восемьсот','девятьсот',
    'тысяча','тысячи','тысяч','тысячу','миллион','миллиона','миллионов',
    'миллиард','миллиарда','миллиардов',
}


def apply_asr_noise(text, noise_level=0.6):
    """Применяет ASR-искажения к тексту."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]

        # Замена на известное ASR-искажение
        if random.random() < noise_level and w in ASR_ERRORS:
            variants = [k for k, v in ASR_ERRORS.items() if v == w]
            if variants:
                result.append(random.choice(variants))
                i += 1
                continue

        # Слитное написание числительных (две тысячи -> дветысячи)
        # Если два слова подряд — числительные, склеиваем их
        if (i + 1 < len(words)
                and w in _NUMERAL_TOKENS
                and words[i + 1] in _NUMERAL_TOKENS
                and random.random() < 0.15):
            result.append(w + words[i + 1])
            i += 2
            continue

        # Склейка с соседом (любые слова)
        if (
            i + 1 < len(words)
            and random.random() < 0.15
            and len(w) > 2
            and len(words[i + 1]) > 2
        ):
            result.append(w + words[i + 1])
            i += 2
            continue

        # Замена букв (фонетическая)
        if random.random() < 0.10 and len(w) > 3:
            subs = {
                "о": "а",
                "е": "и",
                "и": "е",
                "я": "а",
                "т": "д",
                "д": "т",
                "б": "п",
                "п": "б",
                "с": "з",
                "з": "с",
            }
            w_list = list(w)
            for j in range(len(w_list)):
                if w_list[j] in subs and random.random() < 0.3:
                    w_list[j] = subs[w_list[j]]
            w = "".join(w_list)

        # Пропуск буквы
        if random.random() < 0.08 and len(w) > 4:
            pos = random.randint(1, len(w) - 2)
            w = w[:pos] + w[pos + 1 :]

        # Levenshtein-вариация для неизвестных слов
        # Для слов, которые не числительные и не в ASR_ERRORS,
        # применяем случайную редакционную операцию (вставка, удаление, замена, перестановка)
        if random.random() < 0.12 and len(w) > 3 and not any(c.isdigit() for c in w):
            op = random.choice(['sub', 'del', 'ins', 'trans'])
            pos = random.randint(1, len(w) - 2)
            if op == 'sub' and len(w) > 3:
                # Замена буквы на похожую
                nearby = {'а': 'оя', 'о': 'ае', 'е': 'иё', 'и': 'еы',
                          'у': 'ю', 'ю': 'у', 'я': 'а', 'ы': 'и',
                          'т': 'дс', 'д': 'т', 'п': 'б', 'б': 'п',
                          'с': 'зт', 'з': 'с', 'к': 'г', 'г': 'к',
                          'в': 'ф', 'ф': 'в', 'р': 'л', 'л': 'р'}
                if w[pos] in nearby:
                    w = w[:pos] + random.choice(nearby[w[pos]]) + w[pos+1:]
            elif op == 'del' and len(w) > 5:
                w = w[:pos] + w[pos+1:]
            elif op == 'ins' and len(w) < 12:
                extra = random.choice('аеиоуыя')
                w = w[:pos] + extra + w[pos:]
            elif op == 'trans' and pos + 1 < len(w):
                w = w[:pos] + w[pos+1] + w[pos] + w[pos+2:]

        result.append(w)
        i += 1
    return " ".join(result)


# ──────────────────────────────────────────────
# 4. Генерация из шаблонов
# ──────────────────────────────────────────────


def generate_from_templates(count=5000):
    """Генерирует пары из контекстных шаблонов."""
    pairs = []
    for _ in range(count):
        tpl = random.choice(CONTEXT_TEMPLATES)
        if "{num1}" in tpl:
            n1 = random.choices(
                [200, 300, 500, 1000, 2000, 5000, 10000, 50000],
                weights=[20, 20, 15, 15, 10, 10, 5, 5],
                k=1,
            )[0]
            n2 = n1 + random.choice([500, 1000, 5000, 10000])
            clean = tpl.format(num1=number_to_words(n1), num2=number_to_words(n2))
            answer = tpl.format(num1=str(n1), num2=str(n2))
        else:
            n = random.choices(
                [
                    random.randint(1, 9),
                    random.randint(10, 99),
                    random.randint(100, 999),
                    random.randint(1000, 9999),
                    random.randint(10000, 99999),
                    random.randint(100000, 999999),
                ],
                weights=[24, 33, 34, 5, 3, 1],
                k=1,
            )[0]
            clean = tpl.format(num=number_to_words(n))
            answer = tpl.format(num=str(n))

        if random.random() < 0.4:
            filler = random.choice(FILLERS)
            pos = random.randint(0, len(clean.split()))
            cl = clean.split()
            cl.insert(pos, filler)
            an = answer.split()
            an.insert(min(pos, len(an)), filler)
            clean, answer = " ".join(cl), " ".join(an)

        noisy = apply_asr_noise(clean)
        pairs.append(
            {
                "task_text": noisy,
                "ground_truth": answer,
                "source": "template",
                "num_type": "cardinal",
                "noise_level": "noisy" if noisy != clean else "clean",
            }
        )
    return pairs


# ──────────────────────────────────────────────
# 5. Группировка (сумма vs перечисление)
# ──────────────────────────────────────────────


def generate_grouping_examples(count=3000):
    """Генерирует примеры суммы и перечисления."""
    pairs = []

    sum_cases = [
        (
            lambda: (random.randint(1, 9) * 1000 + random.randint(1, 9) * 100, False),
            lambda v: number_to_words(v),
        ),
        (
            lambda: (random.randint(1, 9) * 100 + random.randint(1, 9) * 10, False),
            lambda v: number_to_words(v),
        ),
        (
            lambda: (random.randint(1, 9) * 10000 + random.randint(1, 9) * 1000, False),
            lambda v: number_to_words(v),
        ),
        (
            lambda: (
                random.randint(1, 9) * 1000
                + random.randint(1, 9) * 100
                + random.randint(1, 9) * 10,
                False,
            ),
            lambda v: number_to_words(v),
        ),
    ]
    enum_cases = [
        lambda: ("200 300", f"{number_to_words(200)} {number_to_words(300)}"),
        lambda: ("300 400", f"{number_to_words(300)} {number_to_words(400)}"),
        lambda: ("20 30", f"{number_to_words(20)} {number_to_words(30)}"),
        lambda: ("50 70", f"{number_to_words(50)} {number_to_words(70)}"),
        lambda: ("5 7", f"{number_to_words(5)} {number_to_words(7)}"),
        lambda: (
            "100 200 300",
            f"{number_to_words(100)} {number_to_words(200)} {number_to_words(300)}",
        ),
    ]

    for _ in range(count):
        is_sum = random.random() < 0.6
        if is_sum:
            gen, to_words = random.choice(sum_cases)
            val, _ = gen()
            answer = str(val)
            clean = to_words(val)
        else:
            answer, clean = random.choice(enum_cases)()

        tpl = random.choice(CONTEXT_TEMPLATES)
        if "{num}" in tpl:
            clean_t = tpl.format(num=clean)
            answer_t = tpl.format(num=answer)
        else:
            clean_t, answer_t = clean, answer

        noisy = apply_asr_noise(clean_t)
        pairs.append(
            {
                "task_text": noisy,
                "ground_truth": answer_t,
                "source": "grouping",
                "num_type": "cardinal",
                "noise_level": "noisy" if noisy != clean_t else "clean",
            }
        )
    return pairs


# ──────────────────────────────────────────────
# 6. Порядковые
# ──────────────────────────────────────────────

ORDINAL_TEMPLATES = [
    "до {num} числа",
    "с {num} числа",
    "после {num} числа",
    "{num} день",
    "до {num} декабря",
    "с {num} января",
    "на {num} месте",
    "{num} год",
    "в {num} ряду",
]


def generate_ordinals(count=1500):
    pairs = []
    for _ in range(count):
        val = random.randint(1, 31)
        word = number_to_ordinal_words(val)
        tpl = random.choice(ORDINAL_TEMPLATES)
        clean = tpl.format(num=word)
        answer = tpl.format(num=str(val))
        noisy = apply_asr_noise(clean)
        pairs.append(
            {
                "task_text": noisy,
                "ground_truth": answer,
                "source": "ordinal",
                "num_type": "ordinal",
                "noise_level": "noisy" if noisy != clean else "clean",
            }
        )
    return pairs


# ──────────────────────────────────────────────
# 7. Парсинг новостей (RSS + Wikipedia)
# ──────────────────────────────────────────────

NEWS_FEEDS = [
    "https://tass.ru/rss/v2.xml",
    "https://ria.ru/export/rss2/archive/index.xml",
    "https://www.interfax.ru/rss.asp",
    "https://russian.rt.com/rss/news",
    "https://lenta.ru/rss/news",
]


def fetch_news_sentences(max_sentences=2000):
    """Парсит RSS и Wikipedia, извлекает предложения с числами."""
    sentences = []
    import xml.etree.ElementTree as ET

    import requests

    # RSS
    for url in NEWS_FEEDS:
        try:
            resp = requests.get(
                url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (ITNResearchBot)"}
            )
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.iter("item"):
                for field in ["title", "description"]:
                    e = item.find(field)
                    if e is not None and e.text:
                        text = re.sub(r"<[^>]+>", "", e.text)
                        for part in re.split(r"[.!?]+", text):
                            if re.search(r"\d+", part) and len(part) > 20:
                                sentences.append(part.strip())
        except Exception as ex:
            print(f"    [WARN] RSS {url}: {ex}")

    # Wikipedia: Random articles
    wiki_api = "https://ru.wikipedia.org/w/api.php"
    try:
        for _ in range(50):
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": 1,
            }
            r = requests.get(wiki_api, params=params, timeout=10).json()
            pages = r.get("query", {}).get("random", [])
            for p in pages:
                title = p.get("title", "")
                ext_params = {
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "exlimit": 1,
                    "exintro": True,
                }
                r2 = requests.get(wiki_api, params=ext_params, timeout=10).json()
                for _, page_data in r2.get("query", {}).get("pages", {}).items():
                    text = page_data.get("extract", "")
                    for part in re.split(r"[.!?]+", text):
                        if re.search(r"\d+", part) and len(part) > 30:
                            sentences.append(part.strip()[:200])
    except Exception as ex:
        print(f"    [WARN] Wikipedia: {ex}")

    # Дедупликация
    seen = set()
    unique = []
    for s in sentences:
        key = s.lower()[:50]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:max_sentences]


def process_news_sentences(sentences, max_pairs=5000):
    """Извлекает числа из предложений и создаёт пары."""
    pairs = []
    for sent in sentences:
        nums = re.findall(r"\b(\d{1,6})\b", sent)
        for num_str in nums[:2]:
            n = int(num_str)
            if n == 0 or n > 999999:
                continue
            word_form = number_to_words(n)
            clean = sent.replace(num_str, word_form, 1)
            answer = sent

            if random.random() < 0.3:
                filler = random.choice(FILLERS)
                cl = clean.split()
                cl.insert(random.randint(0, len(cl)), filler)
                clean = " ".join(cl)
                an = answer.split()
                an.insert(random.randint(0, len(an)), filler)
                answer = " ".join(an)

            noisy = apply_asr_noise(clean, noise_level=0.5)
            pairs.append(
                {
                    "task_text": noisy,
                    "ground_truth": answer,
                    "source": "news",
                    "num_type": "cardinal",
                    "noise_level": "noisy" if noisy != clean else "clean",
                }
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


# ──────────────────────────────────────────────
# 8. Сборка датасета
# ──────────────────────────────────────────────


def generate_dataset(output_path="data/synthetic.f", total_target=30000):
    print("Генерация синтетического датасета...\n")
    all_pairs = []

    print("[1/4] Парсинг новостей и Wikipedia...")
    news_sents = fetch_news_sentences()
    print(f"      Предложений с числами: {len(news_sents)}")
    news_pairs = process_news_sentences(news_sents)
    print(f"      Пар: {len(news_pairs)}")
    all_pairs.extend(news_pairs)

    print(f"[2/4] Генерация из шаблонов...")
    tp = generate_from_templates(max(5000, total_target // 4))
    print(f"      Пар: {len(tp)}")
    all_pairs.extend(tp)

    print(f"[3/4] Генерация группировки...")
    gp = generate_grouping_examples(max(3000, total_target // 5))
    print(f"      Пар: {len(gp)}")
    all_pairs.extend(gp)

    print(f"[4/4] Генерация порядковых...")
    op = generate_ordinals(max(1500, total_target // 10))
    print(f"      Пар: {len(op)}")
    all_pairs.extend(op)

    random.shuffle(all_pairs)
    all_pairs = all_pairs[:total_target]

    df = pl.DataFrame(all_pairs)
    df.write_ipc(output_path)

    print(f"\nСохранено: {output_path}")
    print(f"  Строк:      {len(df)}")
    print(f"  Из новостей: {len(news_pairs)}")
    print(f"  Шаблоны:    {len(tp)}")
    print(f"  Группировка: {len(gp)}")
    print(f"  Порядковые:  {len(op)}")
    print(
        f"  Средняя длина task_text: {df.select(pl.col('task_text').str.len_bytes().mean()).item():.1f}"
    )
    return df


if __name__ == "__main__":
    import time

    start = time.time()
    df = generate_dataset()
    print(f"\nВремя: {time.time() - start:.1f} сек")
