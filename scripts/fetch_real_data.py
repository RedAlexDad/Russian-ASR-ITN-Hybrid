#!/usr/bin/env python3
"""
Сбор реальных данных с числами из интернета.

Источники:
  1. Wikipedia (русская) — random articles + extracts
  2. Новостные RSS — полные статьи по ссылкам
  3. Прямой парсинг новостных сайтов

Процесс:
  1. Сбор сырого текста
  2. Разбивка на предложения
  3. Фильтр: только предложения с числами
  4. Сохранение в data/raw_sentences.json
  5. Конвертация чисел в слова и создание task/ground_truth пар
  6. Сохранение в data/real.f
"""

import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# 1. Wikipedia
# ──────────────────────────────────────────────

WIKI_API = "https://ru.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ITNResearchBot/1.0 (research; itn@example.com)"}


def fetch_wikipedia(count=200):
    """Парсит случайные статьи из Wikipedia."""
    sentences = []
    seen_titles = set()

    for batch in range(count // 10 + 1):
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "random",
                "rnnamespace": 0,
                "rnlimit": 10,
            }
            r = requests.get(WIKI_API, params=params, timeout=15, headers=HEADERS)
            if not r.ok:
                continue

            pages = r.json().get("query", {}).get("random", [])
            titles = [p["title"] for p in pages if p["title"] not in seen_titles]
            for t in titles:
                seen_titles.add(t)

            # Get extracts for all titles at once
            ext_params = {
                "action": "query",
                "format": "json",
                "titles": "|".join(titles[:5]),
                "prop": "extracts",
                "explaintext": True,
                "exlimit": 5,
                "exchars": 2000,
            }
            r2 = requests.get(WIKI_API, params=ext_params, timeout=15, headers=HEADERS)
            if r2.ok:
                for _, data in r2.json().get("query", {}).get("pages", {}).items():
                    text = data.get("extract", "")
                    # Разбивка на предложения
                    for part in re.split(r"(?<=[.!?])\s+", text):
                        if re.search(r"\b\d{1,6}\b", part) and len(part) > 15:
                            sentences.append(part.strip()[:300])

            time.sleep(0.5)

        except Exception as ex:
            print(f"    [WARN] Wikipedia batch {batch}: {ex}")
            time.sleep(2)

    return sentences


# ──────────────────────────────────────────────
# 2. Новостные RSS с полными статьями
# ──────────────────────────────────────────────

NEWS_FEEDS = [
    ("https://tass.ru/rss/v2.xml", "tass"),
    ("https://ria.ru/export/rss2/archive/index.xml", "ria"),
    ("https://www.interfax.ru/rss.asp", "interfax"),
    ("https://russian.rt.com/rss/news", "rt"),
    ("https://lenta.ru/rss/news", "lenta"),
]


def fetch_news_articles(max_articles=50):
    """Получает полные тексты новостных статей по RSS-ссылкам."""
    sentences = []
    links_seen = set()

    for feed_url, source in NEWS_FEEDS:
        try:
            r = requests.get(feed_url, timeout=10, headers=HEADERS)
            if r.status_code != 200:
                continue

            root = ET.fromstring(r.content)
            links = []
            for item in root.iter("item"):
                link = item.find("link")
                if link is not None and link.text and link.text not in links_seen:
                    links.append(link.text)

            for link in links[:15]:
                if len(sentences) >= max_articles * 3:
                    break

                try:
                    ar = requests.get(link, timeout=10, headers=HEADERS)
                    if ar.status_code != 200:
                        continue

                    soup = BeautifulSoup(ar.text, "html.parser")
                    # Удаляем скрипты, стили, нав
                    for tag in soup(["script", "style", "nav", "header", "footer"]):
                        tag.decompose()

                    # Ищем основной контент
                    text = ""
                    for sel in [
                        "article",
                        ".article-body",
                        ".text",
                        ".news-text",
                        ".article__text",
                        "main",
                        ".content",
                        ". material-text",
                    ]:
                        el = soup.select_one(sel)
                        if el:
                            text = el.get_text(separator=" ", strip=True)
                            break
                    if not text:
                        text = soup.get_text(separator=" ", strip=True)

                    # Разбивка на предложения
                    for part in re.split(r"(?<=[.!?])\s+", text):
                        clean = re.sub(r"\s+", " ", part).strip()
                        if re.search(r"\b\d{1,6}\b", clean) and 30 < len(clean) < 300:
                            sentences.append(clean)

                    links_seen.add(link)
                    time.sleep(0.3)

                except Exception as ex:
                    print(f"    [WARN] Article {link[:50]}: {ex}")

        except Exception as ex:
            print(f"    [WARN] Feed {feed_url}: {ex}")

    return sentences


# ──────────────────────────────────────────────
# 3. Конвертация в ITN-формат
# ──────────────────────────────────────────────


def number_to_words(n):
    """Конвертер числа в слова (без external зависимостей)."""
    if n == 0:
        return "ноль"

    ONES = [
        "",
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
    ]
    ONES_F = [
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
    TEENS = [
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
    TENS = [
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
    HUNS = [
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
    T_TAIL = {
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

    def _h(n, fem=False):
        if n == 0:
            return []
        p = []
        p.append(HUNS[n // 100])
        n %= 100
        if 10 <= n <= 19:
            p.append(TEENS[n - 10])
            return [x for x in p if x]
        if n >= 20:
            p.append(TENS[n // 10])
            n %= 10
        if n > 0:
            p.append(ONES_F[n] if fem else ONES[n])
        return [x for x in p if x]

    parts = []
    if n >= 1000000:
        m = n // 1000000
        parts.extend(_h(m))
        parts.append("миллионов" if m > 1 else "миллион")
        n %= 1000000
    if n >= 1000:
        t = n // 1000
        parts.extend(_h(t, fem=True))
        parts.append(T_TAIL.get(t, "тысяч"))
        n %= 1000
    if n > 0:
        parts.extend(_h(n))
    return " ".join(parts)


def sentences_to_itn_pairs(sentences):
    """Из предложений с числами создаёт пары task/ground_truth.

    Для каждого числа в предложении:
      - ground_truth: исходное предложение (с цифрами)
      - task_text: предложение, где число заменено на словесную форму
    """
    pairs = []
    seen = set()

    for sent in sentences:
        # Нормализация
        sent = sent.lower()
        sent = re.sub(r"\s+", " ", sent).strip()

        # Ищем числа в предложении
        numbers = re.findall(r"\b(\d{1,6})\b", sent)
        if not numbers:
            continue

        # Берём первое число (чтобы не плодить миллион комбинаций)
        for num_str in numbers[:1]:
            n = int(num_str)
            if n == 0 or n > 999999:
                continue

            word_form = number_to_words(n)
            if not word_form:
                continue

            # Заменяем первое вхождение числа на словесную форму
            task = re.sub(r"\b" + num_str + r"\b", word_form, sent, count=1)
            answer = sent

            # Дедупликация
            key = hashlib.md5((task + answer).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            pairs.append({"task_text": task, "ground_truth": answer})

    return pairs


# ──────────────────────────────────────────────
# 4. Сборка и сохранение
# ──────────────────────────────────────────────


def main():
    import polars as pl

    print("Сбор реальных данных из интернета...\n")

    all_sentences = []

    print("[1/3] Wikipedia...")
    wiki_sents = fetch_wikipedia(count=200)
    print(f"      Предложений: {len(wiki_sents)}")
    all_sentences.extend(wiki_sents)

    print("[2/3] Новостные статьи...")
    news_sents = fetch_news_articles(max_articles=50)
    print(f"      Предложений: {len(news_sents)}")
    all_sentences.extend(news_sents)

    # Дедупликация
    seen = set()
    unique = []
    for s in all_sentences:
        key = s.lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    print("\n[3/3] Конвертация в ITN-формат...")
    print(f"      Уникальных предложений: {len(unique)}")

    pairs = sentences_to_itn_pairs(unique)
    print(f"      Пар task/ground_truth: {len(pairs)}")

    # Сохраняем сырые предложения
    raw_path = "data/raw_sentences.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=1)
    print(f"      Сырые предложения: {raw_path}")

    # Сохраняем ITN-датасет
    if pairs:
        df = pl.DataFrame(pairs)
        df.write_ipc("data/real.f")
        print(f"      ITN-датасет: data/real.f ({len(df)} rows)")

        # Базовая статистика
        print("\nСтатистика:")
        nums = []
        for gt in df["ground_truth"].to_list():
            nums.extend(re.findall(r"\d+", str(gt)))
        from collections import Counter

        dist = Counter(len(n) for n in nums)
        print(f"  Всего чисел: {len(nums)}")
        for k in sorted(dist):
            print(f"  {k}-digit: {dist[k]} ({dist[k] / len(nums) * 100:.1f}%)")
    else:
        print("  [WARN] Нет пар для сохранения!")


if __name__ == "__main__":
    main()
