#!/usr/bin/env python3
"""
Гибридный нормализатор: парсер (97.6%) + ruT5 fallback (тяжёлые случаи).

Логика:
  1. Запустить rule-based парсер
  2. Оценить уверенность парсера (хардкодные правила)
  3. Если уверенность >= 0.75 — ответ парсера
  4. Если уверенность < 0.75 и модель загружена — ответ ruT5

Загрузка модели lazy — только при первом fallback.
"""

import os
import re
import sys

from src.lexicon import is_ordinal_word, lookup_word
from src.normalizer import normalize_text

# ── Lazy model loading ──
_model = None
_tokenizer = None


def _load_model():
    """Загружает ruT5 + LoRA адаптеры (один раз)."""
    global _model, _tokenizer
    if _model is not None:
        return True

    model_path = "models/ruT5-itn"
    if not os.path.exists(model_path):
        return False

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        base_name = "cointegrated/ruT5-small"
        _tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        _model = PeftModel.from_pretrained(base, model_path, tokenizer=None)
        _model.eval()
        return True
    except Exception as e:
        print(f"[HYBRID] Model load failed: {e}", file=sys.stderr)
        return False


def _t5_generate(text, max_len=64):
    """Генерирует ответ через ruT5."""
    if _model is None:
        return None
    import torch

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=max_len,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Confidence estimation ──


def _parser_confidence(text, pred):
    """Оценивает уверенность парсера в результате (0.0 - 1.0).

    Хардкодные правила, проверенные на 12 ошибках парсера в calibration.f.
    """
    score = 1.0
    tokens = text.lower().split()
    pred_tokens = pred.lower().split()

    # 1. "тыщ"/"тыща" — парсер конвертит в "1000" вместо сохранения
    if any(w in ("тыщ", "тыща") for w in tokens):
        score -= 0.4

    # 2. "дваста" — ASR-искажение "двести"
    if "дваста" in tokens:
        score -= 0.3

    # 3. "двеси" + compound-числа (200 350000 вместо 235000)
    if "двеси" in tokens:
        nums = [int(x) for x in re.findall(r"\d+", pred)]
        for i in range(len(nums) - 1):
            if nums[i] <= 300 and nums[i + 1] >= 1000:
                score -= 0.3
                break

    # 4. Порядковые в выводе — парсер их не конвертировал
    _ord_suffixes = ("ый", "ий", "ой", "ая", "ое", "ые")
    _ord_roots = ("перв", "втор", "трет", "четверт", "пят", "шест",
                  "седьм", "восьм", "девят", "десят",
                  "одиннадцат", "двенадцат", "тринадцат",
                  "четырнадцат", "пятнадцат", "шестнадцат",
                  "семнадцат", "восемнадцат", "девятнадцат",
                  "двадцат", "тридцат")
    _non_ordinals = {
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
        "международный", "современный", "второй",
    }
    for w in pred_tokens:
        wc = w.strip(".,!?;:()[]{}«»\"").replace("ё", "е").replace("Ё", "Е").lower()
        if wc in _non_ordinals:
            continue
        if wc.endswith(_ord_suffixes):
            for r in _ord_roots:
                if r in wc:
                    score -= 0.3
                    break

    # 5. "тысячам"/"тысячами" в выводе — парсер оставил форму мн.ч.
    if any(w in ("тысячам", "тысячами") for w in pred_tokens):
        score -= 0.3

    # 6. "миллион" во входе — возможна склейка
    if any("миллион" in w for w in tokens):
        score -= 0.3

    # 7. Слово с префиксом числа + "тысяч" — compound (дветысячи→2000+500)
    _number_prefixes = {"две", "три", "четыре", "пять", "шест",
                        "сем", "восем", "девят", "десят"}
    for w in tokens:
        if "тысяч" in w:
            for pfx in _number_prefixes:
                if w.startswith(pfx) and len(w) > len(pfx) + 3:
                    score -= 0.3
                    break

    # 8. Сдвоенные "два два" — парсер склеивает неверно
    for i in range(len(tokens) - 1):
        if tokens[i] == "два" and tokens[i + 1] == "два":
            score -= 0.3
            break

    return max(0.1, min(1.0, score))


# ── Main hybrid function ──


def hybrid_normalize(text):
    """Гибридная нормализация: парсер + ruT5 fallback."""
    if not isinstance(text, str) or not text.strip():
        return text

    # Шаг 1: rule-based парсер
    pred_rule = normalize_text(text)
    confidence = _parser_confidence(text, pred_rule)

    # Шаг 2: если уверенность низкая — пробуем ruT5
    if confidence < 0.75:
        if _load_model():
            pred_t5 = _t5_generate(text)
            if pred_t5 and pred_t5 != text:
                return pred_t5

    return pred_rule
