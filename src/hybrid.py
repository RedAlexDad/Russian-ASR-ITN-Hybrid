#!/usr/bin/env python3
"""
Гибридный нормализатор: парсер (97.6%) + ruT5 fallback (тяжёлые случаи).

Логика:
  1. Запустить rule-based парсер
  2. Оценить уверенность парсера
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
        _tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        _model = PeftModel.from_pretrained(base, model_path)
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

    Учитывает:
      - наличие ASR-искажений в исходном тексте
      - наличие слов вне словаря
      - склейки числительных
      - аномалии в структуре числа
    """
    score = 1.0

    # 1. ASR-искажения в исходном тексте
    from src.dicts.asr_errors import ASR_ERRORS

    asr_count = sum(1 for w in text.lower().split() if w in ASR_ERRORS)
    if asr_count > 0:
        score -= 0.15 * min(asr_count, 3)
        # Если текст содержит "двеси", "тыщ" — почти всегда проблема
        for w in text.lower().split():
            if w in ("двеси", "дваста", "тыщ", "тыщи"):
                score -= 0.1

    # 2. Неизвестные слова (похожи на числительные но не в словаре)
    unknown = 0
    for w in text.lower().split():
        if any(c.isdigit() for c in w):
            continue
        if lookup_word(w) is None and not is_ordinal_word(w):
            unknown += 1
    if unknown > 2:
        score -= 0.1 * min(unknown, 5)

    # 3. Склейки (слитные написания)
    merged = 0
    for w in text.lower().split():
        if any(x in w for x in ("тысячи", "тысяч", "миллион", "миллиард")):
            if len(w) > 12:
                merged += 1
    if merged > 0:
        score -= 0.2

    # 4. Есть числа в предсказании — проверяем разумность
    nums = re.findall(r"\d+", pred)
    if not nums and re.findall(r"\d+", text):
        score -= 0.3  # были цифры в исходнике, но нет в ответе

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
            # Если ruT5 вернул то же самое или None — оставляем парсер

    return pred_rule
