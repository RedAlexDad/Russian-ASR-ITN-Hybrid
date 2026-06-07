# ════════════════════════════════════════════════════════════
# ЛОКАЛЬНЫЙ ЗАПУСК (через .venv, без Docker)
# ════════════════════════════════════════════════════════════

.PHONY: run-local evaluate-local errors-local test-local eda-local synthetic-local train-local fetch-data-local

VENV_PY := .venv/bin/python3
VENV_PY := .venv/bin/python

run-local:
	@printf "$(GREEN)${BOLD}[RUN]$(NC)      Нормализация: $(INPUT) → $(OUTPUT)\n"
	$(VENV_PY) main.py run "$(INPUT)" -o "$(OUTPUT)"

evaluate-local:
	@printf "$(YELLOW)${BOLD}[EVAL]$(NC)     Оценка accuracy на: $(CALIB)\n"
	$(VENV_PY) main.py evaluate "$(CALIB)"

errors-local:
	@printf "$(RED)${BOLD}[ERRORS]$(NC)    Первые $(or $(N),15) ошибок на: $(CALIB)\n"
	$(VENV_PY) main.py errors "$(CALIB)" -n $(or $(N),15)

test-local:
	@printf "$(BLUE)${BOLD}[TEST]$(NC)     Запуск тестов...\n"
	$(VENV_PY) -m pytest -v

validate-local:
	@printf "$(BLUE)${BOLD}[VALIDATE]$(NC) Проверка ноутбука...\n"
	$(VENV_PY) scripts/validate_notebook.py $(NOTEBOOK)

eda-local:
	@printf "$(BLUE)${BOLD}[EDA]$(NC)      Запуск EDA...\n"
	$(VENV_PY) scripts/eda.py

synthetic-local:
	@printf "$(BLUE)${BOLD}[SYNTHETIC]$(NC) Генерация синтетического датасета...\n"
	$(VENV_PY) scripts/generate_synthetic.py

train-local:
	@printf "$(BLUE)${BOLD}[TRAIN]$(NC)    Обучение ruT5-small на синтетике...\n"
	$(VENV_PY) scripts/train.py --data data/synthetic.f \
	  --epochs $(or $(EPOCHS),3) \
	  --batch-size $(or $(BATCH_SIZE),8) \
	  --lora-r $(LORA_R) \
	  --lora-alpha $(LORA_ALPHA) \
	  --noise-level $(NOISE_LEVEL) \
	  $(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES),) \
	  $(if $(MODEL_PATH),--model-path $(MODEL_PATH),) \
	  --mlflow

train-noisy:
	@printf "$(BLUE)${BOLD}[TRAIN-NOISY]$(NC) Обучение с noisy-данными (r=$(LORA_R))...\n"
	$(VENV_PY) scripts/train.py --data data/synthetic.f \
	  --epochs $(or $(EPOCHS),10) \
	  --batch-size $(or $(BATCH_SIZE),8) \
	  --lora-r $(LORA_R) \
	  --lora-alpha $(LORA_ALPHA) \
	  --noise-level noisy \
	  $(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES),) \
	  $(if $(MODEL_PATH),--model-path $(MODEL_PATH),) \
	  --mlflow

train-quick:
	@printf "$(BLUE)${BOLD}[TRAIN-QUICK]$(NC) Быстрое обучение (200 samples, 1 epoch)...\n"
	$(VENV_PY) scripts/train.py --data data/synthetic.f --quick --mlflow

fetch-data-local:
	@printf "$(BLUE)${BOLD}[FETCH-DATA]$(NC) Загрузка test.f и calibration.f с Google Drive...\n"
	@mkdir -p data
	gdown --folder "https://drive.google.com/drive/folders/1jgm9hGvZETTzhn4bu43e5UlI5gT12pud" -O data/ --remaining-ok
	@for f in data/*/*.f; do \
	  if [ -f "$$f" ]; then \
	    mv "$$f" data/; \
	  fi; \
	done
	@rmdir data/*/ 2>/dev/null; true

train-ngram-lm:
	@printf "$(BLUE)${BOLD}[NGRAM]$(NC)    Обучение char n-gram LM (order=$(NGRAM_ORDER))...\n"
	$(VENV_PY) scripts/train_ngram_lm.py --order $(NGRAM_ORDER)

fetch-real-local:
	@printf "$(BLUE)${BOLD}[FETCH]$(NC)     Сбор реальных данных из интернета...\n"
	$(VENV_PY) scripts/fetch_real_data.py

evaluate-sequence-local:
	@printf "$(YELLOW)${BOLD}[EVAL-SEQ]$(NC)  Sequence parser accuracy на: $(CALIB)\n"
	$(VENV_PY) main.py evaluate "$(CALIB)" --parser-type sequence

errors-sequence-local:
	@printf "$(RED)${BOLD}[ERR-SEQ]$(NC)    Первые $(or $(N),15) ошибок (sequence) на: $(CALIB)\n"
	$(VENV_PY) main.py errors "$(CALIB)" -n $(or $(N),15) --parser-type sequence

evaluate-synthetic-local:
	@printf "$(YELLOW)${BOLD}[EVAL-SYNTH]$(NC) Оценка accuracy на синтетике...\n"
	$(VENV_PY) -c "import polars as pl; from src.normalizer import normalize_text; \
	  df = pl.read_ipc('data/synthetic.f'); \
	  c = sum(1 for r in df.iter_rows(named=True) if normalize_text(r['task_text']) == r['ground_truth']); \
	  print(f'  Accuracy: {c}/{len(df)} = {c/len(df)*100:.2f}%')"

evaluate-real-local:
	@printf "$(YELLOW)${BOLD}[EVAL-REAL]$(NC) Оценка accuracy на реальных данных...\n"
	$(VENV_PY) -c "import polars as pl; from src.normalizer import normalize_text; \
	  df = pl.read_ipc('data/real.f'); \
	  c = sum(1 for r in df.iter_rows(named=True) if normalize_text(r['task_text']) == r['ground_truth']); \
	  print(f'  Accuracy: {c}/{len(df)} = {c/len(df)*100:.2f}%')"
