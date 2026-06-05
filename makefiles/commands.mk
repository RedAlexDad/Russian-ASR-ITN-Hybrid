# ════════════════════════════════════════════════════════════
# КОМАНДЫ В КОНТЕЙНЕРЕ
# ════════════════════════════════════════════════════════════

.PHONY: run evaluate errors test validate eda

run:
	$(require-container)
	@printf "$(GREEN)${BOLD}[RUN]$(NC)      Нормализация: $(INPUT) → $(OUTPUT)\n"
	$(COMPOSE) exec app $(PY) main.py run "$(INPUT)" -o "$(OUTPUT)"

evaluate:
	$(require-container)
	@printf "$(YELLOW)${BOLD}[EVAL]$(NC)     Оценка accuracy на: $(CALIB)\n"
	$(COMPOSE) exec app $(PY) main.py evaluate "$(CALIB)"

errors:
	$(require-container)
	@printf "$(RED)${BOLD}[ERRORS]$(NC)    Первые $(or $(N),15) ошибок на: $(CALIB)\n"
	$(COMPOSE) exec app $(PY) main.py errors "$(CALIB)" -n $(or $(N),15)

test:
	$(require-container)
	@printf "$(BLUE)${BOLD}[TEST]$(NC)     Запуск тестов...\n"
	$(COMPOSE) exec app $(PY) -m pytest -v

validate:
	@printf "$(BLUE)${BOLD}[VALIDATE]$(NC) Проверка ноутбука...\n"
	$(PY) scripts/validate_notebook.py $(NOTEBOOK)

eda:
	$(require-container)
	@printf "$(BLUE)${BOLD}[EDA]$(NC)      Запуск EDA...\n"
	$(COMPOSE) exec app $(PY) scripts/eda.py

synthetic:
	$(require-container)
	@printf "$(BLUE)${BOLD}[SYNTHETIC]$(NC) Генерация синтетического датасета...\n"
	$(COMPOSE) exec app $(PY) scripts/generate_synthetic.py

evaluate-synthetic:

evaluate-synthetic:
	$(require-container)
	@printf "$(YELLOW)${BOLD}[EVAL-SYNTH]$(NC) Оценка accuracy на синтетике...\n"
	$(COMPOSE) exec app $(PY) -c "import polars as pl; from src.normalizer import normalize_text; df = pl.read_ipc('data/synthetic.f'); \
	  [print(f'  {l}: {c}/{s.height} = {c/s.height*100:.2f}%') for l in ['clean','noisy'] \
	  if (s:=df.filter(pl.col('noise_level')==l)).height>0 and (c:=sum(1 for r in s.iter_rows(named=True) \
	  if normalize_text(r['task_text'])==r['ground_truth']))]; \
	  c=sum(1 for r in df.iter_rows(named=True) if normalize_text(r['task_text'])==r['ground_truth']); \
	  print(f'  overall: {c}/{len(df)} = {c/len(df)*100:.2f}%')"

fetch-real:
	$(require-container)
	@printf "$(BLUE)${BOLD}[FETCH]$(NC)     Сбор реальных данных из интернета...\n"
	$(COMPOSE) exec app $(PY) scripts/fetch_real_data.py

evaluate-real:
	$(require-container)
	@printf "$(YELLOW)${BOLD}[EVAL-REAL]$(NC) Оценка accuracy на реальных данных...\n"
	$(COMPOSE) exec app $(PY) -c "import polars as pl; from src.normalizer import normalize_text; \
	  df = pl.read_ipc('data/real.f'); \
	  c = sum(1 for r in df.iter_rows(named=True) if normalize_text(r['task_text']) == r['ground_truth']); \
	  print(f'  Accuracy: {c}/{len(df)} = {c/len(df)*100:.2f}%')"

train:
	@printf "$(BLUE)${BOLD}[TRAIN]$(NC)    Обучение ruT5-small на синтетике...\n"
	$(COMPOSE) exec app $(PY) scripts/train.py --data data/synthetic.f --epochs $(or $(EPOCHS),3) --batch-size $(or $(BATCH_SIZE),8) $(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES),) --mlflow

