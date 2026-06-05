# ════════════════════════════════════════════════════════════
# КОМАНДЫ В КОНТЕЙНЕРЕ
# ════════════════════════════════════════════════════════════

.PHONY: run evaluate errors test validate eda

run:
	@printf "$(GREEN)${BOLD}[RUN]$(NC)      Нормализация: $(INPUT) → $(OUTPUT)\n"
	$(COMPOSE) exec app $(PY) main.py run "$(INPUT)" -o "$(OUTPUT)"

evaluate:
	@printf "$(YELLOW)${BOLD}[EVAL]$(NC)     Оценка accuracy на: $(CALIB)\n"
	$(COMPOSE) exec app $(PY) main.py evaluate "$(CALIB)"

errors:
	@printf "$(RED)${BOLD}[ERRORS]$(NC)    Первые $(or $(N),15) ошибок на: $(CALIB)\n"
	$(COMPOSE) exec app $(PY) main.py errors "$(CALIB)" -n $(or $(N),15)

test:
	@printf "$(BLUE)${BOLD}[TEST]$(NC)     Запуск тестов...\n"
	$(COMPOSE) exec app $(PY) -m pytest -v

validate:
	@printf "$(BLUE)${BOLD}[VALIDATE]$(NC) Проверка ноутбука...\n"
	$(PY) scripts/validate_notebook.py $(NOTEBOOK)

eda:
	@printf "$(BLUE)${BOLD}[EDA]$(NC)      Запуск EDA...\n"
	$(COMPOSE) exec app $(PY) scripts/eda.py
