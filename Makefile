# Russian ASR ITN Hybrid — Makefile

SHELL := /bin/bash

# ════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════

DATA_DIR  := data
INPUT     ?= $(DATA_DIR)/test.f
CALIB     ?= $(DATA_DIR)/calibration.f
OUTPUT    ?= answer.f
PY        := python3

# ════════════════════════════════════════════════════════════
# ЦВЕТА
# ════════════════════════════════════════════════════════════

BLUE    := \033[0;34m
GREEN   := \033[0;32m
YELLOW  := \033[1;33m
RED     := \033[0;31m
CYAN    := \033[0;36m
MAGENTA := \033[0;35m
BOLD    := \033[1m
NC      := \033[0m

# ════════════════════════════════════════════════════════════
# ЦЕЛИ
# ════════════════════════════════════════════════════════════

.PHONY: all help run evaluate errors test clean docker

all: help

help:
	@printf "$(BOLD)╔══════════════════════════════════════════════════════════════╗$(NC)\n"
	@printf "$(BOLD)║$(NC)  $(CYAN)russian-asr-itn-hybrid$(NC) — обратная текстовая нормализация    $(BOLD)║$(NC)\n"
	@printf "$(BOLD)╚══════════════════════════════════════════════════════════════╝$(NC)\n"
	@echo ""
	@printf "  $(CYAN)использование:$(NC)  make $(GREEN)<цель>$(NC)\n"
	@echo ""
	@printf "$(CYAN)╭─ Основные команды ───────────────────────────────────────────$(NC)\n"
	@printf "$(GREEN)  run$(NC)               Запустить нормализацию:          $(YELLOW)make run INPUT=data/test.f$(NC)\n"
	@printf "$(YELLOW)  evaluate$(NC)           Оценить accuracy:                $(YELLOW)make evaluate CALIB=data/calibration.f$(NC)\n"
	@printf "$(RED)  errors$(NC)             Показать ошибки:                 $(YELLOW)make errors CALIB=data/calibration.f N=15$(NC)\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@printf "$(CYAN)╭─ Разработка ─────────────────────────────────────────────────$(NC)\n"
	@printf "$(BLUE)  test$(NC)               Запустить тесты (pytest)\n"
	@printf "$(MAGENTA)  clean$(NC)              Очистить временные файлы\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@printf "$(CYAN)╭─ Docker (compose v2) ────────────────────────────────────────$(NC)\n"
	@printf "$(GREEN)  docker$(NC)             Запустить через Docker:         $(YELLOW)make docker CMD=evaluate$(NC)\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@echo ""
	@printf "  $(YELLOW)переменные:$(NC)\n"
	@printf "    INPUT=path   — входной .feather        (умолч: data/test.f)\n"
	@printf "    CALIB=path   — calibration.f            (умолч: data/calibration.f)\n"
	@printf "    OUTPUT=path  — выходной .feather        (умолч: answer.f)\n"
	@printf "    CMD=target   — команда для docker       (умолч: help)\n"
	@printf "    N=число      — количество ошибок        (умолч: 15)\n"
	@echo ""
	@printf "  $(YELLOW)примеры:$(NC)\n"
	@printf "    make evaluate\n"
	@printf "    make run INPUT=data/test.f\n"
	@printf "    make docker CMD=evaluate\n"

run:
	@printf "$(GREEN)${BOLD}[RUN]$(NC)      Нормализация: $(INPUT) → $(OUTPUT)\n"
	$(PY) main.py run "$(INPUT)" -o "$(OUTPUT)"

evaluate:
	@printf "$(YELLOW)${BOLD}[EVAL]$(NC)     Оценка accuracy на: $(CALIB)\n"
	$(PY) main.py evaluate "$(CALIB)"

errors:
	@printf "$(RED)${BOLD}[ERRORS]$(NC)    Первые $(or $(N),15) ошибок на: $(CALIB)\n"
	$(PY) main.py errors "$(CALIB)" -n $(or $(N),15)

test:
	@printf "$(BLUE)${BOLD}[TEST]$(NC)     Запуск тестов...\n"
	$(PY) -m pytest -v

clean:
	@printf "$(MAGENTA)${BOLD}[CLEAN]$(NC)    Очистка...\n"
	rm -rf __pycache__ src/__pycache__ src/dicts/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Готово\n"

docker:
	@printf "$(GREEN)${BOLD}[DOCKER]$(NC)   Сборка: docker compose run --rm app $(or $(CMD),help)\n"
	docker compose build
	docker compose run --rm app $(or $(CMD),help)

# ════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
