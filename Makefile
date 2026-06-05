.PHONY: help run evaluate errors test clean docker-build docker-run
.DEFAULT_GOAL := help

DATA_DIR  := data
INPUT     ?= $(DATA_DIR)/test.f
CALIB     ?= $(DATA_DIR)/calibration.f
OUTPUT    ?= answer.f
PY        := python3

# ─── Colors ──────────────────────────────────────────
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
BLUE   := \033[0;34m
MAGENTA:= \033[0;35m
CYAN   := \033[0;36m
BOLD   := \033[1m
RESET  := \033[0m

# ─── Help (default) ──────────────────────────────────
help:
	@printf "$(CYAN)%-10s$(RESET) %s\n"  "target"     "description"
	@printf "$(CYAN)%-10s$(RESET) %s\n"  "------"     "-----------"
	@printf "$(GREEN)%-10s$(RESET) %s\n" "run"        "Запустить нормализацию:  make run INPUT=data/test.f"
	@printf "$(YELLOW)%-10s$(RESET) %s\n" "evaluate"   "Оценить accuracy:        make evaluate CALIB=data/calibration.f"
	@printf "$(RED)%-10s$(RESET) %s\n"   "errors"     "Показать ошибки:         make errors CALIB=data/calibration.f N=15"
	@printf "$(BLUE)%-10s$(RESET) %s\n"  "test"       "Запустить тесты (pytest)"
	@printf "$(MAGENTA)%-10s$(RESET) %s\n" "clean"    "Очистить временные файлы"
	@printf "$(GREEN)%-10s$(RESET) %s\n"  "docker"   "Собрать и запустить через Docker:  make docker CMD=evaluate"
	@echo ""
	@printf "$(BOLD)Переменные:$(RESET)\n"
	@printf "  INPUT=path   — входной .feather (умолч: data/test.f)\n"
	@printf "  CALIB=path   — calibration.f (умолч: data/calibration.f)\n"
	@printf "  OUTPUT=path  — выходной .feather (умолч: answer.f)\n"
	@printf "  N=число      — количество ошибок (умолч: 15)\n"

# ─── Commands ────────────────────────────────────────
run:
	@printf "$(GREEN)[RUN]$(RESET) Нормализация: $(INPUT) → $(OUTPUT)\n"
	$(PY) main.py run "$(INPUT)" -o "$(OUTPUT)"

evaluate:
	@printf "$(YELLOW)[EVAL]$(RESET) Оценка accuracy на: $(CALIB)\n"
	$(PY) main.py evaluate "$(CALIB)"

errors:
	@printf "$(RED)[ERRORS]$(RESET) Первые $(or $(N),15) ошибок на: $(CALIB)\n"
	$(PY) main.py errors "$(CALIB)" -n $(or $(N),15)

test:
	@printf "$(BLUE)[TEST]$(RESET) Запуск тестов...\n"
	$(PY) -m pytest -v

clean:
	@printf "$(MAGENTA)[CLEAN]$(RESET) Очистка...\n"
	rm -rf __pycache__ src/__pycache__ src/dicts/__pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@printf "$(GREEN)[OK]$(RESET) Готово\n"

docker:
	@printf "$(GREEN)[DOCKER]$(RESET) Сборка и запуск: docker compose run --rm app $(CMD)\n"
	docker compose build
	docker compose run --rm app $(CMD)
