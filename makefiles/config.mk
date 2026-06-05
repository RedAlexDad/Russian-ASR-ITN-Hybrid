# ════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ════════════════════════════════════════════════════════════

SHELL := /bin/bash

DATA_DIR  := data
INPUT     ?= $(DATA_DIR)/test.f
CALIB     ?= $(DATA_DIR)/calibration.f
OUTPUT    ?= answer.f
PY        := python3
COMPOSE   := docker compose

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
# ЗАЩИТНЫЙ МЕХАНИЗМ: проверка что контейнер запущен
# ════════════════════════════════════════════════════════════

define require-container
	@if ! docker ps --format '{{.Names}}' | grep -q 'russian-asr-itn-hybrid'; then \
		printf "$(RED)${BOLD}[x]$(NC) $(RED)Контейнер не запущен.$(NC)\n" >&2; \
		printf "$(YELLOW)${BOLD}[!]$(NC) $(YELLOW)Запустите: make up$(NC)\n" >&2; \
		exit 1; \
	fi
endef
