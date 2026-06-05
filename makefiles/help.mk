# ════════════════════════════════════════════════════════════
# HELP
# ════════════════════════════════════════════════════════════

.PHONY: help

help:
	@printf "$(BOLD)╔══════════════════════════════════════════════════════════════╗$(NC)\n"
	@printf "$(BOLD)║$(NC)  $(CYAN)russian-asr-itn-hybrid$(NC) — обратная текстовая нормализация  $(BOLD)║$(NC)\n"
	@printf "$(BOLD)╚══════════════════════════════════════════════════════════════╝$(NC)\n"
	@echo ""
	@printf "  $(CYAN)использование:$(NC)  make $(GREEN)<цель>$(NC)\n"
	@echo ""
	@printf "$(CYAN)╭─ Docker (compose v2) ────────────────────────────────────────$(NC)\n"
	@printf "$(GREEN)  deploy$(NC)             Рекомендуемый старт: down + build + up\n"
	@printf "$(GREEN)  build$(NC)              Сборка образа (с кэшем)\n"
	@printf "$(GREEN)  up$(NC)                 Запустить контейнер (фон)\n"
	@printf "$(RED)  down$(NC)               Остановить контейнер\n"
	@printf "$(MAGENTA)  clean$(NC)             Очистить образы и volumes\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@printf "$(CYAN)╭─ Команды в контейнере ──────────────────────────────────────$(NC)\n"
	@printf "$(GREEN)  run$(NC)               Запустить нормализацию: $(YELLOW)make run INPUT=data/test.f$(NC)\n"
	@printf "$(YELLOW)  evaluate$(NC)           Оценить accuracy:       $(YELLOW)make evaluate$(NC)\n"
	@printf "$(RED)  errors$(NC)             Показать ошибки:        $(YELLOW)make errors N=15$(NC)\n"
	@printf "$(BLUE)  test$(NC)               Запустить тесты (pytest)\n"
	@printf "$(BLUE)  validate$(NC)           Проверить ноутбук:      $(YELLOW)make validate NOTEBOOK=notebooks/eda.ipynb$(NC)\n"
	@printf "$(BLUE)  synthetic$(NC)         Сгенерировать синтетический датасет\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@printf "$(CYAN)╭─ EDA ────────────────────────────────────────────────────────$(NC)\n"
	@printf "$(BLUE)  eda$(NC)                Запустить EDA (8 графиков в reports/plots/)\n"
	@printf "$(CYAN)╰──────────────────────────────────────────────────────────────$(NC)\n"
	@echo ""
	@printf "  $(YELLOW)переменные:$(NC)\n"
	@printf "    INPUT=path   — входной .feather        (умолч: data/test.f)\n"
	@printf "    CALIB=path   — calibration.f            (умолч: data/calibration.f)\n"
	@printf "    OUTPUT=path  — выходной .feather        (умолч: answer.f)\n"
	@printf "    NOTEBOOK=path — путь до ноутбука         (умолч: notebooks/eda.ipynb)\n"
	@printf "    N=число      — количество ошибок        (умолч: 15)
    EPOCHS=число — эпох обучения            (умолч: 3)
    BATCH_SIZE=число — размер батча         (умолч: 8)\n"
	@echo ""
	@printf "  $(YELLOW)пример:$(NC)\n"
	@printf "    make deploy && make synthetic && make evaluate && make down\n"
