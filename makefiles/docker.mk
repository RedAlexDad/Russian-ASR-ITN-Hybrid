# ════════════════════════════════════════════════════════════
# DOCKER LIFECYCLE
# ════════════════════════════════════════════════════════════

.PHONY: build rebuild up down clean

build:
	@printf "$(GREEN)${BOLD}[BUILD]$(NC)    Сборка образа...\n"
	$(COMPOSE) build

rebuild:
	@printf "$(YELLOW)${BOLD}[REBUILD]$(NC)   Пересборка без кэша...\n"
	$(COMPOSE) build --no-cache

up:
	@printf "$(GREEN)${BOLD}[UP]$(NC)       Запуск контейнера...\n"
	$(COMPOSE) up -d
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Контейнер запущен\n"

down:
	@printf "$(RED)${BOLD}[DOWN]$(NC)     Остановка контейнера...\n"
	$(COMPOSE) down
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Контейнер остановлен\n"

clean:
	@printf "$(MAGENTA)${BOLD}[CLEAN]$(NC)    Очистка Docker-ресурсов...\n"
	$(COMPOSE) down -v --rmi all --remove-orphans 2>/dev/null || true
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Готово\n"
