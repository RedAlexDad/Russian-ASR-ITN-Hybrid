# ════════════════════════════════════════════════════════════
# DOCKER LIFECYCLE
# ════════════════════════════════════════════════════════════

.PHONY: build up down deploy clean

build:
	@printf "$(GREEN)${BOLD}[BUILD]$(NC)    Сборка образа...\n"
	$(COMPOSE) build

up:
	@printf "$(GREEN)${BOLD}[UP]$(NC)       Запуск контейнера...\n"
	$(COMPOSE) up -d
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Контейнер запущен\n"

down:
	@printf "$(RED)${BOLD}[DOWN]$(NC)     Остановка контейнера...\n"
	$(COMPOSE) down
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Контейнер остановлен\n"

deploy:
	@printf "$(GREEN)${BOLD}[DEPLOY]$(NC)  down + build + up...\n"
	$(MAKE) down
	$(MAKE) build
	$(MAKE) up
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Развёрнуто\n"

clean:
	@printf "$(MAGENTA)${BOLD}[CLEAN]$(NC)    Очистка Docker-ресурсов...\n"
	$(COMPOSE) down -v --rmi all --remove-orphans 2>/dev/null || true
	@printf "$(GREEN)${BOLD}[OK]$(NC)       Готово\n"
