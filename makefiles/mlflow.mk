# ════════════════════════════════════════════════════════════
# MLFLOW — трекинг экспериментов
# ════════════════════════════════════════════════════════════

MLFLOW_PORT   ?= 5001
MLFLOW_HOST   ?= 0.0.0.0
MLFLOW_DB     ?= mlflow.db
MLFLOW_ARTIFACTS ?= $(CURDIR)/mlflow-artifacts

.PHONY: mlflow-up mlflow-down mlflow-ui mlflow-clean

mlflow-up:
	@printf "$(GREEN)${BOLD}[MLFLOW]$(NC)  Запуск MLflow сервера на порту $(MLFLOW_PORT)...\n"
	@mkdir -p $(MLFLOW_ARTIFACTS)
	nohup mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///$(MLFLOW_DB) \
		--default-artifact-root $(MLFLOW_ARTIFACTS) \
		> /tmp/mlflow.log 2>&1 &
	@sleep 2
	@printf "$(GREEN)${BOLD}[OK]$(NC)       MLflow UI: http://localhost:$(MLFLOW_PORT)\n"

mlflow-down:
	@printf "$(RED)${BOLD}[MLFLOW]$(NC)  Остановка MLflow сервера...\n"
	@pkill -f "mlflow" 2>/dev/null || true
	@sleep 2
	@printf "$(GREEN)${BOLD}[OK]$(NC)       MLflow остановлен\n"

mlflow-ui:
	@printf "$(BLUE)${BOLD}[MLFLOW]$(NC)  Открыть MLflow UI...\n"
	@xdg-open http://localhost:$(MLFLOW_PORT) 2>/dev/null || \
		open http://localhost:$(MLFLOW_PORT) 2>/dev/null || \
		printf "  http://localhost:$(MLFLOW_PORT)\n"

mlflow-clean:
	@printf "$(MAGENTA)${BOLD}[MLFLOW]$(NC)  Очистка MLflow данных...\n"
	rm -rf $(MLFLOW_DB) $(MLFLOW_ARTIFACTS) /tmp/mlflow.log
	@printf "$(GREEN)${BOLD}[OK]$(NC)       MLflow данные удалены\n"
