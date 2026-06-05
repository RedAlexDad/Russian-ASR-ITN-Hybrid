# ──────────────────────────────────────────────────────────────
# russian-asr-itn-hybrid — корневой Makefile
# Все цели вынесены в makefiles/*.mk
# ──────────────────────────────────────────────────────────────

include makefiles/config.mk
include makefiles/help.mk
include makefiles/docker.mk
include makefiles/commands.mk
include makefiles/local.mk
include makefiles/mlflow.mk

# Цель по умолчанию
.DEFAULT_GOAL := help
