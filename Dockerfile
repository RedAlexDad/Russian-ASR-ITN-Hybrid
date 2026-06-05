# ====================================================================
# Stage 1: builder — установка Python-зависимостей
# ====================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ====================================================================
# Stage 2: runtime — минимальный образ
# ====================================================================
FROM python:3.12-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# site-packages из builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# исходный код (слои по частоте изменений)
COPY src/ src/
COPY main.py Makefile ./
COPY makefiles/ makefiles/
COPY scripts/ scripts/
COPY reports/ reports/

CMD ["tail", "-f", "/dev/null"]
