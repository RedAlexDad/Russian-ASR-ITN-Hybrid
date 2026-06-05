FROM python:3.12-slim

WORKDIR /app

# Layer 1: system dependencies (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python dependencies (cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: source code (cached until src/ changes)
COPY src/ src/

# Layer 4: entry points and config (cached until these files change)
COPY main.py Makefile ./
COPY makefiles/ makefiles/
COPY scripts/ scripts/

# Layer 5: everything else (reports, docs — changes least often)
COPY reports/ reports/

CMD ["tail", "-f", "/dev/null"]
