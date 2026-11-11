# One image for the API and the worker (different commands in compose).
# Build from the repository root:  docker build -f backend/Dockerfile -t prana-app .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend/src:/app/src

# git: sparse checkouts for retrieval when RAG_ENABLED=1.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 prana \
 && mkdir -p /var/cache/prana && chown prana /var/cache/prana

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Semgrep is a CLI with its own heavy dependencies (they conflict with the
# app's pins), so it lives in an isolated venv and only the command is exposed.
# Rule packs are fetched once at build time: reviews never download rules and
# the image pins the exact rules it was built with.
RUN python -m venv /opt/semgrep-venv \
 && /opt/semgrep-venv/bin/pip install semgrep==1.183.0 \
 && ln -s /opt/semgrep-venv/bin/semgrep /usr/local/bin/semgrep \
 && mkdir -p /opt/semgrep \
 && python -c "import urllib.request as u; [u.urlretrieve(f'https://semgrep.dev/c/p/{p}', f'/opt/semgrep/{p}.yml') for p in ('default', 'secrets')]" \
 && semgrep scan --metrics off --quiet --config /opt/semgrep/default.yml --config /opt/semgrep/secrets.yml /dev/null
ENV SEMGREP_CONFIGS=/opt/semgrep/default.yml,/opt/semgrep/secrets.yml \
    SEMGREP_SEND_METRICS=off

# The model-side package is needed only for its stdlib client and retrieval
# contract; nothing here imports torch.
COPY src/prana src/prana
COPY backend/src backend/src

USER prana
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "prana_app.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
