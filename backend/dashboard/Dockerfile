# Streamlit dashboard (FR-10). Build from the repository root:
#   docker build -f backend/dashboard/Dockerfile -t prana-dashboard .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_BASE_URL_PATH=dashboard \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer

RUN useradd --create-home --uid 10001 prana
WORKDIR /app
COPY backend/dashboard/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY backend/dashboard/app.py backend/dashboard/data.py ./

USER prana
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/dashboard/_stcore/health', timeout=4).status == 200 else 1)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
