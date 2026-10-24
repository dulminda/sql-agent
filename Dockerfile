FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY benchmarks ./benchmarks
RUN pip install . && sqlagent data --dest /app/data/chinook.sqlite
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# Pass ANTHROPIC_API_KEY at runtime to enable /v1/ask. Mount your own DB and set --db to use it.
CMD ["sqlagent", "serve", "--db", "/app/data/chinook.sqlite", "--config", "configs/agent_full.yaml", "--host", "0.0.0.0", "--port", "8000"]
