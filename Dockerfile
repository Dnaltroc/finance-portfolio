# Streamlit app over finlib. Build: docker build -t finance-portfolio .
#                            Run:   docker run -p 8501:8501 finance-portfolio
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencies first so code edits do not reinstall them (layer cache).
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Non-root user. The theme lives in its home so Streamlit finds it from any working dir.
RUN useradd --uid 10001 --create-home --home-dir /home/app app \
 && mkdir /work && chown app /work
COPY --chown=app .streamlit/config.toml /home/app/.streamlit/config.toml

# Code, plus the offline fixtures the providers read when there is no SEC identity.
COPY finlib finlib
COPY projects projects
COPY app app
COPY tests/fixtures tests/fixtures

USER 10001
# The SQLite cache and any local files are written to the working dir, not the code dir.
WORKDIR /work
ENV PYTHONPATH=/app HOME=/home/app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "/app/app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
