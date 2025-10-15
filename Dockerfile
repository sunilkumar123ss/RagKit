FROM python:3.12-slim
WORKDIR /app
ARG RAGKIT_EXTRAS=web,dense,providers,mcp,langchain
COPY pyproject.toml README.md ./
COPY ragkit ./ragkit
RUN python -m pip install --no-cache-dir ".[${RAGKIT_EXTRAS}]" \
    && groupadd --gid 10001 ragkit \
    && useradd --uid 10001 --gid ragkit --no-create-home ragkit \
    && mkdir -p /data/models \
    && chown -R ragkit:ragkit /data
ENV RAGKIT_DATA_DIR=/data/corpus HF_HOME=/data/models PYTHONUNBUFFERED=1
USER ragkit
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=3)" || exit 1
CMD ["ragkit", "serve", "--host", "0.0.0.0", "--port", "8000"]
