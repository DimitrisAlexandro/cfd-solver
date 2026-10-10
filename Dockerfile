# AeroLab on your own server:  docker compose up -d   ->  http://<server>:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AEROLAB_CASES=/data/cases NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# optional MKL direct solver for the coupled u-v-p option (~1 GB): docker compose build --build-arg PARDISO=1
ARG PARDISO=0
RUN if [ "$PARDISO" = "1" ]; then pip install pypardiso; fi

COPY .streamlit .streamlit
COPY app app
COPY src src
COPY data data
COPY validation validation

RUN useradd -m aerolab && mkdir -p /data/cases /tmp/numba && chown -R aerolab /data /tmp/numba
USER aerolab
VOLUME /data/cases
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/AeroLab.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
