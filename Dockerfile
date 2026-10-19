# AeroLab on your own server:  docker compose up -d   ->  http://<server>:8501
# One image, two services (docker-compose.yml): the web app, and the worker that runs the
# queued CFD jobs (python src/run.py jobqueue worker).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AEROLAB_CASES=/data/cases NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# optional MKL direct solver for the coupled u-v-p option (~1 GB): docker compose build --build-arg PARDISO=1
ARG PARDISO=0
RUN if [ "$PARDISO" = "1" ]; then pip install pypardiso; fi
# MUMPS for the adjoint (2x faster factorisation than the SuperLU fallback; ~60 MB):
# on by default, docker compose build --build-arg MUMPS=0 to leave it out. Debian ships
# no pkg-config files for the sequential MUMPS libraries, so they are written here.
ARG MUMPS=1
RUN if [ "$MUMPS" = "1" ]; then \
      { apt-get update \
        && apt-get install -y --no-install-recommends libmumps-seq-dev pkg-config gcc \
        && arch=$(gcc -print-multiarch) && mkdir -p /usr/local/lib/pkgconfig \
        && for t in s d c z; do \
             printf 'Name: %smumps_seq\nDescription: MUMPS\nVersion: 5\nLibs: -L/usr/lib/%s -l%smumps_seq -lmumps_common_seq -lmpiseq_seq\nCflags: -I/usr/include/mumps_seq\n' \
               $t $arch $t > /usr/local/lib/pkgconfig/${t}mumps_seq.pc || exit 1; \
           done \
        && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pip install python-mumps \
        && python -c "import mumps" \
        && echo "MUMPS installed"; \
      } || echo "WARNING: MUMPS could not be installed; the adjoint uses SuperLU (slower, same results)"; \
      apt-get purge -y gcc pkg-config >/dev/null 2>&1 || true; \
      apt-get autoremove -y >/dev/null 2>&1 || true; rm -rf /var/lib/apt/lists/*; \
    fi

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
