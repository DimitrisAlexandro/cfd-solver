# AeroLab on your own server:  docker compose up -d --build   ->  http://<server>:8501
#
# One image, two services (docker-compose.yml): the web app, and the worker that runs the
# queued CFD jobs (python src/run.py jobqueue worker).
#
# Build options (docker compose build --build-arg NAME=value):
#   MUMPS=1          MUMPS for the adjoint (default on; 0 = SuperLU only, slower, same results)
#   NATIVE_MARCH     CPU target of the 3D solver's C++ kernel (default "native" = the CPU of the
#                    machine that builds the image; use x86-64-v2 for an image that runs anywhere)
#   PARDISO=1        MKL direct solver for the coupled u-v-p option (~1 GB, default off)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AEROLAB_CASES=/data/cases NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app

# 1. System libraries (rarely change, so this layer stays cached):
#    g++ and OpenMP for the 3D solver's C++ kernel, and the MUMPS libraries.
#    Debian ships no pkg-config files for the sequential MUMPS libraries: written here.
#    A failed install does not stop the build: the code falls back to numpy / SuperLU.
ARG MUMPS=1
RUN { apt-get update || true; } \
    && { apt-get install -y --no-install-recommends g++ libgomp1 \
         || echo "WARNING: no C++ compiler: the 3D solver will use its numpy fallback (slower)"; } \
    && if [ "$MUMPS" = "1" ]; then \
         { apt-get install -y --no-install-recommends libmumps-seq-dev pkg-config \
           && arch=$(gcc -print-multiarch) && mkdir -p /usr/local/lib/pkgconfig \
           && for t in s d c z; do \
                printf 'Name: %smumps_seq\nDescription: MUMPS\nVersion: 5\nLibs: -L/usr/lib/%s -l%smumps_seq -lmumps_common_seq -lmpiseq_seq\nCflags: -I/usr/include/mumps_seq\n' \
                  $t $arch $t > /usr/local/lib/pkgconfig/${t}mumps_seq.pc || exit 1; \
              done; \
         } || echo "WARNING: MUMPS libraries not installed"; \
       fi; \
    rm -rf /var/lib/apt/lists/*

# 2. Python packages
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN if [ "$MUMPS" = "1" ]; then \
      { PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pip install python-mumps \
        && python -c "import mumps" && echo "MUMPS installed"; \
      } || echo "WARNING: MUMPS could not be installed; the adjoint uses SuperLU (slower, same results)"; \
    fi
ARG PARDISO=0
RUN if [ "$PARDISO" = "1" ]; then pip install pypardiso; fi

# 3. The code
COPY .streamlit .streamlit
COPY app app
COPY src src
COPY data data
COPY validation validation

# 4. Compile the 3D kernel now: at run time the code folder is read-only for the app user
ARG NATIVE_MARCH=native
ENV AEROLAB_NATIVE_MARCH=${NATIVE_MARCH}
RUN python -c "import sys; sys.path.insert(0, 'src'); from aerolab.fv3d import native as N; \
print('3D kernel: compiled' if N.lib() is not None else '3D kernel: ' + str(N._err))"

# 5. Run as an unprivileged user; results live in the /data/cases volume
RUN useradd -m aerolab && mkdir -p /data/cases /tmp/numba && chown -R aerolab /data /tmp/numba
USER aerolab
VOLUME /data/cases
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/AeroLab.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
