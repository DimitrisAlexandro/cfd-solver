# Deployment

Running AeroLab for a team: Streamlit Community Cloud or your own server with Docker.

## Put it online for a team

**Option A: Streamlit Community Cloud (free; aircraft tools and short runs).**
1. Go to https://share.streamlit.io, sign in with GitHub and allow access to this private repository.
2. Click **Create app** and fill in:
   - repository `DimitrisAlexandro/cfd-solver`
   - branch `main`
   - main file `app/AeroLab.py`
   - Advanced settings: Python 3.12
3. The app is private: invite your team by email with **Share**.

`app/requirements.txt` is the dependency list there.

Limits:
- At most 2 CPU cores and 2.7 GB of memory, and one private app per account.
- The app sleeps after 12 h without visitors.
- Results are lost when it restarts, so download what you need. Polar runs work but are slow.

**Option B: your own server (Docker; full speed, results kept).**
```bash
git clone https://github.com/DimitrisAlexandro/cfd-solver.git && cd cfd-solver
cp .env.example .env          # set AEROLAB_PASSWORD (empty = no password)
docker compose up -d --build  # http://<server>:8501
```
- Two services from one image:
  - `aerolab` is the web app.
  - `aerolab-worker` runs the queued analyses.
  
  Updating or restarting the app does not stop runs.
- Results, polars, aircraft and the job queue are kept in the `aerolab-cases` volume across updates.
- Update with `git pull && docker compose up -d --build`.
- For HTTPS, put the server behind a reverse proxy, for example Caddy: `aerolab.example.com { reverse_proxy localhost:8501 }`.
- The coupled solver's MKL library is optional: `docker compose build --build-arg PARDISO=1`.
- MUMPS, the faster adjoint solver, is built into the image. `--build-arg MUMPS=0` leaves it
  out; the adjoint then uses SuperLU and gives the same results, more slowly.
- Worker settings go in `.env`: `AEROLAB_WORKER_SLOTS`, `AEROLAB_JOB_THREADS`, `AEROLAB_JOB_MEM_GB`.
- The 3D solver's C++ kernel is compiled while the image is built, tuned to the CPU of the
  machine that builds it. To build an image that runs on other machines, set
  `AEROLAB_NATIVE_MARCH=x86-64-v2` in `.env`. Without a compiler the 3D solver still runs, on
  its slower numpy fallback; the build log says which one you got.

A fresh install starts with CFD section polars of the NACA 2412 (clean and with 10°/20° flaps)
and the NACA 0012 (`data/wing_polars`), so the aircraft tools use CFD sections from the start.
