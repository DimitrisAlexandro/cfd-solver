"""Job queue and worker for AeroLab's background runs.

The app only queues a job (a JSON record in <cases>/_jobs). A separate
worker process starts at most `slots` jobs at a time, in their own process
groups, records their state and exit code, and handles cancel requests.
The app never needs the worker's process ids, so the two can run in
different containers that share the cases volume (docker-compose:
services aerolab and aerolab-worker).

Job record (<cases>/_jobs/<id>.json):
    id, kind, title, args, out, meta, cwd, started (= queued time; kept for
    the pages), state: queued | running | finished | failed | cancelled,
    run_started, ended, returncode, pid, attempts

Records are written atomically (temp file + rename). The app writes a
record only when it creates it; every later change is the worker's. A
cancel request is a separate file <id>.cancel, created by the app and
acted on by the worker, so the two never write the same file.

    python src/jobqueue.py worker [--slots N] [--threads T] [--mem-gb G]
    python src/jobqueue.py status

Environment: AEROLAB_CASES (cases folder), AEROLAB_WORKER_SLOTS,
AEROLAB_JOB_THREADS, AEROLAB_JOB_MEM_GB (address-space limit per job).
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CASES = Path(os.environ.get("AEROLAB_CASES", ROOT / "cases"))
JOBS = CASES / "_jobs"
HEARTBEAT = JOBS / "worker.heartbeat"
LOCK = JOBS / "worker.lock"
HEARTBEAT_STALE_S = 15.0
MAX_ATTEMPTS = 2                 # an interrupted job is re-queued once

# result file whose presence marks a finished run, per job kind
DONE = {"polar": "polar.json", "wing": "wing.json", "drag_rise": "drag_rise.json",
        "optimize": "optimize.json", "inverse": "inverse.json", "sizing": "sizing.json",
        "aero_db": "database_done.json", "loads": "loads.json", "performance": "performance.json",
        "grid_study": "grid_study.json", "project": "run.json", "adjoint": "adjoint.json",
        "adjoint_design": "design.png"}
ACTIVE = ("queued", "running")


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------
def _path(jid):
    return JOBS / f"{jid}.json"


def read(jid):
    return json.loads(_path(jid).read_text())


def write(job):
    JOBS.mkdir(parents=True, exist_ok=True)
    tmp = JOBS / f".{job['id']}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(job, indent=1))
    os.replace(tmp, _path(job["id"]))


def records():
    out = []
    for f in JOBS.glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def submit(kind, title, args, out_dir, meta=None, cwd=None):
    """queue `python -u <args>` (run in src/ unless cwd); returns the job id"""
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = {"id": jid, "kind": kind, "title": title, "args": [str(a) for a in args],
           "out": str(out_dir), "meta": meta or {}, "cwd": str(cwd or SRC),
           "started": time.time(), "state": "queued", "attempts": 0}
    write(job)
    return jid


def _cancel_flag(jid):
    return JOBS / f"{jid}.cancel"


def request_cancel(jid):
    _cancel_flag(jid).touch()


def cancel_requested(jid):
    return _cancel_flag(jid).exists()


def _legacy_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                              # zombie = finished
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def status(job):
    """queued | running | finished | failed | cancelled"""
    if "state" not in job:                            # record of the old direct launcher
        if job.get("cancelled"):
            return "cancelled"
        if _legacy_alive(job["pid"]):
            return "running"
        return "finished" if (Path(job["out"]) / DONE.get(job["kind"], "_none_")).exists() \
            else "failed"
    if job["state"] in ACTIVE and cancel_requested(job["id"]):
        return "cancelled" if job["state"] == "queued" else "running"
    return job["state"]


def worker_info():
    """{pid, slots, threads, running, time} of the live worker, or None"""
    if not worker_alive():
        return None
    try:
        return json.loads(HEARTBEAT.read_text())
    except (OSError, ValueError):
        return None


def worker_alive():
    try:
        return time.time() - HEARTBEAT.stat().st_mtime < HEARTBEAT_STALE_S
    except OSError:
        return False


# ----------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------
def _limits(mem_gb):
    def pre():
        os.setsid()                                   # own process group: cancel kills it all
        if mem_gb:
            import resource
            b = int(mem_gb * 1024**3)
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return pre


def _finished_ok(job, rc):
    if rc != 0:
        return False
    f = DONE.get(job["kind"])
    return f is None or (Path(job["out"]) / f).exists()


class Worker:
    def __init__(self, slots, threads, mem_gb=None, poll=1.0, log=print):
        self.slots, self.threads, self.mem_gb = slots, threads, mem_gb
        self.poll, self.log = poll, log
        self.procs = {}                               # job id -> Popen
        self.stop = False

    def _env(self):
        t = str(self.threads)
        return {**os.environ, "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg",
                "OMP_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t, "MKL_NUM_THREADS": t,
                "NUMBA_NUM_THREADS": t}

    def recover(self):
        """jobs left 'running' by a worker that died: re-queue (or fail)"""
        for job in records():
            if job.get("state") != "running" or job["id"] in self.procs:
                continue
            pid = job.get("pid")
            if pid and _legacy_alive(pid):            # survived (same host): stop it, run again
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    pass
            if job.get("attempts", 0) < MAX_ATTEMPTS and not cancel_requested(job["id"]):
                job["state"] = "queued"
                job["note"] = "interrupted (worker restarted); queued again"
            else:
                job["state"] = "cancelled" if cancel_requested(job["id"]) else "failed"
                job["ended"] = time.time()
                job.setdefault("note", "interrupted")
            write(job)
            self.log(f"[worker] recovered {job['id']}: {job['state']}")

    def _start(self, job):
        log = JOBS / f"{job['id']}.log"
        with open(log, "a") as f:
            if job.get("attempts"):
                f.write(f"\n--- attempt {job['attempts'] + 1} ---\n")
            p = subprocess.Popen([sys.executable, "-u"] + job["args"], cwd=job.get("cwd") or SRC,
                                 stdout=f, stderr=subprocess.STDOUT, env=self._env(),
                                 preexec_fn=_limits(self.mem_gb))
        self.procs[job["id"]] = p
        job.update(state="running", pid=p.pid, run_started=time.time(),
                   attempts=job.get("attempts", 0) + 1)
        write(job)
        self.log(f"[worker] started {job['id']} ({job['kind']}): {job['title']}")

    def step(self):
        """one poll: reap finished, apply cancels, start queued jobs"""
        for jid, p in list(self.procs.items()):
            try:
                job = read(jid)
            except (OSError, ValueError):
                job = None
            rc = p.poll()
            if rc is None:
                if cancel_requested(jid):
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except OSError:
                        pass
                continue
            del self.procs[jid]
            if job is None:
                continue
            job["returncode"] = rc
            job["ended"] = time.time()
            job["state"] = "cancelled" if cancel_requested(jid) else \
                ("finished" if _finished_ok(job, rc) else "failed")
            write(job)
            self.log(f"[worker] {jid}: {job['state']} (exit {rc})")
        queued = []
        for j in records():
            if j.get("state") != "queued":
                continue
            if cancel_requested(j["id"]):
                j.update(state="cancelled", ended=time.time())
                write(j)
            else:
                queued.append(j)
        free = self.slots - len(self.procs)
        if free > 0:
            queued.sort(key=lambda j: j["started"])
            for job in queued[:free]:
                self._start(job)
        HEARTBEAT.write_text(json.dumps({"pid": os.getpid(), "slots": self.slots,
                                         "threads": self.threads, "running": len(self.procs),
                                         "time": time.time()}))

    def run(self):
        JOBS.mkdir(parents=True, exist_ok=True)
        lock = open(LOCK, "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.log("[worker] another worker is running on this cases folder; exiting")
            return 1
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *a: setattr(self, "stop", True))
        self.log(f"[worker] {self.slots} slot(s), {self.threads} thread(s) per job, "
                 f"cases {CASES}")
        self.recover()
        while not self.stop:
            self.step()
            time.sleep(self.poll)
        # shutting down: leave running jobs to be re-queued by the next worker
        for jid, p in self.procs.items():
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
        self.log("[worker] stopped")
        return 0


def _env_num(name, default, cast=int):
    v = os.environ.get(name, "").strip()      # unset or empty (docker-compose .env): default
    return cast(v) if v else default


def default_slots():
    n = os.cpu_count() or 2
    return _env_num("AEROLAB_WORKER_SLOTS", max(1, n // 2))


def default_threads(slots):
    n = os.cpu_count() or 2
    return _env_num("AEROLAB_JOB_THREADS", max(1, n // max(slots, 1)))


def spawn_background_worker():
    """start a detached worker (local installs: the app calls this when no
    worker heartbeat is seen). It survives restarts of the app."""
    JOBS.mkdir(parents=True, exist_ok=True)
    log = open(JOBS / "worker.log", "a")
    subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker"],
                     cwd=SRC, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "AEROLAB_CASES": str(CASES)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--slots", type=int)
    w.add_argument("--threads", type=int)
    w.add_argument("--mem-gb", type=float, default=_env_num("AEROLAB_JOB_MEM_GB", None, float))
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        slots = a.slots or default_slots()
        return Worker(slots, a.threads or default_threads(slots), a.mem_gb).run()
    jobs = sorted(records(), key=lambda j: j["started"])
    print(f"worker {'alive' if worker_alive() else 'NOT running'}; {len(jobs)} jobs")
    for j in jobs[-20:]:
        print(f"  {j['id']}  {status(j):9s}  {j['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
