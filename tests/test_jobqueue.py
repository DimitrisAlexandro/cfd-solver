"""The job queue and worker (src/aerolab/workflow/jobqueue.py)."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.workflow import jobqueue as JQ  # noqa: E402


@pytest.fixture
def q(tmp_path, monkeypatch):
    jobs = tmp_path / "_jobs"
    jobs.mkdir()
    monkeypatch.setattr(JQ, "JOBS", jobs)
    monkeypatch.setattr(JQ, "HEARTBEAT", jobs / "worker.heartbeat")
    monkeypatch.setattr(JQ, "LOCK", jobs / "worker.lock")
    return tmp_path


def script(tmp, body):
    f = tmp / f"s{time.time_ns()}.py"
    f.write_text(body)
    return str(f)


def drain(w, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        w.step()
        if not w.procs and not any(j.get("state") == "queued" for j in JQ.records()):
            return
        time.sleep(0.05)
    raise AssertionError("worker did not finish")


def test_finished_failed_and_done_file(q):
    out = q / "out"
    out.mkdir()
    ok = JQ.submit("test", "ok", [script(q, "print('hi')")], out)
    bad = JQ.submit("test", "bad", [script(q, "raise SystemExit(3)")], out)
    nodone = JQ.submit("polar", "exit 0 but no polar.json", [script(q, "pass")], out)
    assert JQ.status(JQ.read(ok)) == "queued"
    drain(JQ.Worker(slots=2, threads=1, log=lambda *a: None))
    assert JQ.status(JQ.read(ok)) == "finished"
    assert "hi" in (JQ.JOBS / f"{ok}.log").read_text()
    r = JQ.read(bad)
    assert JQ.status(r) == "failed" and r["returncode"] == 3
    assert JQ.status(JQ.read(nodone)) == "failed"
    assert JQ.HEARTBEAT.exists()


def test_slots_limit_concurrency(q):
    ids = [JQ.submit("test", f"j{i}", [script(q, "import time; time.sleep(0.6)")], q)
           for i in range(3)]
    w = JQ.Worker(slots=2, threads=1, log=lambda *a: None)
    w.step()
    states = [JQ.read(i)["state"] for i in ids]
    assert states.count("running") == 2 and states.count("queued") == 1
    drain(w)
    assert all(JQ.read(i)["state"] == "finished" for i in ids)


def test_cancel_queued_and_running(q):
    w = JQ.Worker(slots=1, threads=1, log=lambda *a: None)
    long = JQ.submit("test", "long", [script(q, "import time; time.sleep(30)")], q)
    waiting = JQ.submit("test", "waiting", [script(q, "pass")], q)
    w.step()
    assert JQ.read(long)["state"] == "running"
    JQ.request_cancel(waiting)
    assert JQ.status(JQ.read(waiting)) == "cancelled"
    JQ.request_cancel(long)
    t0 = time.time()
    drain(w)
    assert time.time() - t0 < 10
    assert JQ.read(long)["state"] == "cancelled" and JQ.read(waiting)["state"] == "cancelled"


def test_interrupted_job_is_requeued(q):
    jid = JQ.submit("test", "x", [script(q, "pass")], q)
    rec = JQ.read(jid)
    rec.update(state="running", pid=999999999, attempts=1)      # its worker died
    JQ.write(rec)
    w = JQ.Worker(slots=1, threads=1, log=lambda *a: None)
    w.recover()
    assert JQ.read(jid)["state"] == "queued"
    drain(w)
    assert JQ.read(jid)["state"] == "finished" and JQ.read(jid)["attempts"] == 2
    rec = JQ.read(jid)
    rec.update(state="running", pid=999999999)                  # second interruption: give up
    JQ.write(rec)
    w.recover()
    assert JQ.read(jid)["state"] == "failed"


def test_thread_limits_reach_the_job(q):
    jid = JQ.submit("test", "env", [script(q, "import os; print('T', os.environ['OMP_NUM_THREADS'], "
                                               "os.environ['NUMBA_NUM_THREADS'])")], q)
    drain(JQ.Worker(slots=1, threads=3, log=lambda *a: None))
    assert "T 3 3" in (JQ.JOBS / f"{jid}.log").read_text()


def test_legacy_record(q):
    rec = {"id": "old", "kind": "polar", "title": "old", "args": [], "out": str(q),
           "pid": 999999999, "started": 0}
    assert JQ.status(rec) == "failed"
    (q / "polar.json").write_text("{}")
    assert JQ.status(rec) == "finished"
