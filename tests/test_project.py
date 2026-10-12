"""Project graph: fingerprints, stale propagation, shared results, variants, comparison."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project as PR  # noqa: E402


def small(tmp_path):
    p = PR.Project.create("Test plane", "empty", root=tmp_path)
    p.add_node("baseline", "aircraft", "aircraft")
    p.add_node("baseline", "drag", "drag")                       # input found by kind
    p.add_node("baseline", "stability", "stability")
    return p


def test_set_path():
    d = {"mass": 1.0, "cg": [1, 2, 3], "surfaces": [{"name": "wing", "span": 10}, {"name": "fin", "span": 1}]}
    PR.set_path(d, "surfaces.fin.span", 2.0)
    PR.set_path(d, "cg.0", 2.5)
    PR.set_path(d, "surfaces.0.span", 11)
    assert d["surfaces"][1]["span"] == 2.0 and d["cg"][0] == 2.5 and d["surfaces"][0]["span"] == 11
    with pytest.raises(KeyError):
        PR.set_path(d, "surfaces.tail.span", 1)


def test_graph_structure(tmp_path):
    p = small(tmp_path)
    assert p.order("baseline")[0] == "aircraft"
    assert set(p.downstream("baseline", "aircraft")) == {"drag", "stability"}
    with pytest.raises(ValueError):
        p.add_node("baseline", "loads2", "loads", inputs={"aircraft": "nope"})
    with pytest.raises(ValueError):
        p.remove_node("baseline", "aircraft")                    # used by drag
    p.nodes("baseline")["aircraft"]["inputs"] = {"x": "drag"}    # make a cycle by hand
    with pytest.raises(ValueError, match="cycle"):
        p.order("baseline")


def test_run_stale_and_recompute(tmp_path):
    p = small(tmp_path)
    assert {s["state"] for s in p.status("baseline").values()} == {"not run"}
    res = p.run("baseline", log=lambda *_: None)
    assert res == {"aircraft": "ok", "drag": "ok", "stability": "ok"}
    st = p.status("baseline")
    assert all(s["state"] == "up to date" for s in st.values())
    assert not any(s["code_changed"] for s in st.values())
    # a downstream-only change: only that node goes stale
    p.set_params("baseline", "drag", speed=70.0)
    st = p.status("baseline")
    assert st["drag"]["state"] == "stale" and st["drag"]["reasons"] == ["speed: 60 → 70"]
    assert st["stability"]["state"] == "up to date"
    assert p.plan("baseline") == ["drag"]
    # an upstream change propagates
    p.set_override("baseline", "aircraft", "mass", 800.0)
    st = p.status("baseline")
    assert st["aircraft"]["reasons"] == ["mass: — → 800"]
    assert st["stability"]["state"] == "stale"
    assert st["stability"]["reasons"] == ["input changed: Aircraft model"]
    assert st["stability"]["last_dir"] is not None             # the old result is still readable
    m = p.metrics("baseline", st)
    assert m["stability"]["current"] is False and m["stability"]["values"]["trim_LD"] > 5
    p.run("baseline", log=lambda *_: None)
    m = p.metrics("baseline")
    assert m["aircraft"]["values"]["mass"] == 800.0 and m["stability"]["current"]
    # undo the change: the earlier results are found again without running (content-addressed)
    p.set_override("baseline", "aircraft", "mass", None)
    p.set_params("baseline", "drag", speed=60.0)
    assert all(s["state"] == "up to date" for s in p.status("baseline").values())


def test_variants_share_results_and_compare(tmp_path):
    p = small(tmp_path)
    p.run("baseline", log=lambda *_: None)
    p.create_variant("same")                                     # identical: nothing to run
    assert all(s["state"] == "up to date" for s in p.status("same").values())
    p.create_variant("heavy", changes={"aircraft.mass": 850.0, "drag.speed": 55.0})
    st = p.status("heavy")
    assert st["aircraft"]["state"] == "stale" and "mass: — → 850" in st["aircraft"]["reasons"]
    assert p.plan("heavy") == ["aircraft", "drag", "stability"]
    p.run("heavy", log=lambda *_: None)
    rows = {(r["node_id"], r["key"]): r for r in p.compare(["baseline", "heavy"])}
    m = rows[("aircraft", "mass")]
    assert m["values"] == {"baseline": 750.0, "heavy": 850.0}
    assert m["change"]["heavy"] == pytest.approx(100 / 750)
    assert rows[("stability", "trim_alpha")]["values"]["heavy"] > rows[("stability", "trim_alpha")]["values"]["baseline"]
    assert ("aircraft", "mass", None, 850.0) in p.diff("baseline", "heavy")
    assert ("drag", "speed", 60.0, 55.0) in p.diff("baseline", "heavy")
    assert "digraph" in p.dot("heavy")
    # reload from disk
    q = PR.Project.load(p.dir)
    assert set(q.variants) == {"baseline", "same", "heavy"}
    q.delete_variant("same")
    assert "same" not in PR.Project.load(p.dir).variants


def test_failure_is_recorded_and_downstream_skipped(tmp_path):
    p = small(tmp_path)
    p.set_params("baseline", "aircraft", source=str(tmp_path / "missing.json"))
    res = p.run("baseline", log=lambda *_: None)
    assert res["aircraft"] == "failed" and res["drag"] == "skipped"
    st = p.status("baseline")
    assert st["aircraft"]["state"] == "failed" and st["aircraft"]["reasons"]


def test_aircraft_file_content_is_tracked(tmp_path):
    import aircraft as A
    f = tmp_path / "plane.json"
    A.demo_aircraft().save(f)
    p = PR.Project.create("File plane", "empty", aircraft_source=str(f), root=tmp_path)
    p.add_node("baseline", "aircraft", "aircraft", params={"source": str(f)})
    p.run("baseline", log=lambda *_: None)
    assert p.status("baseline")["aircraft"]["state"] == "up to date"
    d = json.loads(f.read_text())
    d["mass"] = 760.0
    f.write_text(json.dumps(d))                                  # edited outside the project
    st = p.status("baseline")["aircraft"]
    assert st["state"] == "stale" and st["reasons"] == ["an input file changed"]


def test_skip_propagates_through_the_chain(tmp_path):
    """a failed node skips its children *and* grandchildren (aeroelastic after a skipped structure)"""
    p = PR.Project.create("Chain", "empty", root=tmp_path)
    p.add_node("baseline", "aircraft", "aircraft", params={"source": str(tmp_path / "missing.json")})
    p.add_node("baseline", "loads", "loads")
    p.add_node("baseline", "structure", "structure")
    p.add_node("baseline", "aeroelastic", "aeroelastic")
    res = p.run("baseline", log=lambda *_: None)
    assert res == {"aircraft": "failed", "loads": "skipped", "structure": "skipped", "aeroelastic": "skipped"}
    # a relative project path still works when the runs execute from src/
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        q = PR.Project.load("chain")
        assert q.dir.is_absolute()
    finally:
        os.chdir(cwd)
