"""Pre-run checks, project templates and reports."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerolab.workflow import preflight as PF  # noqa: E402
from aerolab.workflow import project as PR  # noqa: E402
from aerolab.workflow import report as RP  # noqa: E402
from aerolab.workflow import templates as TP  # noqa: E402


def levels(res):
    return {i["check"]: i["level"] for i in res["items"]}


def test_polar_checks_good_setup():
    r = PF.check_polar("naca2412", 3e6, "-2:10:2")
    lv = levels(r)
    assert lv["Geometry"] == "ok" and lv["Reynolds number"] == "ok" and lv["Angles"] == "ok"
    assert lv["Wall resolution (y+)"] == "ok"
    assert r["level"] in ("ok", "info")
    assert len(r["confidence"]) == 7 and all(c["level"] == "green" for c in r["confidence"])
    assert 0 < r["runtime"]["low_s"] < r["runtime"]["expected_s"] < r["runtime"]["high_s"]


def test_polar_checks_catch_bad_setups():
    assert levels(PF.check_polar("no_such_file.dat", 3e6, "0:4:2"))["Geometry"] == "error"
    assert levels(PF.check_polar("naca2412", 500, "0:4:2"))["Reynolds number"] == "error"
    assert levels(PF.check_polar("naca2412", 3e6, "0:4:2", mach=0.97))["Mach number"] == "error"
    assert levels(PF.check_polar("naca2412", 3e6, "0:4:2", mach=0.55, prandtl_glauert=True))["Compressibility"] == "error"
    assert levels(PF.check_polar("naca2412", 3e6, "0:30:2"))["Angles"] == "error"
    assert levels(PF.check_polar("naca2412", 3e6, "4:0:2"))["Angles"] == "error"           # empty range
    assert levels(PF.check_polar("naca2412", 6e4, "0:4:2"))["Reynolds number"] == "warn"
    assert levels(PF.check_polar("naca2412", 3e6, "0:4:2", refine=0.7))["Mesh level"] == "warn"
    assert levels(PF.check_polar("naca0021", 3e6, "0:4:2"))["Thickness"] == "warn"
    assert levels(PF.check_polar("naca2412@flap=25c25", 3e6, "0:4:2"))["Flap"] == "warn"


def test_yplus_fit_matches_runs():
    """the y+ estimate reproduces the converged NACA 2412 Re 3e6 runs (0.63 at 0°, 1.52 at 16°)"""
    yp = []
    for a in (0.0, 16.0):
        r = PF.check_polar("naca2412", 3e6, [a])
        txt = next(i["detail"] for i in r["items"] if i["check"] == "Wall resolution (y+)")
        yp.append(float(txt.split("≈")[1].split()[0]))
    assert yp[0] == pytest.approx(0.63, rel=0.08) and yp[1] == pytest.approx(1.52, rel=0.08)


def test_mesh_quality_deep_check():
    m = PF.mesh_quality("naca0012", 3e6, 4.0, 0.7)
    assert 10000 < m["cells"] < 25000 and m["nonortho_max"] < 70 and m["clamped"] == 0


def test_runtime_scales():
    a = PF.polar_runtime([0, 2, 4, 6], level=1.0, jobs=1)
    b = PF.polar_runtime([0, 2, 4, 6], level=1.4, jobs=1)
    c = PF.polar_runtime([0, 2, 4, 6], level=1.0, jobs=1, mach=0.7)
    assert b["expected_s"] > 1.6 * a["expected_s"] and c["expected_s"] > 2.5 * a["expected_s"]


@pytest.mark.parametrize("tid", TP.ORDER)
def test_templates_build(tid):
    ac, info = TP.build(tid)
    assert info["converged"] and ac.mass > 0 and ac.S > 0 and len(ac.surfaces) == 3
    assert abs(sum(c["mass"] for c in ac.components if c["group"] in ("empty", "battery"))
               + sum(c.get("max", c["mass"]) for c in ac.components if c["group"] not in ("empty", "battery"))
               - ac.mass) < 0.02 * ac.mass + 1.0
    prm = TP.node_params(tid, info)
    assert prm["loads"]["vc"] > 0 and prm["stability"]["speed"] > 0
    if tid == "glider":
        assert prm["performance"]["power"] == 0 and not ac.propulsion
    if TP.TEMPLATES[tid]["req"].get("engine") == "electric" and tid != "glider":
        assert ac.propulsion["engine"]["battery_kwh"] > 0


def test_template_project_runs_and_reports(tmp_path):
    p = TP.create_project("uav", "Survey UAV", root=tmp_path)
    assert p.data["template"]["id"] == "uav"
    pre = PF.check_project(p)
    assert pre["plan"] and pre["runtime_s"] > 0
    secs = [i for v in pre["nodes"].values() for i in v["items"] if i["check"] == "Section data"]
    assert any(i["level"] == "warn" for i in secs)        # UAV Re ~5e5 against cached polars at 2.2e6+
    res = p.run("baseline", targets=["stability", "drag"], log=lambda *_: None)
    assert set(res.values()) == {"ok"}
    p.create_variant("long wing", changes={"aircraft.surfaces.wing.span": 4.0})
    p.run("long wing", targets=["stability", "drag"], log=lambda *_: None)
    for blocks, fmt in ((RP.design_report(p, "baseline", tmp_path / "fig"), "pdf"),
                        (RP.design_report(p, "baseline", tmp_path / "fig"), "docx"),
                        (RP.trade_report(p, ["baseline", "long wing"], tmp_path / "fig"), "pdf"),
                        (RP.trade_report(p, ["baseline", "long wing"], tmp_path / "fig"), "docx")):
        out = RP.write(blocks, tmp_path / f"r_{len(blocks)}.{fmt}", fmt)
        assert out.stat().st_size > 20000
        if fmt == "docx":
            xml = zipfile.ZipFile(out).read("word/document.xml").decode()
            assert "Assumptions, limits and validation" in xml
    titles = [b[2] for b in RP.trade_report(p, ["baseline", "long wing"], tmp_path / "fig") if b[0] == "h"]
    assert "Key findings" in titles and "Results" in titles


def test_project_preflight_errors(tmp_path):
    p = PR.Project.create("Bad", "empty", root=tmp_path)
    p.add_node("baseline", "aircraft", "aircraft", params={"source": "demo", "overrides": {"surfaces.nope.span": 3}})
    p.add_node("baseline", "loads", "loads")
    p.add_node("baseline", "structure", "structure", params={"spars": "0.7,0.2"})
    pre = PF.check_project(p)
    assert pre["level"] == "error"
    lv = {(nid, i["check"]): i["level"] for nid, v in pre["nodes"].items() for i in v["items"]}
    assert lv[("aircraft", "Aircraft changes")] == "error" and lv[("structure", "Spars")] == "error"


def test_polar_cache_is_part_of_the_fingerprint(tmp_path, monkeypatch):
    cache = tmp_path / "polars"
    cache.mkdir()
    monkeypatch.setattr(PR, "POLAR_DIR", cache)
    p = PR.Project.create("Cache", "empty", root=tmp_path)
    p.add_node("baseline", "aircraft", "aircraft")
    p.run("baseline", log=lambda *_: None)
    assert p.status("baseline")["aircraft"]["state"] == "up to date"
    (cache / "naca2412_Re1e+06_M0.10_L1_free.json").write_text(json.dumps([]))
    st = p.status("baseline")["aircraft"]
    assert st["state"] == "stale" and "section polar library" in st["reasons"][0]
