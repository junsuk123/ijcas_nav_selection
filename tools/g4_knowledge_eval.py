"""G4 / H3: knowledge-management comparison of Ontology-K and Flat-K on two change tasks.

C1 new mission HeightHold (Height + Velocity); C2 new candidate GNSS-only (position/velocity, no
attitude).  Reported: edits needed (ontology triples vs flat code lines; selector code edits),
behavioural equivalence after the change (fuzz over all three tasks), and whether seeded
omissions are caught before run time (ontology integrity checks) or not (flat).
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from ijcas.knowledge.checks import run_checks  # noqa: E402
from ijcas.selector.common import load_config, signals  # noqa: E402
from ijcas.selector.facts import Output  # noqa: E402
from ijcas.selector.flat_ext import FlatRulesExt  # noqa: E402
from ijcas.selector.ontology import OntologyReasoner, build_graph  # noqa: E402
from test_selector_equivalence import FakeAlign, random_snapshot  # noqa: E402

CH = ROOT / "ijcas/knowledge/changes"


def code_lines(path):
    """Logical source lines excluding docstrings, comments and blank lines."""
    import ast
    src = Path(path).read_text()
    doc_lines = set()
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and \
                isinstance(body[0].value.value, str):
            doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(1 for i, line in enumerate(src.splitlines(), 1)
               if i not in doc_lines and line.strip() and not line.strip().startswith("#"))


def main():
    cfg = load_config()
    g0 = build_graph(cfg)
    g1 = build_graph(cfg, extra=[CH / "c1_heighthold.ttl", CH / "c2_gnss_only.ttl"])
    res = {"ontology_triples_base": len(g0), "ontology_triples_added": len(g1) - len(g0),
           "ontology_selector_code_lines_changed": 0,
           "flat_code_lines_added": code_lines(ROOT / "ijcas/selector/flat_ext.py"),
           "integrity_after_change": {k: v for k, v in run_checks(g1).items() if v}}
    onto = OntologyReasoner(cfg, extra=[CH / "c1_heighthold.ttl", CH / "c2_gnss_only.ttl"])
    flat = FlatRulesExt(cfg)
    rng = np.random.default_rng(7)
    mism, n = 0, 0
    t = 0.0
    for k in range(3000):
        t += 0.1
        snap = random_snapshot(rng, t)
        if rng.random() < 0.85:
            snap.outputs["gnss_only"] = Output(t - float(rng.choice([0.05, 0.25, 1.0])),
                                               np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))
        sig = signals(snap)
        al = {c: FakeAlign(bool(rng.random() < 0.7), t - rng.choice([0.0, 2.0, 40.0]))
              for c in ("openvins", "glim", "ins_dr")}
        for task in ("GlobalWaypoint", "LocalFlight", "HeightHold"):
            so, _, _ = onto.evaluate(task, sig, t, snap.outputs, al, 30.0)
            sf, _, _ = flat.evaluate(task, sig, t, snap.outputs, al, 30.0)
            n += 1
            mism += so != sf
    res["equivalence_after_change"] = {"ticks": n, "mismatches": mism,
        "history": "first flat implementation of C2 mapped GnssPosition=Unknown to Unusable for the "
                   "new candidate (292/9000 mismatching ticks, HeightHold); found by this fuzz "
                   "comparison and fixed in flat_ext.py. The ontology change needed no code."}
    # seeded omissions
    gx = build_graph(cfg, extra=[CH / "c1_heighthold.ttl"])
    gx.parse(data="""@prefix : <https://incsl.sejong.ac.kr/ijcas/nav#> .
        :GnssOnly a :Baseline ; :candidateId "gnss_only" ; :requiresEssential :GnssPosition .
        :gnss_global a :Provision ; :ofEstimator :GnssOnly ; :ofCapability :GlobalPosition ; :validUnder :GnssPosition .
        """, format="turtle")                       # forgot the velocity / height provisions
    gy = build_graph(cfg)
    gy.parse(data="""@prefix : <https://incsl.sejong.ac.kr/ijcas/nav#> .
        :Radar a :Sensor ; :providesMeasurement :RadarRange . :RadarRange a :MeasurementCapability .
        :RadarOdom a :LocalEstimator ; :candidateId "radar" ; :requiresEssential :Inertial , :RadarRange .
        """, format="turtle")                       # new sensor without any quality indicator
    res["seeded_omissions"] = {
        "C2_without_velocity_provision": {k: v for k, v in run_checks(gx).items() if v},
        "new_sensor_without_indicator": {k: v for k, v in run_checks(gy).items() if v},
        "flat_equivalent": "no static check: the same omissions compile and run, the candidate is "
                           "silently never usable (found only by inspecting behaviour)"}
    out = ROOT / "results/g4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "knowledge_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
