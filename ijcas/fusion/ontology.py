"""What the fusion takes from the ontology (ijcas/knowledge/ijcas_nav.ttl + ijcas_fusion.ttl).

1. Protected graph edges: sensor -> solution for every :requiresEssential / :usesAiding capability of
   an estimator whose solution fus:contributesToFusion, and solution -> fusion node.
2. Shared measurement sources: two contributing estimators that require or use the same capability
   (all three use :Inertial).  Their errors may be correlated by an unknown amount, so the fusion
   process' fus:requiresMethodUnderSharedSource ("covariance_intersection") is enforced; an
   independent-information fusion is allowed only as an explicitly named comparison.
3. Alignment requirement: fus:requiresAlignment -> the solution is eligible only with a valid causal
   alignment (already part of the eligibility policy, ijcas/rgat/features.py CandidateBank).
The actual error correlation is not known to the ontology and is never assumed.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import rdflib

KDIR = Path(__file__).resolve().parents[1] / "knowledge"
TTLS = (KDIR / "ijcas_nav.ttl", KDIR / "ijcas_fusion.ttl")
PFX = """PREFIX : <https://incsl.sejong.ac.kr/ijcas/nav#>
PREFIX fus: <https://incsl.sejong.ac.kr/ijcas/fusion#>
"""
# capability -> sensor-quality node (same mapping as the v1 graph, tests/test_rgat.py)
CAP_NODE = {"Inertial": "S_imu", "MagHeading": "S_imu", "BaroHeight": "S_imu",
            "GnssPosition": "S_gnss", "GnssVelocity": "S_gnss", "VisualFeatures": "S_cam",
            "LidarStructure": "S_lidar"}
SOLUTION_NODE = {"gnss_ins": "N_gi", "vio": "N_vio", "lio": "N_lio"}


def _l(x) -> str:
    return str(x).split("#")[-1]


@lru_cache(maxsize=None)
def graph(ttls: tuple = TTLS) -> rdflib.Graph:
    g = rdflib.Graph()
    for t in ttls:
        g.parse(str(t))
    return g


def contributing(g: rdflib.Graph | None = None) -> dict:
    """{solution id: estimator local name} for the solutions that contribute to the fusion."""
    g = g or graph()
    q = PFX + """SELECT ?id ?e WHERE { ?s fus:contributesToFusion ?f ; fus:solutionId ?id .
                                         ?e fus:producesEstimate ?s . }"""
    return {str(r.id): _l(r.e) for r in g.query(q)}


def capabilities(g: rdflib.Graph | None = None) -> dict:
    """{estimator: {capability: 'essential' | 'aiding'}}"""
    g = g or graph()
    out = {}
    for prop, kind in ((":requiresEssential", "essential"), (":usesAiding", "aiding")):
        for r in g.query(PFX + f"SELECT ?e ?m WHERE {{ ?e {prop} ?m . }}"):
            out.setdefault(_l(r.e), {})[_l(r.m)] = kind
    return out


def definitional_pairs(g: rdflib.Graph | None = None) -> frozenset:
    """(sensor node, solution node) pairs of the protected sensor -> solution edges."""
    g = g or graph()
    caps = capabilities(g)
    out = set()
    for sol, est in contributing(g).items():
        for m in caps.get(est, {}):
            if m in CAP_NODE:
                out.add((CAP_NODE[m], SOLUTION_NODE[sol]))
    return frozenset(out)


def shared_sources(g: rdflib.Graph | None = None) -> dict:
    """{(sol_a, sol_b): sorted shared capabilities} for contributing solutions."""
    g = g or graph()
    caps = capabilities(g)
    con = contributing(g)
    out = {}
    for a, b in combinations(sorted(con), 2):
        common = sorted(set(caps.get(con[a], {})) & set(caps.get(con[b], {})))
        if common:
            out[(a, b)] = common
    return out


def required_method(g: rdflib.Graph | None = None) -> str | None:
    g = g or graph()
    if not shared_sources(g):
        return None
    r = list(g.query(PFX + "SELECT ?m WHERE { ?f a fus:FusionProcess ; fus:requiresMethodUnderSharedSource ?m . }"))
    return str(r[0].m) if r else None


def requires_alignment(g: rdflib.Graph | None = None) -> dict:
    g = g or graph()
    q = PFX + "SELECT ?id ?a WHERE { ?s fus:solutionId ?id ; fus:requiresAlignment ?a . }"
    return {str(r.id): bool(r.a.toPython()) for r in g.query(q)}


def check_method(method: str, comparison: bool = False):
    """Refuse a fusion method that ignores the shared sources unless it is a named comparison."""
    req = required_method()
    if req and method != req and not comparison:
        raise ValueError(f"solutions share {shared_sources()}; the ontology requires {req}, got {method}")
