"""Ontology-K: navigation-output selection driven by the IJCAS ontology (Notion 03, G4).

The domain knowledge lives in ``ijcas/knowledge/ijcas_nav.ttl`` (sensors, measurement
capabilities, estimators, provisions, environment contexts, tasks, rules) plus the quality
indicators of ``config/selector/thresholds.yaml`` converted to ``:QualityIndicator`` individuals.
At start-up the selector *compiles* the graph with SPARQL (OWL/RDF model and numeric decision
module separated, Notion 05); every tick it evaluates the compiled structure with one generic
procedure, so adding a sensor, an estimator or a task is a change of the graph, not of this code.
Each derived state carries its justification as (rule, ontology IRI, evidence) records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rdflib
from rdflib import Literal, Namespace, RDF

from .common import (LIMITED, RANK, UNKNOWN, UNUSABLE, USABLE, Decision, best,
                     indicator_state, load_config, worst)

NAV = Namespace("https://incsl.sejong.ac.kr/ijcas/nav#")
TTL = Path(__file__).resolve().parents[1] / "knowledge/ijcas_nav.ttl"


def build_graph(cfg: dict, ttl: str | Path = TTL, extra=()) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(ttl))
    for e in extra:
        g.parse(str(e))
    g.bind("", NAV)
    for ind in cfg["indicators"]:
        node = NAV[f"ind_{ind['id']}"]
        g.add((node, RDF.type, NAV.QualityIndicator))
        g.add((node, NAV.signal, Literal(ind["signal"])))
        g.add((node, NAV.validForS, Literal(float(ind.get("valid_for_s", 1.0)))))
        g.add((node, NAV.missingPolicy, Literal(ind.get("missing", "unknown"))))
        for target in ind["assesses"]:
            g.add((node, NAV.assesses, NAV[target]))
        for key, prop in (("invalid_below", NAV.invalidBelow), ("degraded_below", NAV.degradedBelow),
                          ("invalid_above", NAV.invalidAbove), ("degraded_above", NAV.degradedAbove)):
            if key in ind:
                g.add((node, prop, Literal(float(ind[key]))))
    return g


Q_INDICATORS = """
SELECT ?ind ?target ?signal ?valid ?missing ?ib ?db ?ia ?da WHERE {
  ?ind a :QualityIndicator ; :assesses ?target ; :signal ?signal ; :validForS ?valid ;
       :missingPolicy ?missing .
  OPTIONAL { ?ind :invalidBelow ?ib } OPTIONAL { ?ind :degradedBelow ?db }
  OPTIONAL { ?ind :invalidAbove ?ia } OPTIONAL { ?ind :degradedAbove ?da } }"""
Q_ESTIMATORS = """
SELECT ?e ?id WHERE { ?e :candidateId ?id . }"""
Q_ESSENTIAL = """
SELECT ?e ?m WHERE { ?e :requiresEssential ?m . }"""
Q_PROVISIONS = """
SELECT ?p ?e ?c ?align ?dr WHERE {
  ?p a :Provision ; :ofEstimator ?e ; :ofCapability ?c .
  OPTIONAL { ?p :requiresAlignment ?align } OPTIONAL { ?p :deadReckonLimitS ?dr } }"""
Q_PROV_LIST = """
SELECT ?p ?rel ?m WHERE { ?p ?rel ?m . VALUES ?rel { :validUnder :alternativelyUnder :limitedWithout } }"""
Q_TASKS = """
SELECT ?task ?cap ?age ?alarm WHERE {
  ?task a :TaskRequirement ; :requiresCapability ?cap ; :maxOutputAgeS ?age ; :alarmLimitM ?alarm . }"""
Q_SHARED = """
SELECT ?sensor ?m ?e WHERE { ?sensor :sharedBy ?e ; :providesMeasurement ?m . }"""


def _local(iri) -> str:
    return str(iri).split("#")[-1]


@dataclass
class Compiled:
    indicators: dict = field(default_factory=dict)     # target -> [indicator dict]
    candidates: dict = field(default_factory=dict)     # estimator local name -> candidate id
    essential: dict = field(default_factory=dict)      # estimator -> [measurement]
    provisions: dict = field(default_factory=dict)     # (estimator, capability) -> dict
    tasks: dict = field(default_factory=dict)          # task -> {caps, max_age, alarm}
    shared: dict = field(default_factory=dict)         # measurement -> [estimators] (R10)


def compile_graph(g: rdflib.Graph) -> Compiled:
    c = Compiled()
    ns = {"": NAV}
    for r in g.query(Q_INDICATORS, initNs=ns):
        ind = {"id": _local(r.ind)[4:], "signal": str(r.signal), "valid_for_s": float(r.valid),
               "missing": str(r.missing)}
        for key, val in (("invalid_below", r.ib), ("degraded_below", r.db),
                         ("invalid_above", r.ia), ("degraded_above", r.da)):
            if val is not None:
                ind[key] = float(val)
        c.indicators.setdefault(_local(r.target), []).append(ind)
    for r in g.query(Q_ESTIMATORS, initNs=ns):
        c.candidates[_local(r.e)] = str(r.id)
    for r in g.query(Q_ESSENTIAL, initNs=ns):
        c.essential.setdefault(_local(r.e), []).append(_local(r.m))
    for r in g.query(Q_PROVISIONS, initNs=ns):
        c.provisions[(_local(r.e), _local(r.c))] = {
            "iri": _local(r.p), "align": bool(r.align.toPython()) if r.align is not None else False,
            "dr_limit": float(r.dr) if r.dr is not None else None,
            "validUnder": [], "alternativelyUnder": [], "limitedWithout": []}
    by_iri = {v["iri"]: v for v in c.provisions.values()}
    for r in g.query(Q_PROV_LIST, initNs=ns):
        p = by_iri.get(_local(r.p))
        if p is not None:
            p[_local(r.rel)].append(_local(r.m))
    for r in g.query(Q_TASKS, initNs=ns):
        t = c.tasks.setdefault(_local(r.task), {"caps": [], "max_age": float(r.age),
                                                "alarm": float(r.alarm)})
        t["caps"].append(_local(r.cap))
    for r in g.query(Q_SHARED, initNs=ns):
        c.shared.setdefault(_local(r.m), []).append(_local(r.e))
    for v in c.essential.values():
        v.sort()
    for t in c.tasks.values():
        t["caps"].sort()
    return c


class OntologyReasoner:
    """Per-tick candidate states for a task from facts + compiled ontology (R01-R05, R08, R10)."""

    def __init__(self, cfg: dict | None = None, ttl: str | Path = TTL, extra=()):
        self.cfg = cfg or load_config()
        self.graph = build_graph(self.cfg, ttl, extra)
        self.c = compile_graph(self.graph)
        self.last_usable: dict = {}            # (estimator, capability) -> last time Usable

    def node_states(self, sig: dict, t: float) -> tuple[dict, dict]:
        states, reasons = {}, {}
        for target, inds in self.c.indicators.items():
            res = [indicator_state(i, sig, t) for i in sorted(inds, key=lambda d: d["id"])]
            states[target] = worst(s for s, _ in res)
            reasons[target] = [r for s, r in res if s != USABLE]
        return states, reasons

    def evaluate(self, task: str, sig: dict, t: float, outputs: dict, alignment: dict,
                 stale_alignment_s: float) -> tuple[dict, dict, dict]:
        node, node_reasons = self.node_states(sig, t)
        req = self.c.tasks[task]
        states, reasons = {}, {}
        for est, cid in sorted(self.c.candidates.items()):
            why = []
            st = USABLE
            # R01: essential inputs and estimator-level indicators (initialisation / tracking)
            for m in self.c.essential.get(est, []):
                ms = node.get(m, UNKNOWN)
                if ms in (UNUSABLE, UNKNOWN):
                    why.append(f"R01:{m}={ms}:" + ",".join(node_reasons.get(m, [])))
                st = worst([st, ms if ms != LIMITED else USABLE])
            if est in node:
                if node[est] != USABLE:
                    why.append(f"R01:{est}={node[est]}:" + ",".join(node_reasons.get(est, [])))
                st = worst([st, node[est]])
            out = outputs.get(cid)
            if cid != "ins_dr":
                if out is None:
                    why.append("R01:not_initialised")
                    st = UNUSABLE
                elif t - out.stamp > req["max_age"]:
                    why.append(f"R01:output_age={t - out.stamp:.2f}s>{req['max_age']}")
                    st = UNUSABLE
            # R05 (+R02 AND/OR provisions, R03/R04 through the measurements, R08 alignment)
            cap_states = []
            for cap in req["caps"]:
                p = self.c.provisions.get((est, cap))
                if p is None:
                    why.append(f"R05:no_provision:{cap}")
                    cap_states.append(UNUSABLE)
                    continue
                s = USABLE
                if p["validUnder"]:
                    s = worst(node.get(m, UNKNOWN) for m in p["validUnder"])
                if p["alternativelyUnder"]:
                    s = worst([s, best(node.get(m, UNKNOWN) for m in p["alternativelyUnder"])])
                if any(node.get(m, UNKNOWN) != USABLE for m in p["limitedWithout"]):
                    s = worst([s, LIMITED])
                key = (est, cap)
                if s == USABLE:
                    self.last_usable[key] = t
                if p["dr_limit"] is not None:
                    if p["dr_limit"] == 0.0 and s == USABLE:
                        s = LIMITED                                     # inherently limited (DR)
                        why.append(f"R09:{p['iri']}:inherently_limited")
                    elif s in (UNUSABLE, UNKNOWN) and t - self.last_usable.get(key, -1e9) <= p["dr_limit"]:
                        why.append(f"R02:{p['iri']}:dead_reckoning<={p['dr_limit']}s")
                        s = LIMITED
                if p["align"]:
                    al = alignment.get(cid)
                    if al is None or not al.valid:
                        why.append(f"R08:{p['iri']}:no_alignment")
                        s = UNUSABLE
                    elif al.age(t) > stale_alignment_s:
                        why.append(f"R08:{p['iri']}:alignment_age={al.age(t):.0f}s")
                        s = worst([s, LIMITED])
                    elif al.age(t) > 0.5:
                        s = worst([s, LIMITED])                          # frozen: drift grows
                if s != USABLE:
                    why.append(f"R05:{cap}={s}")
                cap_states.append(s)
            st = worst([st] + cap_states)
            # R10: common-cause dependency of the shared IMU
            for m, ests in self.c.shared.items():
                if est in ests and node.get(m, UNKNOWN) != USABLE:
                    why.append(f"R10:shared_{m}={node.get(m, UNKNOWN)}")
                    st = worst([st, node.get(m, UNKNOWN) if node.get(m) != LIMITED else LIMITED])
            states[cid], reasons[cid] = st, why
        return states, reasons, node
