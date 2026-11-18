"""Validate docs/ontology_v2 examples: parse, SHACL conformance, negative cases, extension regression."""
import sys
from pathlib import Path

import rdflib
from pyshacl import validate

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
BASE = ["core.ttl", "nav.ttl", "env.ttl", "profile_ijcas.ttl", "rig.ttl", "example_run.ttl", "eval_scene.ttl"]


def graph(files, extra_ttl=""):
    g = rdflib.Graph()
    for f in files:
        g.parse(D / f)
    if extra_ttl:
        g.parse(data=extra_ttl, format="turtle")
    return g


shapes = rdflib.Graph().parse(D / "shapes.ttl")


def run(name, g, expect):
    ok, rep_g, rep = validate(g, shacl_graph=shapes, inference="none", allow_warnings=True)
    msgs = sorted({str(o) for o in rep_g.objects(None, rdflib.URIRef("http://www.w3.org/ns/shacl#resultMessage"))})
    status = "PASS" if ok == expect else "FAIL"
    print(f"[{status}] {name}: conforms={ok} (expected {expect}); {len(msgs)} message(s)")
    for m in msgs:
        print("        -", m[:160])
    return ok == expect


P = """@prefix run: <https://incsl.sejong.ac.kr/ijcas/run/E4_201#> . @prefix rig: <https://incsl.sejong.ac.kr/ijcas/rig#> .
@prefix ijcas: <https://incsl.sejong.ac.kr/ijcas/profile#> . @prefix core: <https://incsl.sejong.ac.kr/onto/core#> .
@prefix nav: <https://incsl.sejong.ac.kr/onto/nav#> . @prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix prov: <http://www.w3.org/ns/prov#> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""
results = []
for f in sorted(D.glob("*.ttl")):
    n = len(rdflib.Graph().parse(f))
    print(f"parsed {f.name}: {n} triples")
results.append(run("T1 base example", graph(BASE), True))
results.append(run("T2 observation without value or status", graph(BASE, P + """
run:obs_bad a sosa:Observation ; sosa:madeBySensor rig:context_monitor ; sosa:usedProcedure rig:proc_fast_t20 ;
  sosa:observedProperty nav:FastCornerCount ; sosa:hasFeatureOfInterest run:frame_cam0_1264 ;
  sosa:resultTime "2026-11-20T09:01:03.17Z"^^xsd:dateTime ."""), False))
results.append(run("T3 decision used a report received after the decision", graph(BASE, P + """
run:obs_late a sosa:Observation ; sosa:madeBySensor rig:context_monitor ; sosa:usedProcedure rig:proc_count_ov_points ;
  sosa:observedProperty nav:TrackedFeatureCount ; sosa:hasFeatureOfInterest run:ovpts_0631 ; sosa:hasSimpleResult 30 ;
  sosa:resultTime "2026-11-20T09:01:03.30Z"^^xsd:dateTime ; core:receptionTime "2026-11-20T09:01:03.31Z"^^xsd:dateTime .
run:elig_vio_6320 prov:used run:obs_late ."""), False))
results.append(run("T4 eligibility used the zone hypothesis", graph(BASE, P + """
run:elig_lio_6320 prov:used run:hyp_6320 ."""), False))
results.append(run("T5 value and NotComputable together", graph(BASE, P + """
run:obs_eig_0701 sosa:hasSimpleResult 0.0 ."""), False))
results.append(run("T6 + UWB extension (new sensor, extra property)", graph(BASE + ["ext_uwb.ttl"]), True))
pump = graph(["core.ttl", "ext_pump.ttl"])
results.append(run("T7 pump domain with core only", pump, True))
results.append(run("T8 pump check without decision time (subclass targeting)", graph(["core.ttl", "ext_pump.ttl"], """
@prefix plant: <https://example.org/plant/data#> . @prefix pump: <https://example.org/plant/pump#> .
@prefix core: <https://incsl.sejong.ac.kr/onto/core#> . @prefix prov: <http://www.w3.org/ns/prov#> .
plant:chk_89 a pump:InspectionCheck ; core:appliedPolicy plant:pol_inspect ."""), False))
nav_terms = {str(t) for t in pump.all_nodes() if "onto/nav#" in str(t)}
print("        pump graph nav terms:", sorted(nav_terms) or "none")

# competency queries: unchanged answers after importing extension modules
Q = {
 "estimators_requiring_imu": """PREFIX nav: <https://incsl.sejong.ac.kr/onto/nav#> PREFIX ssn: <http://www.w3.org/ns/ssn/>
   SELECT ?e WHERE { ?e ssn:implements ?p . ?p nav:requiresInput nav:ImuSamples } ORDER BY ?e""",
 "evidence_of_selection": """PREFIX prov: <http://www.w3.org/ns/prov#> PREFIX nav: <https://incsl.sejong.ac.kr/onto/nav#>
   SELECT ?r ?obs WHERE { ?s a nav:SelectionResult ; prov:wasGeneratedBy ?a . ?a prov:used ?r . ?r prov:wasDerivedFrom ?obs } ORDER BY ?r ?obs""",
 "inputs_of_ssn_hasInput": """PREFIX ssn: <http://www.w3.org/ns/ssn/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   SELECT ?p ?i WHERE { ?prop rdfs:subPropertyOf* ssn:hasInput . ?p ?prop ?i } ORDER BY ?p ?i""",
}
g0, g1 = graph(BASE), graph(BASE + ["ext_uwb.ttl", "ext_pump.ttl"])
for k, q in Q.items():
    a0 = [tuple(map(str, r)) for r in g0.query(q)]
    a1 = [tuple(map(str, r)) for r in g1.query(q)]
    same = a0 == a1
    results.append(same)
    print(f"[{'PASS' if same else 'FAIL'}] Q {k}: {len(a0)} rows before, {len(a1)} after, identical={same}")
print("ALL PASS" if all(results) else "SOME FAILED")
