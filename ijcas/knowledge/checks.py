"""Knowledge integrity checks over the ontology graph (G4, Notion 03 "누락·충돌").

Each check is a SPARQL query returning offending nodes; an empty result means the check holds.
Flat-K has no equivalent: the same omissions there only show up as silent behaviour at run time
(e.g. a measurement nobody assesses is always 'Unknown', so every candidate needing it is
silently excluded - which is exactly what happened to MagHeading before it had an indicator).
"""
from __future__ import annotations

from rdflib import Graph

PREFIX = "PREFIX : <https://incsl.sejong.ac.kr/ijcas/nav#>\n"
CHECKS = {
    "task_capability_without_provider": """
        SELECT DISTINCT ?task ?cap WHERE {
          ?task a :TaskRequirement ; :requiresCapability ?cap .
          FILTER NOT EXISTS { ?p a :Provision ; :ofCapability ?cap . } }""",
    "measurement_without_indicator": """
        SELECT DISTINCT ?m WHERE {
          { ?x :requiresEssential ?m } UNION { ?p :validUnder ?m } UNION
          { ?p :alternativelyUnder ?m } UNION { ?p :limitedWithout ?m }
          FILTER NOT EXISTS { ?i a :QualityIndicator ; :assesses ?m . } }""",
    "measurement_without_sensor": """
        SELECT DISTINCT ?m WHERE { ?m a :MeasurementCapability .
          FILTER NOT EXISTS { ?s :providesMeasurement ?m . } }""",
    "and_or_conflict": """
        SELECT ?p ?m WHERE { ?p :validUnder ?m ; :alternativelyUnder ?m . }""",
    "context_degrades_nothing": """
        SELECT ?c WHERE { ?c a :EnvironmentContext .
          FILTER NOT EXISTS { ?c :degrades ?m . } }""",
    "rule_without_implementation": """
        SELECT ?r WHERE { ?r a :Rule . FILTER NOT EXISTS { ?r :implementsRule ?impl . } }""",
    "candidate_cannot_serve_any_task": """
        SELECT ?e WHERE { ?e :candidateId ?id .
          FILTER NOT EXISTS {
            ?task a :TaskRequirement .
            FILTER NOT EXISTS { ?task :requiresCapability ?cap .
                                FILTER NOT EXISTS { ?p :ofEstimator ?e ; :ofCapability ?cap . } } } }""",
    "estimator_without_essential_input": """
        SELECT ?e WHERE { ?e :candidateId ?id . FILTER NOT EXISTS { ?e :requiresEssential ?m . } }""",
}


def run_checks(g: Graph) -> dict:
    out = {}
    for name, q in CHECKS.items():
        rows = [tuple(str(v).split("#")[-1] for v in r) for r in g.query(PREFIX + q)]
        out[name] = rows
    return out
