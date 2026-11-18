"""Fusion computation graph (12 nodes): 4 observation-condition, 4 sensor-quality, 3 navigation-solution
nodes and 1 fusion node; the v1 graph code (ijcas/rgat/graph.py) is reused with this spec.

Relations: env>sensor, env>nav, sensor>nav:def (protected, from the ontology), sensor>nav:aux,
nav>nav, nav>fusion (protected: every solution reaches the fusion node), self.
Variant without ontology constraints (ablation): the definitional sensor -> solution edges become
ordinary gated "sensor>nav:aux" edges, and solution -> fusion edges are gated too ("nav>fusion:open").
The 'environment' nodes carry sensor-derived observation conditions (diagnostics), not the true
environment (docs/ONTOLOGY_REDESIGN_KO.md 1.1).
"""
from __future__ import annotations

from ..rgat.graph import ENV, SENSOR, GraphSpec, candidate_edges
from . import ontology as O

NAV3 = ("N_gi", "N_vio", "N_lio")
FUSION = ("F_nav",)
NODES = ENV + SENSOR + NAV3 + FUSION
NODE_KIND = {**{n: "env" for n in ENV}, **{n: "sensor" for n in SENSOR}, **{n: "nav" for n in NAV3},
             "F_nav": "fusion"}
KINDS = ("env", "sensor", "nav", "fusion")
SOLUTIONS = ("gnss_ins", "vio", "lio")
RELATIONS = ("env>sensor", "env>nav", "sensor>nav:def", "sensor>nav:aux", "nav>nav", "nav>fusion",
             "nav>fusion:open", "self")


def spec(ontology_constraints: bool = True) -> GraphSpec:
    return GraphSpec(nodes=NODES, node_kind=NODE_KIND, kinds=KINDS, relations=RELATIONS,
                     definitional=O.definitional_pairs() if ontology_constraints else frozenset(),
                     out_nodes=NAV3, candidates=SOLUTIONS,
                     nav_fusion_rel="nav>fusion" if ontology_constraints else "nav>fusion:open")


def edges(ontology_constraints: bool = True):
    return candidate_edges(spec(ontology_constraints))
