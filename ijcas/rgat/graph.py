"""Ontology-constrained candidate graph (design section 2).

Three computational node kinds (environment/motion condition, sensor quality, navigation
algorithm).  The drone-state output node is not a computational node: it only carries the state of
the selected algorithm (design 2.1), so it has no features and sends no messages.

Candidate relations are every directed pair allowed by node kind and direction:
  env -> sensor, env -> nav, sensor -> nav, nav -> nav  (+ an always-on self loop per node).
Definitional relations taken from the ontology (``:requiresEssential`` / ``:usesAiding`` in
ijcas/knowledge/ijcas_nav.ttl: every estimator requires the IMU, OpenVINS the camera, FAST-LIO2 the
LiDAR, PX4 EKF2 is aided by GNSS) are *protected*: always active and never pruned, because not
using them would assert something false about the algorithm, not just save computation.
All other candidate edges carry a learned gate and may be pruned offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENV = ("E_gnss", "E_vis", "E_geo", "E_mot")          # GNSS reception, visual, geometry, motion
SENSOR = ("S_gnss", "S_imu", "S_cam", "S_lidar")       # sensor quality
NAV = ("N_gi", "N_vio", "N_lio", "N_ins")              # GNSS/INS (PX4 EKF2), VIO, LIO, INS
NODES = ENV + SENSOR + NAV
NODE_KIND = {**{n: "env" for n in ENV}, **{n: "sensor" for n in SENSOR}, **{n: "nav" for n in NAV}}
KINDS = ("env", "sensor", "nav")
# candidate algorithm order used everywhere (score vector, costs, validity)
CANDIDATES = ("gnss_ins", "vio", "lio", "ins")
NAV_OF = dict(zip(CANDIDATES, NAV))

# relation types (R-GAT / R-GCN weights are per relation)
RELATIONS = ("env>sensor", "env>nav", "sensor>nav:def", "sensor>nav:aux", "nav>nav", "self")
GATED_RELATIONS = ("env>sensor", "env>nav", "sensor>nav:aux", "nav>nav")

# definitional relations (ontology): sensor -> algorithm
DEFINITIONAL = {("S_imu", "N_gi"), ("S_imu", "N_vio"), ("S_imu", "N_lio"), ("S_imu", "N_ins"),
                ("S_cam", "N_vio"), ("S_lidar", "N_lio"), ("S_gnss", "N_gi")}
# ontology individuals behind the definitional edges (checked against the TTL in tests)
ONTOLOGY_NAME = {"N_gi": "PX4_EKF2", "N_vio": "OpenVINS", "N_lio": "FAST_LIO", "N_ins": "INS_DR",
                 "S_imu": "Inertial", "S_cam": "VisualFeatures", "S_lidar": "LidarStructure",
                 "S_gnss": "GnssPosition"}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: str

    @property
    def protected(self) -> bool:
        return self.rel in ("sensor>nav:def", "self")

    @property
    def name(self) -> str:
        return f"{self.src}->{self.dst}"


def candidate_edges() -> list[Edge]:
    """All edges allowed by kind, direction and the definitional constraints (self loops last)."""
    out = []
    for s in ENV:
        for d in SENSOR:
            out.append(Edge(s, d, "env>sensor"))
        for d in NAV:
            out.append(Edge(s, d, "env>nav"))
    for s in SENSOR:
        for d in NAV:
            out.append(Edge(s, d, "sensor>nav:def" if (s, d) in DEFINITIONAL else "sensor>nav:aux"))
    for s in NAV:
        for d in NAV:
            if s != d:
                out.append(Edge(s, d, "nav>nav"))
    out += [Edge(n, n, "self") for n in NODES]
    return out


def edge_index(edges: list[Edge]) -> dict:
    idx = {n: i for i, n in enumerate(NODES)}
    return {"src": np.array([idx[e.src] for e in edges]),
            "dst": np.array([idx[e.dst] for e in edges]),
            "rel": np.array([RELATIONS.index(e.rel) for e in edges]),
            "gated": np.array([not e.protected for e in edges]),
            "names": [e.name for e in edges]}


def restrict(edges: list[Edge], keep_names: set[str]) -> list[Edge]:
    """Final structure E_*: protected edges + the kept gated ones (pruned edges are not computed)."""
    return [e for e in edges if e.protected or e.name in keep_names]
