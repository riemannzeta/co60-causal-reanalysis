"""Causal DAG for the Taiwan Co-60 cohort, plus a self-contained d-separation / backdoor
analyzer (no dowhy/pgmpy dependency -- just networkx, so the identifiability logic is
transparent and auditable).

Run directly to print the DAG, the open backdoor paths under the *available* adjustment
set, and what would be needed for identification. Also writes results/dag.json and dag.dot.

The DAG encodes the data-generating process argued for in the plan:
  - Dose is effectively assigned at the BUILDING/APARTMENT level (shared gamma field).
  - SES/neighborhood is an unmeasured common cause of building choice and baseline cancer.
  - The external comparison induces SELECTION (healthy/young-cohort) -- represented as a
    node conditioned on by the external-SIR design.
  - Reconstructed (assigned) dose differs from TRUE dose via a dose-error node.
"""
from __future__ import annotations

import itertools
import json
import os

import networkx as nx

# ---- Nodes -------------------------------------------------------------------------
# measured (available for adjustment) vs unmeasured, for reporting.
MEASURED = {"AssignedDose", "AgeAtExposure", "AttainedAge", "Sex", "CalendarPeriod", "Building"}
UNMEASURED = {"TrueDose", "DoseError", "SES", "Smoking", "ReproHistory", "Selection"}

# ---- Directed edges (cause -> effect) ----------------------------------------------
EDGES = [
    # dose generation / measurement
    ("Building", "TrueDose"),        # rebar contamination is per-building
    ("SES", "Building"),             # who ends up in which 1982 Taipei apartment
    ("TrueDose", "AssignedDose"),    # reconstruction targets true dose...
    ("DoseError", "AssignedDose"),   # ...but is corrupted by reconstruction error
    ("AgeAtExposure", "AssignedDose"),  # occupancy/activity weighting uses age/sex
    ("Sex", "AssignedDose"),
    # outcome
    ("TrueDose", "Cancer"),          # the causal effect of interest
    ("AgeAtExposure", "Cancer"),
    ("AttainedAge", "Cancer"),
    ("Sex", "Cancer"),
    ("CalendarPeriod", "Cancer"),
    ("SES", "Cancer"),               # SES -> baseline cancer risk & screening
    ("Smoking", "Cancer"),
    ("ReproHistory", "Cancer"),      # reproductive history -> breast cancer
    ("SES", "Smoking"),
    ("AgeAtExposure", "AttainedAge"),
    # selection: the external-SIR design conditions on cohort membership, which is a
    # collider-ish selection node driven by SES (who lives here) and age structure.
    ("SES", "Selection"),
    ("AgeAtExposure", "Selection"),
]

# The effect we want: cumulative radiation dose -> Cancer. We measure AssignedDose but the
# causal parent of Cancer is TrueDose. That gap (measurement error) is itself part of the
# identification problem, so we analyze the AssignedDose -> Cancer path family.
TREATMENT = "AssignedDose"
OUTCOME = "Cancer"

# Adjustment sets to compare
AVAILABLE_ADJUSTMENT = {"AgeAtExposure", "AttainedAge", "Sex", "CalendarPeriod"}


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(EDGES)
    return g


# ---- d-separation via active-path classification -----------------------------------
def _is_collider(g: nx.DiGraph, a, b, c) -> bool:
    """On the path a - b - c, is b a collider (a -> b <- c)?"""
    return g.has_edge(a, b) and g.has_edge(c, b)


def _path_active(g: nx.DiGraph, path, Z) -> bool:
    """Is an undirected path d-connected (active) given conditioning set Z?

    Standard rules: a non-collider node on the path blocks iff it is IN Z; a collider node
    blocks iff neither it nor any descendant is in Z.
    """
    desc_cache = {}
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]
        if _is_collider(g, a, b, c):
            if b not in desc_cache:
                desc_cache[b] = nx.descendants(g, b) | {b}
            if not (desc_cache[b] & Z):
                return False  # collider not opened -> path blocked
        else:
            if b in Z:
                return False  # non-collider conditioned on -> path blocked
    return True


def backdoor_paths(g: nx.DiGraph, treatment, outcome, Z):
    """Return (open, blocked) backdoor paths given adjustment set Z.

    A backdoor path is any path between treatment and outcome whose first edge points INTO
    the treatment (treatment <- ...). We enumerate simple paths on the undirected skeleton.
    """
    ug = g.to_undirected()
    open_paths, blocked_paths = [], []
    for path in nx.all_simple_paths(ug, treatment, outcome):
        # backdoor: edge from path[1] into treatment
        if not g.has_edge(path[1], treatment):
            continue
        if _path_active(g, path, Z):
            open_paths.append(path)
        else:
            blocked_paths.append(path)
    return open_paths, blocked_paths


def analyze(Z=None):
    g = build_graph()
    Z = set(Z or [])
    open_p, blocked_p = backdoor_paths(g, TREATMENT, OUTCOME, Z)
    # descendants of treatment must not be in Z (backdoor criterion condition 1)
    bad_controls = Z & (nx.descendants(g, TREATMENT))
    return {
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "adjustment_set": sorted(Z),
        "open_backdoor_paths": [" - ".join(p) for p in open_p],
        "blocked_backdoor_paths": [" - ".join(p) for p in blocked_p],
        "descendant_controls_included": sorted(bad_controls),
        "identified": len(open_p) == 0 and len(bad_controls) == 0,
    }


def to_dot(path):
    g = build_graph()
    lines = ["digraph co60 {", '  rankdir=LR;', '  node [shape=box, fontname="Helvetica"];']
    for n in g.nodes:
        style = "filled" if n in UNMEASURED else "solid"
        fill = "#eeeeee" if n in UNMEASURED else "white"
        tag = " (unmeasured)" if n in UNMEASURED else ""
        lines.append(f'  "{n}" [style={style}, fillcolor="{fill}", label="{n}{tag}"];')
    for a, b in g.edges:
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    from co60lib import RESULTS, banner
    os.makedirs(RESULTS, exist_ok=True)

    banner("DAG: nodes and edges")
    g = build_graph()
    print(f"nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    print(f"measured/adjustable: {sorted(MEASURED)}")
    print(f"unmeasured (latent): {sorted(UNMEASURED)}")

    banner("Backdoor analysis with the AVAILABLE adjustment set")
    avail = analyze(AVAILABLE_ADJUSTMENT)
    print(f"adjust for: {avail['adjustment_set']}")
    print(f"identified by backdoor criterion? {avail['identified']}")
    print("OPEN backdoor paths (confounding NOT removed):")
    for p in avail["open_backdoor_paths"]:
        print(f"   * {p}")
    print(f"({len(avail['blocked_backdoor_paths'])} backdoor paths successfully blocked)")

    banner("What WOULD identify the effect?")
    # add the unmeasured confounders and see if it closes
    full = analyze(AVAILABLE_ADJUSTMENT | {"SES", "Smoking", "ReproHistory", "Building"})
    print(f"adjust for: {full['adjustment_set']}")
    print(f"identified? {full['identified']}")
    if full["open_backdoor_paths"]:
        print("still open (dose-error path is not closeable by adjustment):")
        for p in full["open_backdoor_paths"]:
            print(f"   * {p}")

    result = {
        "available_adjustment": avail,
        "full_adjustment": full,
        "note": (
            "The available adjustment set leaves SES/Smoking/ReproHistory backdoor paths OPEN. "
            "Identification therefore rests on the authors' quasi-randomization assumption "
            "(residents unaware of contamination => dose independent of lifestyle | age), which "
            "the bias analyses probe but cannot verify. The TrueDose->AssignedDose (DoseError) "
            "edge is a measurement problem, not a confounding one, and is handled separately."
        ),
    }
    save_path = os.path.join(RESULTS, "dag.json")
    with open(save_path, "w") as fh:
        json.dump(result, fh, indent=2)
    to_dot(os.path.join(os.path.dirname(save_path), "..", "dag.dot"))
    print(f"\nwrote {save_path} and dag.dot")


if __name__ == "__main__":
    main()
