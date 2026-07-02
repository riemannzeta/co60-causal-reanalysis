"""Load every digitized table and check it against the cohort totals reported in the
source papers. Fails loudly (non-zero exit) on any mismatch so the pipeline cannot run on
a bad transcription.
"""
from __future__ import annotations

import sys

from co60lib import load_csv, save_result, banner

# Reported totals we validate against (Hsieh/Chang 2017 abstract + Table 1).
EXPECTED = {
    "subjects": 6242,
    "solid_cases": 247,          # after 10-yr latency exclusion
    "solid_py": 97106,
    "leuk_cases": 11,
    "leuk_py": 147984,
    "all_cancers_hr_cases": 249,  # Table 2 "all cancers" row
}


def _check(label, got, want, checks):
    ok = got == want
    checks.append((label, got, want, ok))
    return ok


def marginal_totals(df, value):
    """Each stratum_type (sex/age/dose) partitions the cohort, so each should sum to the
    overall total independently. Return the per-partition sums."""
    out = {}
    for st in ["sex", "age_at_exposure", "dose_group"]:
        out[st] = int(df.loc[df.stratum_type == st, value].sum())
    return out


def main():
    checks = []
    banner("QC: BJC 2017 Table 1 (solid cancers)")
    solid = load_csv("bjc2017_table1_solid.csv")
    ov = solid.loc[solid.stratum_type == "overall"].iloc[0]
    _check("solid overall subjects", int(ov.subjects), EXPECTED["subjects"], checks)
    _check("solid overall cases", int(ov.cases), EXPECTED["solid_cases"], checks)
    _check("solid overall person-years", int(ov.person_years), EXPECTED["solid_py"], checks)
    for st, s in marginal_totals(solid, "subjects").items():
        _check(f"solid subjects sum over {st}", s, EXPECTED["subjects"], checks)
    for st, s in marginal_totals(solid, "cases").items():
        _check(f"solid cases sum over {st}", s, EXPECTED["solid_cases"], checks)
    for st, s in marginal_totals(solid, "person_years").items():
        _check(f"solid py sum over {st}", s, EXPECTED["solid_py"], checks)

    banner("QC: BJC 2017 Table 1 (leukaemia)")
    leuk = load_csv("bjc2017_table1_leukaemia.csv")
    lov = leuk.loc[leuk.stratum_type == "overall"].iloc[0]
    _check("leuk overall cases", int(lov.cases), EXPECTED["leuk_cases"], checks)
    _check("leuk overall person-years", int(lov.person_years), EXPECTED["leuk_py"], checks)
    for st, s in marginal_totals(leuk, "cases").items():
        _check(f"leuk cases sum over {st}", s, EXPECTED["leuk_cases"], checks)

    banner("QC: BJC 2017 Table 3 (breast, age x dose joint)")
    breast = load_csv("bjc2017_table3_breast.csv")
    cells = breast[breast.dose_group != "trend"]
    total_breast = int(cells.cases.sum())
    # Table 2 lists 35 female breast cancers; Table 3 (with latency + age split) sums to 35.
    _check("breast Table 3 cells sum to Table 2 count", total_breast, 35, checks)

    banner("QC: BJC 2017 Table 2 (HRs)")
    hr = load_csv("bjc2017_table2_hr.csv")
    _check("all_cancers HR-table case count", int(hr.loc[hr.site == "all_cancers", "cases"].iloc[0]),
           EXPECTED["all_cancers_hr_cases"], checks)

    banner("QC: Doss 2018 and Chen 2007 load")
    doss = load_csv("doss2018_sir.csv")
    chen1 = load_csv("chen2007_table1_dose.csv")
    chen2 = load_csv("chen2007_table2_outcomes.csv")
    _check("doss rows", len(doss), 2, checks)
    _check("chen table1 rows", len(chen1), 5, checks)
    _check("chen observed cancer deaths", int(chen2.loc[chen2.outcome == "cancer_deaths", "observed"].iloc[0]),
           7, checks)

    # report
    banner("QC RESULTS")
    n_pass = sum(ok for *_, ok in checks)
    for label, got, want, ok in checks:
        mark = "PASS" if ok else "FAIL"
        extra = "" if ok else f"   (got {got}, want {want})"
        print(f"  [{mark}] {label}{extra}")
    print(f"\n{n_pass}/{len(checks)} checks passed")

    save_result("qc.json", {
        "n_pass": n_pass, "n_total": len(checks),
        "checks": [{"label": l, "got": g, "want": w, "ok": ok} for l, g, w, ok in checks],
    })

    if n_pass != len(checks):
        print("\nQC FAILED -- transcription error, stopping pipeline.")
        sys.exit(1)
    print("\nAll transcription QC checks passed.")


if __name__ == "__main__":
    main()
