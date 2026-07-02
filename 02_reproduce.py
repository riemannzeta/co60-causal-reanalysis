"""Validation stage: reproduce the published headline numbers from the digitized aggregates,
to prove the data + our modeling machinery agree with the source papers before we trust any
*new* estimates in 03/04.

Two reproduction targets:
  (A) Doss 2018 external SIR (0.84, 0.75) -- exactly reproducible from observed/expected counts.
  (B) BJC 2017 internal dose-response HRs -- approximately reproducible from the aggregate
      dose-group cells (we lack the individual continuous doses, so we assign representative
      group doses; the sign, ordering and rough magnitude should match).
"""
from __future__ import annotations

import math

import numpy as np

from co60lib import (REP_DOSE_SV, DoseFit, banner, fit_err_linear, fit_loglinear_hr,
                     load_csv, save_result, sir_byar)


def reproduce_doss():
    banner("(A) Reproduce Doss 2018 external SIR from counts")
    doss = load_csv("doss2018_sir.csv")
    rows = []
    for _, r in doss.iterrows():
        got = sir_byar(int(r.observed), float(r.expected))
        pub = (float(r.published_sir), float(r.published_ci95_low), float(r.published_ci95_high))
        print(f"  {r.dataset}: obs={int(r.observed)} exp={r.expected}")
        print(f"     recomputed SIR = {got['sir']:.3f}  (95% CI {got['ci_low']:.3f}-{got['ci_high']:.3f})")
        print(f"     published  SIR = {pub[0]:.2f}  (95% CI {pub[1]:.2f}-{pub[2]:.2f})")
        agree = abs(got["sir"] - pub[0]) < 0.02
        print(f"     point estimate matches published: {agree}")
        rows.append({"dataset": r.dataset, **got, "published_sir": pub[0],
                     "published_ci": [pub[1], pub[2]], "matches": bool(agree)})
    return rows


def _cells_from_marginal(df):
    d = df[df.stratum_type == "dose_group"].copy()
    d = d.rename(columns={"stratum": "dose_group"})
    d["dose_sv"] = d["dose_group"].map(REP_DOSE_SV)
    return d


def reproduce_bjc_marginal(name, df, label):
    """Unadjusted dose-response on the marginal dose-group cells (all we have for the full
    cohort). We validate the *direction and significance* of the gradient -- NOT its
    magnitude, which is not recoverable from aggregate cells (we lack within-group mean
    doses, and the marginal table is not age-adjusted). Magnitude sensitivity to the dose
    assignment is explored in 04_bias_analysis.py."""
    banner(f"(B) Reproduce {label} dose-response (marginal dose groups, UNADJUSTED)")
    d = _cells_from_marginal(df)
    cells = [(g, int(c), int(p)) for g, c, p in zip(d.dose_group, d.cases, d.person_years)]
    print("   cells (group, cases, py):", cells)
    # assignment-FREE evidence: crude rate per group, and monotonic ordering
    rates = (d.cases.values / d.person_years.values) * 1e4
    print("   crude rate per 10k py:", {g: round(r, 2) for g, r in zip(d.dose_group, rates)})
    monotone = bool(np.all(np.diff(rates) >= 0))
    print(f"   rate increases monotonically with dose group: {monotone}  (assignment-free)")
    ll = fit_loglinear_hr(d.cases.values, d.person_years.values, d.dose_sv.values)
    err = fit_err_linear(d.cases.values, d.person_years.values, d.dose_sv.values)
    print(f"   log-linear HR per 100 mSv = {ll.estimate:.3f} (95% CI {ll.ci_low:.3f}-{ll.ci_high:.3f}, p={ll.pvalue:.3g})")
    err_ci = (f"(95% CI {err.ci_low:.2f}-{err.ci_high:.2f})" if np.isfinite(err.se) else "(CI unstable)")
    print(f"   linear ERR per Sv        = {err.estimate:.2f} {err_ci}  p={err.pvalue:.3g}")
    if "UNSTABLE" in err.note:
        print(f"     note: {err.note}")
    print("   (magnitude depends on within-group dose assignment + is age-confounded; "
          "we validate sign+significance only)")
    return {"target": label, "monotone_gradient": monotone,
            "crude_rate_per10k": {g: float(r) for g, r in zip(d.dose_group, rates)},
            "loglinear": ll.as_dict(), "err": err.as_dict()}


def reproduce_breast_ageadjusted():
    """Table 3 is a JOINT age x dose table, so here we CAN adjust for age-at-exposure and
    reproduce the key age-modification: a much steeper dose-response in those exposed <=20."""
    banner("(B) Reproduce BJC 2017 breast-cancer age-at-exposure modification (Table 3)")
    breast = load_csv("bjc2017_table3_breast.csv")
    cells = breast[breast.dose_group != "trend"].copy()
    cells["dose_sv"] = cells["dose_group"].map(REP_DOSE_SV)
    out = {}
    for age, pub in [("le20", 1.38), ("gt20", 1.07)]:
        sub = cells[cells.age_at_exposure == age]
        ll = fit_loglinear_hr(sub.cases.values, sub.person_years.values, sub.dose_sv.values)
        print(f"   IAE {age}: HR per 100 mSv = {ll.estimate:.3f} "
              f"(95% CI {ll.ci_low:.3f}-{ll.ci_high:.3f}); published trend {pub}")
        out[age] = {"hr_per_100msv": ll.estimate, "ci": [ll.ci_low, ll.ci_high],
                    "pvalue": ll.pvalue, "published_trend": pub}
    steeper = out["le20"]["hr_per_100msv"] > out["gt20"]["hr_per_100msv"]
    print(f"   dose-response steeper for the young-exposed (<=20): {steeper}  "
          f"<- the paper's central qualitative claim")
    out["young_steeper"] = bool(steeper)
    return out


def main():
    result = {}
    result["doss_sir"] = reproduce_doss()
    solid = load_csv("bjc2017_table1_solid.csv")
    leuk = load_csv("bjc2017_table1_leukaemia.csv")
    result["solid_marginal"] = reproduce_bjc_marginal("solid", solid, "solid cancers")
    result["leuk_marginal"] = reproduce_bjc_marginal("leuk", leuk, "leukaemia")
    result["breast_age"] = reproduce_breast_ageadjusted()

    banner("Reproduction summary")
    doss_ok = all(r["matches"] for r in result["doss_sir"])
    print(f"  Doss external SIRs reproduced exactly: {doss_ok}")
    print(f"  Solid marginal dose-response positive: {result['solid_marginal']['err']['estimate'] > 0}")
    print(f"  Leukaemia marginal dose-response positive: {result['leuk_marginal']['err']['estimate'] > 0}")
    print(f"  Young-exposed breast steeper: {result['breast_age']['young_steeper']}")
    save_result("reproduce.json", result)


if __name__ == "__main__":
    main()
