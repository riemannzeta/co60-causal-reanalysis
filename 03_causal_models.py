"""Causal stage: the internal dose-response estimand, and a numerical reconciliation of the
apparent contradiction between the external SIR (~0.84, "protective") and the internal
dose-response (>0, "harmful").

Key Pearl-style move: the external SIR conditions on a non-exchangeable reference
population (Selection node in dag.py). The internal dose gradient does not. Two facts do
the reconciling:

  1. The article's headline "~35% lower" is the *crude*, all-ages comparison
     (247 cancers / 97,106 py = 254 per 100k vs Taiwan's national 390 per 100k -> SIR 0.65).
     Age-standardizing to the cohort's own age distribution -- what Doss actually does --
     moves it to SIR 0.84. So a substantial part of the crude 35% is age structure (the
     national rate is inflated by an elderly tail this cohort barely has).
  2. But a deficit *survives* standardization: Doss 0.84 (~16%), Hwang 2008 0.75 (~25%).
     That residual is NOT protection and NOT the cohort "being young" (its attained age by
     2012 is ~45, older than Taiwan's mean) -- it is healthy-cohort SELECTION.

Selection is the crux: an external deficit driven by selection cannot detect the internal
dose-response, because high- and low-dose residents *share* the selection. So the two
findings are not in conflict -- they answer different questions.
"""
from __future__ import annotations

import numpy as np

from co60lib import (REP_DOSE_SV, banner, fit_err_linear, fit_loglinear_hr, load_csv,
                     save_result)

# Taiwan national all-ages cancer incidence rate, 2008-2012, per 100k person-years, as
# cited by Southwood & Chalmers (Taiwan Cancer Registry). This is the crude reference the
# article's "35% lower" headline is computed against.
TAIWAN_NATIONAL_RATE_PER_100K = 390.0


def internal_breast_err():
    """Primary internal causal estimand for breast cancer: age-at-exposure-adjusted
    dose-response (Table 3 is a joint age x dose table, so age is genuinely controlled)."""
    banner("Internal causal dose-response: female breast (age-at-exposure adjusted)")
    breast = load_csv("bjc2017_table3_breast.csv")
    cells = breast[breast.dose_group != "trend"].copy()
    cells["dose_sv"] = cells["dose_group"].map(REP_DOSE_SV)
    # pooled model with age-at-exposure as the baseline stratum (adjusts for age)
    err = fit_err_linear(cells.cases.values, cells.person_years.values,
                         cells.dose_sv.values, strata=cells.age_at_exposure.values)
    ll = fit_loglinear_hr(cells.cases.values, cells.person_years.values,
                          cells.dose_sv.values, strata=cells.age_at_exposure.values)
    print(f"   pooled ERR per Sv (age-adjusted) = {err.estimate:.2f}"
          + (f" (95% CI {err.ci_low:.2f}-{err.ci_high:.2f})" if np.isfinite(err.se) else " (CI unstable)"))
    print(f"   pooled HR per 100 mSv (age-adj)  = {ll.estimate:.3f} "
          f"(95% CI {ll.ci_low:.3f}-{ll.ci_high:.3f}, p={ll.pvalue:.3g})")
    print("   -> internal gradient is POSITIVE after adjusting for age at exposure.")
    return {"err_per_sv": err.as_dict(), "hr_per_100msv": ll.as_dict()}


def age_gradient_vs_dose_gradient():
    """Mechanism note: the age-at-exposure rate gradient dwarfs the dose gradient. This is
    why a *crude* external comparison misleads -- NOT why the standardized SIR is <1 (Doss's
    SIR already conditions on age; see external_deficit_decomposition below). Note the cohort
    is not simply 'young': by 2012 its attained age (~45) exceeds Taiwan's mean -- the issue
    is the compressed age *distribution* (few elderly person-years), which the crude national
    all-ages rate is not."""
    banner("Mechanism: age gradient >> dose gradient (why a CRUDE comparison misleads)")
    solid = load_csv("bjc2017_table1_solid.csv")
    age = solid[solid.stratum_type == "age_at_exposure"].copy()
    dose = solid[solid.stratum_type == "dose_group"].copy()
    age_rate = (age.cases.values / age.person_years.values) * 1e4
    dose_rate = (dose.cases.values / dose.person_years.values) * 1e4
    age_grad = age_rate.max() / age_rate.min()
    dose_grad = dose_rate.max() / dose_rate.min()
    print(f"   age-at-exposure rates per 10k: {dict(zip(age.stratum, np.round(age_rate,2)))}")
    print(f"   -> oldest/youngest rate ratio = {age_grad:.1f}x")
    print(f"   dose-group rates per 10k:      {dict(zip(dose.stratum, np.round(dose_rate,2)))}")
    print(f"   -> highest/lowest dose rate ratio = {dose_grad:.1f}x")
    print(f"   age gradient is ~{age_grad/dose_grad:.0f}x larger than the dose gradient.")
    return {"age_gradient": float(age_grad), "dose_gradient": float(dose_grad),
            "age_rates": dict(zip(age.stratum, age_rate.tolist())),
            "dose_rates": dict(zip(dose.stratum, dose_rate.tolist()))}


def external_deficit_decomposition():
    """Decompose the external deficit into (a) age structure and (b) surviving selection,
    using only published numbers -- no invented reference weights.

    crude expected  = national all-ages rate x cohort person-years   -> the article's 35%
    standardized expected = Doss's age-standardized expected (296.4)  -> SIR 0.84 survives

    The gap between crude and standardized expected is the age-structure component; the gap
    that remains between standardized expected and observed is the healthy-cohort SELECTION
    component. Neither is evidence of radiation being protective."""
    banner("External deficit: crude 35% vs standardized ~16% -> the residual is SELECTION")
    solid = load_csv("bjc2017_table1_solid.csv")
    overall = solid[solid.stratum_type == "overall"].iloc[0]
    observed = float(overall.cases)          # 247 (Hsieh 2017 Table 1, solid cancers)
    py = float(overall.person_years)         # 97,106

    # (1) crude comparison -- exactly how the article computes "~35% lower"
    crude_expected = TAIWAN_NATIONAL_RATE_PER_100K / 1e5 * py
    crude_sir = observed / crude_expected

    # (2) age-standardized comparison -- Doss's printed values (indirect standardization)
    doss = load_csv("doss2018_sir.csv")
    drow = doss[doss.dataset == "hsieh2017_2012"].iloc[0]
    std_expected = float(drow.expected)      # 296.4
    std_observed = float(drow.observed)      # 249 (Doss's count; ~ = Table 1's 247)
    std_sir = std_observed / std_expected    # 0.84

    # decomposition of the CRUDE deficit into age structure vs surviving selection
    crude_deficit = 1.0 - crude_sir                      # ~0.35
    std_deficit = 1.0 - std_sir                          # ~0.16 (survives standardization)
    # count scale: how many of the "missing" crude cancers are removed by standardization
    missing_crude = crude_expected - observed
    removed_by_age = crude_expected - std_expected
    age_share_counts = removed_by_age / missing_crude
    # ratio scale: fraction of the crude *deficit* that standardization removes
    age_share_ratio = (crude_deficit - std_deficit) / crude_deficit

    print(f"   observed cancers = {observed:.0f} over {py:,.0f} person-years "
          f"({observed/py*1e5:.0f} per 100k)")
    print(f"   crude expected @ {TAIWAN_NATIONAL_RATE_PER_100K:.0f}/100k = {crude_expected:.0f}"
          f"  -> crude SIR = {crude_sir:.2f}  (the article's ~{crude_deficit*100:.0f}% lower)")
    print(f"   age-standardized expected (Doss) = {std_expected:.1f}"
          f"  -> SIR = {std_sir:.2f}  (deficit {std_deficit*100:.0f}% SURVIVES)")
    print(f"   age-structure share of the crude deficit: "
          f"{age_share_counts*100:.0f}% (counts) / {age_share_ratio*100:.0f}% (ratio scale)")
    print(f"   -> the surviving ~{std_deficit*100:.0f}% is healthy-cohort SELECTION, not protection.")
    print("   Selection is shared by high- and low-dose residents, so it cannot mask or create")
    print("   the internal dose-response -- the external deficit is uninformative about it.")
    return {"observed": observed, "person_years": py,
            "national_rate_per_100k": TAIWAN_NATIONAL_RATE_PER_100K,
            "crude_expected": crude_expected, "crude_sir": crude_sir,
            "std_expected": std_expected, "std_sir": std_sir,
            "crude_deficit": crude_deficit, "std_deficit_selection": std_deficit,
            "age_share_counts": float(age_share_counts),
            "age_share_ratio": float(age_share_ratio)}


def main():
    result = {}
    result["internal_breast"] = internal_breast_err()
    result["gradients"] = age_gradient_vs_dose_gradient()
    result["deficit_decomposition"] = external_deficit_decomposition()

    banner("Reconciliation verdict")
    print("  * Internal, age-adjusted dose-response: POSITIVE (dose -> cancer).")
    print("  * The article's crude '~35% lower' is inflated by age structure; standardizing")
    print("    (Doss) leaves a ~16% deficit that SURVIVES -- that residual is healthy-cohort")
    print("    SELECTION, not radiation being protective.")
    print("  * A selection-driven external deficit cannot detect the internal dose-response,")
    print("    because high- and low-dose residents share the selection. A Pearl analysis")
    print("    privileges the internal gradient, which nets the selection out.")
    save_result("causal.json", result)


if __name__ == "__main__":
    main()
