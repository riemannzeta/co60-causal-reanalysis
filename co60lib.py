"""Shared helpers for the Taiwan Co-60 cohort causal reanalysis.

Everything downstream (reproduction, causal models, bias analysis) imports from here so
that the representative dose assignments and the statistical primitives live in one place.

Nothing here is Co-60-specific except REP_DOSE_SV, which turns the published *dose groups*
(<5, 5-99/5-100, >=100 mSv) into point doses so we can fit dose-response models to
aggregate cells. Those assignments are genuinely uncertain and are stress-tested in
04_bias_analysis.py.
"""
from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

# Representative cumulative dose (in Sieverts) assigned to each published dose group.
# Anchored to the cohort dose summary (mean 47.7 mSv, median 6.3 mSv, range <1-2363 mSv):
# the bulk sits low, the top group has a long right tail. These are the base-case values;
# 04_bias_analysis.py sweeps them.
REP_DOSE_SV = {
    "lt5": 0.002,     # <5 mSv    (near the median of the low group)
    "5_99": 0.030,    # 5-99 mSv
    "5_100": 0.030,   # 5-100 mSv (Table 3 uses this label)
    "ge100": 0.250,   # >=100 mSv (long tail to 2.36 Sv)
}


def load_csv(name: str) -> pd.DataFrame:
    """Load a digitized data table, skipping the '#'-prefixed provenance header."""
    path = name if os.path.isabs(name) else os.path.join(DATA, name)
    return pd.read_csv(path, comment="#", skip_blank_lines=True)


def save_result(name: str, obj: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def load_result(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


# --------------------------------------------------------------------------------------
# Standardized Incidence Ratio with an exact-Poisson (Byar) confidence interval.
# --------------------------------------------------------------------------------------
def sir_byar(observed: int, expected: float, conf: float = 0.95) -> dict:
    """SIR = observed/expected with Byar's approximation to the exact Poisson CI.

    Byar's method is the standard for SIR/SMR CIs and is accurate down to small counts.
    """
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    o = observed
    if o == 0:
        lower = 0.0
    else:
        lower = o * (1 - 1 / (9 * o) - z / (3 * math.sqrt(o))) ** 3
    upper = (o + 1) * (1 - 1 / (9 * (o + 1)) + z / (3 * math.sqrt(o + 1))) ** 3
    return {
        "observed": o,
        "expected": expected,
        "sir": o / expected,
        "ci_low": lower / expected,
        "ci_high": upper / expected,
        "conf": conf,
    }


# --------------------------------------------------------------------------------------
# E-value for unmeasured confounding (VanderWeele & Ding, Ann Intern Med 2017).
# --------------------------------------------------------------------------------------
def evalue_rr(rr: float, ci_low: float | None = None) -> dict:
    """E-value for a risk-ratio-like estimate (HR treated as RR; conservative for rare-ish
    outcomes). Returns the E-value for the point estimate and, if given, for the CI limit
    nearest the null. The E-value is the minimum strength of association (on the RR scale)
    that an unmeasured confounder would need with BOTH exposure and outcome to explain away
    the estimate.
    """
    def _ev(x):
        if x <= 1:
            # reflect protective estimates through the null so the formula applies
            x = 1 / x
        return x + math.sqrt(x * (x - 1))

    out = {"rr": rr, "evalue_point": _ev(rr)}
    if ci_low is not None:
        # E-value of the confidence limit closest to the null
        if rr > 1:
            out["evalue_ci"] = 1.0 if ci_low <= 1 else _ev(ci_low)
        else:
            out["evalue_ci"] = _ev(ci_low)
    return out


# --------------------------------------------------------------------------------------
# Aggregate dose-response fitters.
# Cells: cases (Poisson), person_years (offset), dose_sv, and integer stratum codes for
# nuisance/baseline strata (e.g. age-at-exposure).
# --------------------------------------------------------------------------------------
@dataclass
class DoseFit:
    model: str
    param: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pvalue: float
    note: str = ""

    def as_dict(self):
        return asdict(self)


def _stratum_codes(strata):
    """Map arbitrary stratum labels to integer codes 0..k-1 (stable order of appearance)."""
    strata = np.asarray(strata)
    levels = list(dict.fromkeys(strata.tolist()))
    idx = {lv: i for i, lv in enumerate(levels)}
    return np.array([idx[s] for s in strata])


def _strata_design(strata):
    """Return (matrix of stratum dummies incl. intercept, n_strata)."""
    strata = np.asarray(strata)
    levels = list(dict.fromkeys(strata.tolist()))
    idx = {lv: i for i, lv in enumerate(levels)}
    codes = np.array([idx[s] for s in strata])
    D = np.zeros((len(strata), len(levels)))
    D[np.arange(len(strata)), codes] = 1.0
    return D, len(levels)


def fit_err_linear(cases, person_years, dose_sv, strata=None) -> DoseFit:
    """Fit the radiation-epidemiology standard linear excess-relative-risk model:

        rate_i = exp(alpha_{stratum(i)}) * (1 + beta * dose_i)

    beta is the ERR per Sv. Baseline log-rate is free per stratum (so this is an internal,
    stratum-adjusted dose-response -- no external reference population).

    Fit by PROFILE likelihood: for any beta, each stratum baseline has the closed form
        exp(alpha_s) = (sum_s cases) / (sum_s (1+beta*dose) * py),
    so the whole problem reduces to a well-behaved 1-D optimization over beta. This is far
    more stable than a joint gradient fit -- in particular it does not blow up when a
    reference cell has zero cases (which a naive BFGS fit does).
    """
    cases = np.asarray(cases, float)
    py = np.asarray(person_years, float)
    dose = np.asarray(dose_sv, float)
    if strata is None:
        strata = np.zeros(len(cases), int)
    codes = _stratum_codes(strata)
    dmax = float(dose.max())
    beta_lo = -1.0 / dmax + 1e-9  # keep 1 + beta*dose > 0 for the largest dose
    beta_hi = 1.0e4

    def alpha_exp(beta):
        rr = 1 + beta * dose
        ea = np.empty(len(cases))
        for s in np.unique(codes):
            m = codes == s
            denom = np.sum(rr[m] * py[m])
            ea[m] = (cases[m].sum() / denom) if denom > 0 else 0.0
        return ea

    def negprofile(beta):
        rr = 1 + beta * dose
        if np.any(rr <= 0):
            return 1e12
        mu = alpha_exp(beta) * rr * py
        mu = np.clip(mu, 1e-300, None)
        return float(np.sum(mu - cases * np.log(mu)))

    res = optimize.minimize_scalar(negprofile, bounds=(beta_lo, beta_hi), method="bounded",
                                   options={"xatol": 1e-10})
    beta = float(res.x)
    # SE from the curvature of the profile log-likelihood (numerical 2nd derivative)
    h = max(abs(beta) * 1e-3, 1e-4)
    f0, fp, fm = negprofile(beta), negprofile(beta + h), negprofile(beta - h)
    second = (fp - 2 * f0 + fm) / h ** 2
    se = float(np.sqrt(1.0 / second)) if second > 0 else float("nan")
    lr = 2 * (negprofile(0.0) - f0)
    pval = float(stats.chi2.sf(max(lr, 0.0), df=1))
    z = stats.norm.ppf(0.975)
    unstable = (not np.isfinite(se)) or beta > beta_hi * 0.99 or abs(beta) > 1e3
    note = "rate = exp(alpha_stratum)*(1+beta*dose_Sv); profile-likelihood MLE"
    if unstable:
        note += " [UNSTABLE: sparse/zero reference cell -- interpret via log-linear HR]"
    lo = beta - z * se if np.isfinite(se) else float("nan")
    hi = beta + z * se if np.isfinite(se) else float("nan")
    return DoseFit("linear-ERR", "ERR_per_Sv", beta, se, lo, hi, pval, note=note)


def fit_loglinear_hr(cases, person_years, dose_sv, strata=None):
    """Log-linear Poisson dose-response: rate = exp(alpha_stratum + gamma*dose_Sv).
    Returns HR per 100 mSv = exp(gamma*0.1) with a Wald CI. Uses statsmodels if available,
    else a small internal Newton fit.
    """
    cases = np.asarray(cases, float)
    py = np.asarray(person_years, float)
    dose = np.asarray(dose_sv, float)
    if strata is None:
        strata = np.zeros(len(cases), int)
    D, k = _strata_design(strata)
    X = np.column_stack([D, dose])
    offset = np.log(py)
    try:
        import statsmodels.api as sm
        model = sm.GLM(cases, X, family=sm.families.Poisson(), offset=offset)
        fit = model.fit()
        gamma = fit.params[-1]
        se = fit.bse[-1]
        pval = fit.pvalues[-1]
    except Exception:
        gamma, se, pval = _poisson_newton(X, cases, offset)
    z = stats.norm.ppf(0.975)
    hr100 = math.exp(gamma * 0.1)
    return DoseFit("log-linear", "HR_per_100mSv", hr100, float("nan"),
                   math.exp((gamma - z * se) * 0.1), math.exp((gamma + z * se) * 0.1),
                   float(pval), note="HR per 100 mSv = exp(gamma*0.1); rate=exp(alpha+gamma*dose_Sv)")


def _poisson_newton(X, y, offset, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta + offset
        mu = np.exp(eta)
        W = mu
        z = eta - offset + (y - mu) / np.clip(mu, 1e-8, None)
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta + offset
    mu = np.exp(eta)
    cov = np.linalg.inv((X.T * mu) @ X)
    se = math.sqrt(cov[-1, -1])
    pval = 2 * stats.norm.sf(abs(beta[-1] / se))
    return float(beta[-1]), float(se), float(pval)


def banner(title: str):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")
