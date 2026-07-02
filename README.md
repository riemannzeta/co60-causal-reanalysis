# Taiwan Co-60 rebar cohort — a Pearl-style causal reanalysis

Did chronic low-dose-rate gamma exposure in Taiwan's cobalt-60-contaminated apartments
*cause* cancer? This repo reanalyzes the **published aggregate data** on that cohort with an
explicit causal DAG, reproduces the competing published results, estimates the internal
dose–response, and stress-tests it with quantitative bias analysis. The reconciled verdict
is generated into [`synthesis.md`](synthesis.md).

**Read the report:** a self-contained HTML write-up of the methodology, results, and a
response to Southwood & Chalmers' *[Low-dose radiation is probably fine](https://www.worksinprogress.news/p/low-dose-radiation-is-probably-fine)*
(Works in Progress) lives at [`docs/index.html`](docs/index.html). If GitHub Pages is
enabled for this repo (Settings → Pages → `main` / `docs`), it serves at
`https://riemannzeta.github.io/co60-causal-reanalysis/`.

## Background

In 1982–84, one Taiwan steel mill recycled orphaned Co-60 sources into reinforcing rebar
used in ~180–200 Taipei-area buildings. ~6,000–10,000 residents were chronically irradiated
from 1983 until discovery (1992) and relocation. The literature reaches opposite
conclusions:

- **Hwang / Hsieh–Chang** (registry-linked incidence): a positive internal dose–response —
  excess breast cancer and leukaemia.
- **Chen / Luan** (mortality, "hormesis"): cancer *deaths* far below the general population;
  claimed protective effect.
- **Doss** (comment): the external SIR is a *deficit* (~0.84), so site-specific "excesses"
  are multiple-comparison noise.

The user's cited paper, **Tung et al. 1998 (PMID 9600303)**, is the dose-*reconstruction*
methodology paper, not the cancer-outcome study — it parameterizes our measurement-error model.

## Data availability (important)

There is **no public individual-level microdata**. It is held under restricted access by
Taiwanese institutions (Taipei Medical Univ / National Yang-Ming; Taiwan Cancer Registry;
Atomic Energy Council for dosimetry) and released only on-site via Taiwan's Health and
Welfare Data Science Center to IRB-approved, Taiwan-affiliated researchers (no export).

**What is public** — and what this repo uses — are the open-access **aggregate tables**.
They are digitized by hand into `data/*.csv`, each with a provenance header:

| File | Source (open access) |
|---|---|
| `bjc2017_table1_solid.csv`, `bjc2017_table1_leukaemia.csv` | Hsieh/Chang 2017, *BJC*, Table 1 — [PMC5729469](https://pmc.ncbi.nlm.nih.gov/articles/PMC5729469/) |
| `bjc2017_table2_hr.csv` | Hsieh/Chang 2017, Table 2 (HR/100 mSv by site) |
| `bjc2017_table3_breast.csv` | Hsieh/Chang 2017, Table 3 (breast, age×dose joint) |
| `chen2007_table1_dose.csv`, `chen2007_table2_outcomes.csv` | Chen/Luan 2007, *Dose-Response* — [PMC2477708](https://pmc.ncbi.nlm.nih.gov/articles/PMC2477708/) |
| `doss2018_sir.csv` | Doss 2018, *BJC* comment — [PMC5846074](https://pmc.ncbi.nlm.nih.gov/articles/PMC5846074/) |
| `hwang2006_sir.csv` | Hwang 2006, *IJRB* — [PMID 17178625](https://pubmed.ncbi.nlm.nih.gov/17178625/) |

## Run it

```bash
./run.sh          # first run creates .venv (via uv or venv) and installs requirements.txt
# or, if the venv already exists:
make all
```

Pipeline stages (also runnable individually via `make qc|dag|reproduce|causal|bias|synthesis`):

| Stage | Script | Output |
|---|---|---|
| Transcription QC | `01_digitize_qc.py` | fails loudly unless all cells reconcile to reported totals |
| Causal DAG + identifiability | `dag.py` | `results/dag.json`, `dag.dot` (open backdoor paths) |
| Reproduce published results | `02_reproduce.py` | `results/reproduce.json` |
| Internal causal models + reconciliation | `03_causal_models.py` | `results/causal.json` |
| Bias analysis | `04_bias_analysis.py` | `results/bias.json` |
| Synthesis | `05_synthesis.py` | `synthesis.md` |

## Method in one paragraph

We encode the data-generating process as a DAG (`dag.py`) in which dose is assigned at the
**building** level, SES is an unmeasured common cause of building choice and baseline cancer,
and the external-SIR design conditions on a **selection** node. A self-contained
d-separation routine shows the causal effect is **not** identified by the available
adjustment set (age, sex, calendar period) — the SES/smoking/reproductive-history backdoors
stay open — so identification rests on the authors' *quasi-randomization* assumption
(residents were unaware of contamination). We reproduce the external SIR exactly, fit
internal linear-ERR and log-linear dose–response models to the aggregate cells, show the
external "deficit" is an age-structure artifact, and probe robustness with E-values, a
classical-vs-Berkson measurement-error simulation, exposure-model sensitivity, and FDR.

## Key results (computed)

- External SIR **0.84** (0.74–0.95) reproduced exactly from counts.
- Internal dose–response **positive and significant**; age-adjusted breast HR/100 mSv ≈ **1.5**.
- The external deficit is driven by a **~14× age gradient** vs a ~2.6× dose gradient — an
  age-structure artifact, not hormesis.
- Robustness: E-values are **modest** (~1.2–1.6 — the main caveat); classical dose error
  **attenuates** the slope (so estimates are conservative); breast + leukaemia survive **FDR**.

## Limitations

- **Aggregate-only.** No individual-level causal ML (g-methods, TMLE); inference is limited
  to published strata. Representative within-group doses are assumed (sign/significance are
  invariant to them; magnitude is not).
- **Core confounders unmeasured** (smoking, reproductive history, cluster SES) → the effect
  is identified only under the quasi-randomization assumption, which the bias analyses probe
  but cannot verify.
- **Do not pool** Chen/Luan (~10,000, mortality, mean ~0.4 Sv) with Hwang/Hsieh
  (6,242–7,271, incidence, mean ~48 mSv): different denominators and endpoints.
- A definitive answer requires the restricted HWDC microdata (a Taiwan-based collaborator +
  IRB) — a possible Phase 2.

## Layout

```
data/                 digitized source tables (CSV + provenance headers)
co60lib.py            loaders, Byar SIR CI, E-value, linear-ERR + log-linear fitters
dag.py                causal DAG + d-separation / backdoor analyzer
01_digitize_qc.py     transcription integrity gate
02_reproduce.py       reproduce Doss SIR + BJC gradients
03_causal_models.py   internal dose-response + SIR-vs-internal reconciliation
04_bias_analysis.py   E-values, measurement error, exposure sensitivity, FDR
05_synthesis.py       assembles synthesis.md + assumption ledger
run.sh / Makefile     end-to-end harness
```
