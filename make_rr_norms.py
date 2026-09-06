#!/usr/bin/env python3
"""
make_rr_norms.py — RR_irregularity REFERENCE NORMS from published RMSSD data.
No cohort download required.

    python make_rr_norms.py --out tables/
    python refit.py --tables tables --out model/arterialage.json

WHAT THIS PRODUCES, AND WHAT IT DOES NOT
  It writes quantiles_RR_irregularity.csv ONLY -- a percentile table by age
  band and sex. That is exactly what the HMOD tier needs, because the tier
  ranks a subject against a reference distribution and NEVER uses a beta.

  It does NOT write a marginal. RR_irregularity therefore contributes ZERO to
  the risk index. Unlocking the atrial-electrical domain for TIERING is a
  different and much weaker claim than adding a risk term, and conflating the
  two is precisely the error that left dtau's companions contributing nothing
  while looking as though they contributed something.

DERIVATION (every step is checkable)
  1. Median RMSSD by age band and sex: Lifelines Cohort Study,
     Tegegne et al. 2020, N = 84,772, 10-second resting ECGs.
  2. RMSSD is log-normally distributed ("measured on a geometric scale"), so
     sigma is recovered from the published mean/median ratio in the same
     source:  mean/median = exp(sigma^2 / 2)  =>  sigma = sqrt(2*ln(ratio)).
     This yields sigma ~0.62 in mid-life rising to ~0.89 at 65+, the rise
     reflecting an inflating upper tail as AF prevalence climbs.
  3. RR_irregularity = RMSSD / mean(RR), and mean(RR) = 60000 / HR.
     ASSUMED resting HR by age band (below); state it, it is the weakest link.
  4. Percentiles are the lognormal quantiles implied by (1)-(3).

DIRECTION CHECK: higher RR_irregularity = worse. Independently supported --
the Rotterdam Study (12,334 participants, 1,302 incident AF over 9.4 y) found
HIGHER RMSSD associated with new-onset AF, HR 1.33 (95% CI 1.13-1.54). That is
an AF endpoint, not heart failure, so it is NOT used as an effect size here.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import norm as _sn
from pathlib import Path

# Lifelines: median RMSSD (ms) and mean RMSSD (ms), men / women
LIFELINES = {   # age_mid: ((med_m, mean_m), (med_f, mean_f))
    22: ((47.6, 57.3), (52.1, 64.7)), 27: ((42.3, 52.1), (47.5, 58.0)),
    32: ((36.9, 45.4), (42.3, 51.6)), 37: ((32.8, 39.9), (37.9, 46.0)),
    42: ((29.0, 35.2), (33.9, 41.0)), 47: ((26.0, 31.6), (29.2, 35.6)),
    52: ((23.7, 28.7), (26.6, 31.8)), 57: ((21.0, 26.2), (22.5, 27.2)),
    62: ((19.1, 24.8), (20.5, 25.2)), 70: ((16.9, 25.2), (17.2, 23.7)),
}
BANDS = [(0, 50), (50, 60), (60, 70), (70, 200)]
HR_BY_BAND = {(0, 50): 68.0, (50, 60): 67.0, (60, 70): 66.0, (70, 200): 65.0}  # ASSUMED
GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def band_params(lo, hi, sex_idx):
    mids = [m for m in LIFELINES if lo <= m < hi] or \
           [min(LIFELINES, key=lambda m: min(abs(m - lo), abs(m - hi)))]
    med = float(np.mean([LIFELINES[m][sex_idx][0] for m in mids]))
    mean = float(np.mean([LIFELINES[m][sex_idx][1] for m in mids]))
    sigma = float(np.sqrt(max(2.0 * np.log(max(mean / med, 1.001)), 1e-4)))
    return med, sigma


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"  {'band':>10} {'sex':>4} {'RMSSD med':>10} {'sigma':>7} {'meanRR':>8} "
          f"{'p50':>8} {'p95':>8} {'p99':>8}")
    for lo, hi in BANDS:
        for sex in (0, 1):                       # 0 = female, 1 = male
            med, sigma = band_params(lo, hi, 1 - sex)   # LIFELINES idx 0=men
            rr_ms = 60000.0 / HR_BY_BAND[(lo, hi)]
            mu = np.log(med / rr_ms)             # lognormal on the RATIO
            q = {f"p{p}": float(np.exp(mu + sigma * _sn.ppf(p / 100.0))) for p in GRID}
            rows.append({"feature": "RR_irregularity", "age_lo": lo, "age_hi": hi,
                         "sex": float(sex), "n": 0, **q})
            print(f"  {f'{lo}-{hi}':>10} {'M' if sex else 'F':>4} {med:10.1f} {sigma:7.3f} "
                  f"{rr_ms:8.1f} {q['p50']:8.4f} {q['p95']:8.4f} {q['p99']:8.4f}")

    p = f"{a.out}/quantiles_RR_irregularity.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"\n  -> {p}")
    print("  NORMS ONLY. No marginal is written, so RR_irregularity contributes")
    print("  ZERO to the risk index. It unlocks the atrial-electrical domain for")
    print("  TIERING, where abnormal = above the 95th age-sex percentile.")
    print("\n  next:  python refit.py --tables tables --out model/arterialage.json")
