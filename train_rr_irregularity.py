#!/usr/bin/env python3
"""
train_rr_irregularity.py — the last untrained marginal, from MC-MED.

    python train_rr_irregularity.py --data /path/to/mc-med --out tables/

Emits marginal_RR_irregularity.csv and quantiles_RR_irregularity.csv in the
SAME aggregate format as the All of Us export, so `refit.py` picks them up
with no code change.

WHAT MC-MED IS AND ISN'T
  MC-MED is emergency-department monitor data with ICD heart-failure labels.
  Two things differ from the AoU marginals and both are recorded on the
  artefact:
    label_kind = "ED HF (ICD)" -- an ED population, not a community cohort.
    prevalence is far higher (~8%), so the HR->OR rare-outcome approximation
    used for the load priors does NOT hold here. Prevalence centring removes
    the base-rate component; the residual construct mismatch is a stated
    limitation, resolved only by device outcomes.

  This is a PREVALENT label. Per the AoU finding -- 96% of a prevalent-label
  effect was reverse causation -- this marginal is flagged accordingly and
  should be reported as associative, not predictive, until an incident ED
  cohort exists.

RR_irregularity: RMSSD-normalised, per visit, from the ECG-derived RR series.
  irregularity = RMSSD / mean(RR),  winsorised at the 1st/99th percentile.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd

MIN_CELL = 20
BANDS = [(0, 50), (50, 60), (60, 70), (70, 200)]
GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def load_mcmed(root: str) -> pd.DataFrame:
    """Expects the MC-MED release layout. Adjust the three filenames if yours
    differs -- everything downstream is generic."""
    root = Path(root)
    cand = {p.name.lower(): p for p in root.rglob("*.csv")}
    def find(*keys):
        for n, p in cand.items():
            if all(k in n for k in keys):
                return p
        raise SystemExit(f"could not find a CSV matching {keys} under {root}")

    vis = pd.read_csv(find("visit"))
    dx = pd.read_csv(find("diagnos") if any("diagnos" in n for n in cand) else find("dx"))
    num = pd.read_csv(find("numeric") if any("numeric" in n for n in cand) else find("vital"))
    return vis, dx, num


def build(root: str) -> pd.DataFrame:
    vis, dx, num = load_mcmed(root)
    idc = next(c for c in vis.columns if "csn" in c.lower() or "visit" in c.lower())
    agec = next((c for c in vis.columns if c.lower().startswith("age")), None)
    sexc = next((c for c in vis.columns if "gender" in c.lower() or "sex" in c.lower()), None)

    dcol = next(c for c in dx.columns if "icd" in c.lower() or "code" in c.lower())
    hf = dx[dx[dcol].astype(str).str.upper().str.startswith(("I50", "428"))][idc].unique()

    # RR irregularity from the numeric stream: RMSSD / mean RR, per visit
    tcol = next(c for c in num.columns if "time" in c.lower() or "seq" in c.lower())
    mcol = next(c for c in num.columns if "measure" in c.lower() or "name" in c.lower())
    vcol = next(c for c in num.columns if c.lower() in ("value", "val", "result"))
    hr = num[num[mcol].astype(str).str.lower().str.contains("heart|pulse|hr", na=False)]
    hr = hr[[idc, tcol, vcol]].dropna()
    hr[vcol] = pd.to_numeric(hr[vcol], errors="coerce")
    hr = hr[(hr[vcol] > 25) & (hr[vcol] < 220)].sort_values([idc, tcol])
    hr["rr"] = 60000.0 / hr[vcol]

    g = hr.groupby(idc)["rr"]
    rows = []
    for pid, s in g:
        if len(s) < 12:
            continue
        d = np.diff(s.values)
        rows.append({idc: pid, "RR_irregularity": float(np.sqrt(np.mean(d ** 2)) / s.mean()),
                     "n_obs": len(s)})
    feats = pd.DataFrame(rows)

    df = feats.merge(vis[[c for c in [idc, agec, sexc] if c]], on=idc, how="left")
    df["hf"] = df[idc].isin(hf).astype(int)
    df["age"] = pd.to_numeric(df[agec], errors="coerce") if agec else 60.0
    df["sex"] = (df[sexc].astype(str).str.upper().str[0] == "M").astype(float) if sexc else 0.0
    lo, hi = df.RR_irregularity.quantile([.01, .99])
    df["RR_irregularity"] = df.RR_irregularity.clip(lo, hi)
    return df.dropna(subset=["RR_irregularity", "age"])


def emit(df, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    x, y = df.RR_irregularity.values, df.hf.values
    age, sex = df.age.values, df.sex.values
    rows, kept, dropped = [], 0, 0
    for lo, hi in BANDS:
        for s in (0, 1):
            m = (age >= lo) & (age < hi) & (sex == s) & np.isfinite(x)
            if m.sum() < MIN_CELL * 2:
                continue
            ed = np.unique(np.quantile(x[m], np.linspace(0, 1, 16)))
            if len(ed) < 3:
                continue
            idx = np.clip(np.digitize(x[m], ed[1:-1]), 0, len(ed) - 2)
            for b in range(len(ed) - 1):
                sel = idx == b
                if sel.sum() < MIN_CELL:
                    dropped += 1; continue
                kept += 1
                rows.append(dict(feature="RR_irregularity", age_lo=lo, age_hi=hi, sex=s,
                                 lo=float(ed[b]), hi=float(ed[b + 1]), n_raw=int(sel.sum()),
                                 n_weighted=float(sel.sum()),
                                 events_weighted=float(y[m][sel].sum()),
                                 rate=float(y[m][sel].mean())))
    pd.DataFrame(rows).to_csv(f"{out}/marginal_RR_irregularity.csv", index=False)

    q = []
    for lo, hi in BANDS:
        for s in (0, 1):
            m = (age >= lo) & (age < hi) & (sex == s) & np.isfinite(x)
            if m.sum() < 100:
                continue
            r = {"feature": "RR_irregularity", "age_lo": lo, "age_hi": hi,
                 "sex": s, "n": int(m.sum())}
            r.update({f"p{p_}": float(v) for p_, v in zip(GRID, np.percentile(x[m], GRID))})
            q.append(r)
    pd.DataFrame(q).to_csv(f"{out}/quantiles_RR_irregularity.csv", index=False)

    # unadjusted marginal AUC, for the record -- NOT an incremental claim
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, x); auc = max(auc, 1 - auc)
    print(f"\n  visits {len(df):,}  HF {int(y.sum()):,} ({y.mean():.1%})")
    print(f"  bins kept {kept}, suppressed {dropped} (n<{MIN_CELL})")
    print(f"  RR_irregularity unadjusted AUC {auc:.4f}  <- ASSOCIATIVE ONLY:")
    print("     prevalent ED label; the AoU landmark analysis showed 96% of a")
    print("     prevalent-label effect was reverse causation. Do not quote this")
    print("     as a predictive number.")
    print(f"\n  -> {out}/marginal_RR_irregularity.csv")
    print(f"  -> {out}/quantiles_RR_irregularity.csv")
    print("  now:  python refit.py --tables tables --out model/arterialage.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MC-MED root directory")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    emit(build(a.data), a.out)
