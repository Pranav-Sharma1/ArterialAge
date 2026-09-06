#!/usr/bin/env python3
"""
train_rr_quick.py — RR_irregularity marginal, minimum viable, from MIMIC-IV.

    python train_rr_quick.py --data /path/to/mimic-iv --out tables/ --max-subjects 400
    python refit.py --tables tables --out model/arterialage.json

WHY THIS IS FAST
  It does NOT touch waveforms. MIMIC-IV's `chartevents` already stores a
  charted heart rate every ~1 h per ICU stay. RR irregularity is computed from
  the SUCCESSIVE-DIFFERENCE structure of that HR series -- coarse, but the AF
  signal (grossly irregular ventricular response) survives downsampling,
  because it is a variance property, not a beat-timing one.

  If you have waveform RR intervals, use --rr-csv instead: a two-column CSV of
  subject_id, rr_ms. That is strictly better and the rest is unchanged.

WHAT THIS BUYS AND WHAT IT DOESN'T
  MIMIC-IV is an ICU population with a PREVALENT ICD heart-failure label.
  The All of Us landmark analysis measured 96% of a prevalent-label effect to
  be reverse causation, so this marginal is ASSOCIATIVE, not predictive, and
  the artefact records that. It exists so the atrial-electrical domain can be
  SCORED rather than sit permanently NOT ASSESSED -- the tier needs a
  reference distribution, and the tier never uses beta.

  --max-subjects caps the cohort. 400 subjects is enough for ~40 usable bins
  at n>=20; go higher if the run is cheap.
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

MIN_CELL = 20
BANDS = [(0, 50), (50, 60), (60, 70), (70, 200)]
GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]
HR_ITEMIDS = [220045, 211]          # Metavision HR, CareVue HR


def find(root, *names):
    for n in names:
        for p in Path(root).rglob(n):
            return p
        for p in Path(root).rglob(n + ".gz"):
            return p
    return None


def from_chartevents(root, max_subjects):
    ce = find(root, "chartevents.csv"); adm = find(root, "patients.csv")
    dx = find(root, "diagnoses_icd.csv")
    if not (ce and adm and dx):
        raise SystemExit(f"need chartevents.csv, patients.csv, diagnoses_icd.csv under {root}")

    pat = pd.read_csv(adm, usecols=lambda c: c in
                      ("subject_id", "gender", "anchor_age"))
    d = pd.read_csv(dx, usecols=lambda c: c in ("subject_id", "icd_code"))
    d["icd_code"] = d.icd_code.astype(str).str.upper().str.replace(".", "", regex=False)
    hf_ids = set(d[d.icd_code.str.startswith(("I50", "428"))].subject_id.unique())

    keep = pat.subject_id.unique()
    rng = np.random.default_rng(0)
    # sample CASES and CONTROLS separately so a small cap still has events
    cases = np.array([s for s in keep if s in hf_ids])
    ctrls = np.array([s for s in keep if s not in hf_ids])
    n_case = min(len(cases), max_subjects // 2)
    n_ctrl = min(len(ctrls), max_subjects - n_case)
    want = set(np.concatenate([rng.choice(cases, n_case, replace=False),
                               rng.choice(ctrls, n_ctrl, replace=False)]).tolist())
    print(f"  sampling {n_case} HF + {n_ctrl} non-HF subjects")

    rows, seen = [], {}
    for chunk in pd.read_csv(ce, chunksize=2_000_000,
                             usecols=lambda c: c in ("subject_id", "charttime",
                                                     "itemid", "valuenum")):
        c = chunk[(chunk.itemid.isin(HR_ITEMIDS)) & (chunk.subject_id.isin(want))]
        c = c.dropna(subset=["valuenum"])
        c = c[(c.valuenum > 25) & (c.valuenum < 220)]
        for sid, g in c.groupby("subject_id"):
            seen.setdefault(sid, []).append(g[["charttime", "valuenum"]])
        done = sum(1 for v in seen.values() if sum(len(x) for x in v) >= 40)
        if done >= len(want):
            break

    for sid, parts in seen.items():
        g = pd.concat(parts).sort_values("charttime")
        if len(g) < 20:
            continue
        rr = 60000.0 / g.valuenum.values
        dd = np.diff(rr)
        rows.append({"subject_id": sid,
                     "RR_irregularity": float(np.sqrt(np.mean(dd ** 2)) / rr.mean()),
                     "n_obs": len(rr)})
    f = pd.DataFrame(rows)
    f = f.merge(pat, on="subject_id", how="left")
    f["hf"] = f.subject_id.isin(hf_ids).astype(int)
    f["age"] = pd.to_numeric(f.get("anchor_age"), errors="coerce").fillna(65)
    f["sex"] = (f.get("gender", pd.Series("F", index=f.index))
                .astype(str).str.upper().str[0] == "M").astype(float)
    return f


def from_rr_csv(path):
    df = pd.read_csv(path)
    sid = next(c for c in df.columns if "subject" in c.lower() or "id" in c.lower())
    rrc = next(c for c in df.columns if "rr" in c.lower() or "interval" in c.lower())
    rows = []
    for s, g in df.groupby(sid):
        rr = pd.to_numeric(g[rrc], errors="coerce").dropna().values
        rr = rr[(rr > 250) & (rr < 2500)]
        if len(rr) < 30:
            continue
        d = np.diff(rr)
        rows.append({"subject_id": s,
                     "RR_irregularity": float(np.sqrt(np.mean(d ** 2)) / rr.mean()),
                     "n_obs": len(rr)})
    return pd.DataFrame(rows)


def emit(df, out, label, cohort):
    Path(out).mkdir(parents=True, exist_ok=True)
    lo, hi = df.RR_irregularity.quantile([.01, .99])
    df["RR_irregularity"] = df.RR_irregularity.clip(lo, hi)
    x, y = df.RR_irregularity.values, df.hf.values
    age, sex = df.age.values, df.sex.values

    rows, kept, sup = [], 0, 0
    for a_lo, a_hi in BANDS:
        for s in (0, 1):
            m = (age >= a_lo) & (age < a_hi) & (sex == s) & np.isfinite(x)
            if m.sum() < MIN_CELL * 2:
                continue
            nb = max(4, min(15, m.sum() // MIN_CELL))
            ed = np.unique(np.quantile(x[m], np.linspace(0, 1, nb + 1)))
            if len(ed) < 3:
                continue
            idx = np.clip(np.digitize(x[m], ed[1:-1]), 0, len(ed) - 2)
            for b in range(len(ed) - 1):
                sel = idx == b
                if sel.sum() < MIN_CELL:
                    sup += 1; continue
                kept += 1
                rows.append(dict(feature="RR_irregularity", age_lo=a_lo, age_hi=a_hi, sex=s,
                                 lo=float(ed[b]), hi=float(ed[b + 1]),
                                 n_raw=int(sel.sum()), n_weighted=float(sel.sum()),
                                 events_weighted=float(y[m][sel].sum()),
                                 rate=float(y[m][sel].mean())))
    pd.DataFrame(rows).to_csv(f"{out}/marginal_RR_irregularity.csv", index=False)

    q = []
    for a_lo, a_hi in BANDS:
        for s in (0, 1):
            m = (age >= a_lo) & (age < a_hi) & (sex == s) & np.isfinite(x)
            if m.sum() < 40:
                continue
            r = {"feature": "RR_irregularity", "age_lo": a_lo, "age_hi": a_hi,
                 "sex": s, "n": int(m.sum())}
            r.update({f"p{p_}": float(v) for p_, v in zip(GRID, np.percentile(x[m], GRID))})
            q.append(r)
    pd.DataFrame(q).to_csv(f"{out}/quantiles_RR_irregularity.csv", index=False)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, x); auc = max(auc, 1 - auc)
    print(f"\n  subjects {len(df):,}  HF {int(y.sum()):,} ({y.mean():.1%})")
    print(f"  median RR_irregularity: HF {np.median(x[y==1]):.4f} vs non-HF {np.median(x[y==0]):.4f}")
    print(f"  bins kept {kept}, suppressed {sup} (n<{MIN_CELL})")
    print(f"  unadjusted AUC {auc:.4f}")
    print(f"\n  ASSOCIATIVE ONLY — {cohort} is a {label} population with a PREVALENT")
    print("  ICD label. Do not quote this as predictive. It exists so the")
    print("  atrial-electrical domain can be SCORED instead of NOT ASSESSED.")
    print(f"\n  -> {out}/marginal_RR_irregularity.csv")
    print(f"  -> {out}/quantiles_RR_irregularity.csv")
    print("  next:  python refit.py --tables tables --out model/arterialage.json")
    if kept < 12:
        print(f"\n  ! only {kept} bins — raise --max-subjects")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--rr-csv")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--max-subjects", type=int, default=400)
    a = ap.parse_args()
    if a.rr_csv:
        df = from_rr_csv(a.rr_csv)
        if "hf" not in df:
            raise SystemExit("--rr-csv needs an 'hf' column (0/1) and 'age','sex'")
        emit(df, a.out, "waveform", "RR-interval CSV")
    elif a.data:
        emit(from_chartevents(a.data, a.max_subjects), a.out, "ICU", "MIMIC-IV")
    else:
        raise SystemExit("give --data <mimic-iv root> or --rr-csv <file>")
