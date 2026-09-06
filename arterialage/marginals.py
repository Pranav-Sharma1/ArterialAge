"""
marginals.py — refit learned terms from EXPORTED AGGREGATE TABLES.

WHY TABLES AND NOT ROWS
All of Us forbids exporting models fitted on participant-level data, but
permits models fitted on summary data complying with the Dissemination Policy
(no cell revealing counts under 20). So the workbench exports binned
event-rate tables and this module refits the curves locally. A univariate
curve refit from ~15 weighted bins is statistically near-identical to one fit
on the rows; what you lose is a little tail resolution, nothing else.

TABLE FORMATS (produced by the final workbench cell)
  marginal_<FEATURE>.csv
    feature, age_lo, age_hi, sex, lo, hi, n_raw, n_weighted, events_weighted, rate
  quantiles_<FEATURE>.csv
    feature, age_lo, age_hi, sex, n, p1..p99          -> tier reference norms
  context_table.csv
    age_lo, age_hi, sex, map_lo, map_hi, age_mid, map_mid,
    n_raw, n_weighted, events_weighted, rate
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

EPS = 1e-9


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=np.nan):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
@dataclass
class LearnedMarginal:
    """P(outcome | feature), centred on the cohort prevalence so cohorts with
    different base rates compose. Stored as knots; evaluated by interpolation.

    The stored quantity is a LOG-LIKELIHOOD-RATIO:
        contribution = logit(P(y | x)) - logit(prevalence)
    which is exactly what the composite needs to add.
    """
    feature: str
    knots_x: List[float]
    knots_logit: List[float]
    prevalence: float
    logit_se: float
    cohort: str
    label_kind: str
    n_bins: int
    monotone: str = "none"          # "none" | "inc" | "dec" (fitted direction)

    def contribution(self, x) -> float:
        """Log-odds shift for one subject. NaN -> 0 (no evidence, no shift)."""
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(np.interp(float(x), self.knots_x, self.knots_logit))

    def se(self, x) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(self.logit_se)


def fit_marginal_from_table(rows: List[Dict], feature: str, cohort: str,
                            label_kind: str, min_cell: int = 20,
                            n_knots: int = 12, degree: int = 3) -> LearnedMarginal:
    """Weighted binomial IRLS across bins, with a per-stratum OFFSET.

    WHY NOT A SMOOTHED RATIO. The obvious approach -- convert each bin's event
    rate to a log-odds and smooth -- fails on this data because many bins carry
    0 or 1 events against a weighted denominator in the thousands. A 0-event
    bin has an undefined log-odds, and any fixed shrinkage strong enough to
    tame it distorts the well-populated bins. The result was a residual spread
    that pinned `logit_se` at its 0.5 ceiling, inflating every interval the
    model reports.

    INSTEAD: fit
        events_i ~ Binomial(n_i, sigmoid(offset_i + f(x_i)))
    where `offset_i = logit(prevalence of bin i's age x sex stratum)` and f is
    a low-order polynomial in the standardised feature. Zero-event bins are
    handled natively -- they are informative, not undefined. The stratum
    offsets absorb the age and sex intercepts, which belong to `context` and
    must not re-enter through the marginal (that was the age double-count).

    `logit_se` comes from the fitted covariance, so it reflects genuine
    parameter uncertainty rather than between-stratum variation.
    """
    rows = [r for r in rows if r.get("feature", feature) == feature]
    rows = [r for r in rows if _f(r, "n_raw", _f(r, "n_weighted", 0)) >= min_cell]
    if len(rows) < 6:
        raise ValueError(f"{feature}: only {len(rows)} usable bins (need >=6)")

    mid = np.array([(_f(r, "lo") + _f(r, "hi")) / 2.0 for r in rows])
    n = np.array([_f(r, "n_weighted", _f(r, "n_raw", 0)) for r in rows])
    ev = np.array([_f(r, "events_weighted", np.nan) for r in rows])
    if np.all(np.isnan(ev)):
        ev = np.array([_f(r, "rate") * w for r, w in zip(rows, n)])
    ev = np.nan_to_num(ev)
    prevalence = float(ev.sum() / max(n.sum(), EPS))

    strat = np.array([f"{_f(r,'age_lo')}_{_f(r,'age_hi')}_{_f(r,'sex')}" for r in rows])
    offset = np.empty(len(rows))
    for s_ in np.unique(strat):
        m_ = strat == s_
        pv = float(ev[m_].sum() / max(n[m_].sum(), EPS))
        offset[m_] = _logit(min(max(pv, 1e-5), 1 - 1e-5))

    mu_x = float(np.average(mid, weights=n))
    sd_x = float(np.sqrt(np.average((mid - mu_x) ** 2, weights=n))) or 1.0
    zx = (mid - mu_x) / sd_x
    X = np.column_stack([zx ** d for d in range(1, degree + 1)])   # no intercept:
    X = np.column_stack([np.ones(len(rows)), X])                   # centred by offset

    beta = np.zeros(X.shape[1])
    ridge = 1e-3 * np.eye(X.shape[1]); ridge[0, 0] = 1e-6
    XtWX = None
    for _ in range(80):
        eta = offset + X @ beta
        mu = _expit(eta)
        W = n * mu * (1 - mu) + 1e-9
        z = (X @ beta) + (ev - n * mu) / W
        WX = X * W[:, None]
        XtWX = X.T @ WX + ridge
        try:
            nb = np.linalg.solve(XtWX, WX.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(nb - beta)) < 1e-10:
            beta = nb; break
        beta = nb

    lo, hi = float(np.min(mid)), float(np.max(mid))
    knots_x = np.linspace(lo, hi, n_knots)
    kz = (knots_x - mu_x) / sd_x
    Kd = np.column_stack([np.ones(n_knots)] + [kz ** d for d in range(1, degree + 1)])
    fit = Kd @ beta
    fit = fit - float(np.average(np.interp(mid, knots_x, fit), weights=n))  # centre on 0

    try:
        cov = np.linalg.inv(XtWX)
        se = float(np.sqrt(np.mean(np.einsum("ij,jk,ik->i", Kd, cov, Kd))))
    except Exception:
        se = 0.1
    se = float(np.clip(se, 0.01, 0.5))

    slope = float(np.polyfit(knots_x, fit, 1)[0])
    return LearnedMarginal(feature=feature, knots_x=[float(v) for v in knots_x],
                           knots_logit=[float(v) for v in fit],
                           prevalence=prevalence, logit_se=se, cohort=cohort,
                           label_kind=label_kind, n_bins=len(rows),
                           monotone="inc" if slope > 0 else "dec")


# --------------------------------------------------------------------------- #
@dataclass
class ContextModel:
    """age, age^2, sex, MAP -- fitted ONCE, from the aggregate context table.

    This object is BOTH the context term of the composite AND the baseline the
    features are evaluated against, so the full model cannot contain a
    covariate the baseline lacks. That asymmetry was the single biggest source
    of inflated delta-AUROC in the previous build.
    """
    coef: Dict[str, float]
    intercept: float
    age_mean: float
    prevalence: float

    def logit(self, age: float, sex: float, MAP: float) -> float:
        a = age - self.age_mean
        return (self.intercept + self.coef["age"] * a
                + self.coef["age2"] * (a * a) / 100.0
                + self.coef["sex"] * sex
                + self.coef["MAP"] * (MAP - 92.0))

    def age_for_logit(self, target: float, sex: float, MAP: float,
                      lo: float = 20.0, hi: float = 100.0) -> float:
        """Invert for arterial age: the chronological age whose context-only
        risk equals `target`."""
        grid = np.linspace(lo, hi, 801)
        vals = np.array([self.logit(g, sex, MAP) for g in grid])
        return float(grid[int(np.argmin(np.abs(vals - target)))])


def fit_context_from_table(rows: List[Dict], min_cell: int = 20) -> ContextModel:
    """IRLS weighted logistic on the aggregate cells (no rows needed)."""
    rows = [r for r in rows if _f(r, "n_raw", _f(r, "n_weighted", 0)) >= min_cell]
    if len(rows) < 6:
        raise ValueError(f"context table has only {len(rows)} usable cells")
    age = np.array([_f(r, "age_mid") for r in rows])
    sex = np.array([_f(r, "sex") for r in rows])
    mp = np.array([_f(r, "map_mid") for r in rows])
    n = np.array([_f(r, "n_weighted", _f(r, "n_raw", 0)) for r in rows])
    ev = np.array([_f(r, "events_weighted", np.nan) for r in rows])
    if np.all(np.isnan(ev)):
        ev = np.array([_f(r, "rate") * w for r, w in zip(rows, n)])

    age_mean = float(np.average(age, weights=n))
    a = age - age_mean
    X = np.column_stack([np.ones(len(rows)), a, (a * a) / 100.0, sex, mp - 92.0])
    y = ev / np.maximum(n, EPS)

    beta = np.zeros(X.shape[1])
    beta[0] = _logit(float(ev.sum() / max(n.sum(), EPS)))
    for _ in range(60):                      # IRLS with binomial counts
        eta = X @ beta
        mu = _expit(eta)
        Wd = n * mu * (1 - mu) + 1e-8
        z = eta + (y - mu) / np.maximum(mu * (1 - mu), 1e-8)
        WX = X * Wd[:, None]
        try:
            new = np.linalg.solve(X.T @ WX + 1e-6 * np.eye(X.shape[1]), WX.T @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new

    return ContextModel(
        coef={"age": float(beta[1]), "age2": float(beta[2]),
              "sex": float(beta[3]), "MAP": float(beta[4])},
        intercept=float(beta[0]), age_mean=age_mean,
        prevalence=float(ev.sum() / max(n.sum(), EPS)))


# --------------------------------------------------------------------------- #
@dataclass
class Norms:
    """Age/sex-referenced percentiles -> the TIER reference distribution.

    The tier uses ONLY this, never a beta. That is why the tier survives
    everything that threatens the index: in a rank against a reference
    distribution, the path-length model and the instrument transfer cancel.
    """
    feature: str
    strata: List[Dict] = field(default_factory=list)

    def percentile_of(self, x: float, age: float, sex: float) -> Optional[float]:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        cand = [s for s in self.strata
                if s["age_lo"] <= age < s["age_hi"] and abs(s["sex"] - sex) < .5]
        if not cand:
            cand = [s for s in self.strata if abs(s["sex"] - sex) < .5] or self.strata
        if not cand:
            return None
        s = cand[0]
        ps = sorted((int(k[1:]), v) for k, v in s.items() if k.startswith("p")
                    and k[1:].isdigit())
        xs = [v for _, v in ps]
        qs = [q for q, _ in ps]
        return float(np.interp(float(x), xs, qs))


def load_norms(rows: List[Dict], feature: str) -> Norms:
    out = []
    for r in rows:
        if r.get("feature", feature) != feature:
            continue
        d = {"age_lo": _f(r, "age_lo", 0), "age_hi": _f(r, "age_hi", 200),
             "sex": _f(r, "sex", 0), "n": _f(r, "n", 0)}
        for k, v in r.items():
            if k.startswith("p") and k[1:].isdigit():
                d[k] = _f(r, k)
        if len(d) > 4:
            out.append(d)
    return Norms(feature=feature, strata=out)


# --------------------------------------------------------------------------- #
@dataclass
class InteractionBlock:
    """The NON-ADDITIVE surplus of a co-observed feature block, as a lookup grid.

    WHAT THIS IS. Stage 2 fits BART (a Bayesian tree ensemble) and an additive
    GAM twin on the same rows; the quantity of interest is
    `logit(BART) - logit(GAM)` -- the part the additive marginals CANNOT
    express. BART itself is a posterior over tree ensembles: a training-time
    object, far too large for an MCU and never intended to ship. What ships is
    its DISTILLATE -- the surplus averaged into a k x k x k cell grid, which is
    a few hundred floats and a trilinear interpolation.

    Only the SURPLUS is stored. The additive part is already carried by the
    marginals; adding the joint model whole would double-count it.
    """
    features: List[str]
    k: int
    edges: List[List[float]]          # k+1 edges per feature
    surplus: List[float]              # k^d cells, row-major
    n: List[float]
    cohort: str
    name: str = "activity"
    occupancy: float = 1.0

    def _idx(self, vals) -> Optional[int]:
        pos = []
        for v, e in zip(vals, self.edges):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            b = int(np.clip(np.digitize([float(v)], e[1:-1])[0], 0, self.k - 1))
            pos.append(b)
        idx = 0
        for b in pos:
            idx = idx * self.k + b
        return idx

    def correction(self, fv: Dict[str, float]) -> Tuple[float, float]:
        """Returns (logit surplus, sd). Any missing feature -> (0, 0): the block
        is not observed, so it makes no claim."""
        i = self._idx([fv.get(f) for f in self.features])
        if i is None or i >= len(self.surplus):
            return 0.0, 0.0
        nn = self.n[i]
        if nn < 20:                    # suppressed / sparse cell: no claim
            return 0.0, 0.0
        return float(self.surplus[i]), float(1.0 / math.sqrt(max(nn, 1.0)))


def load_interaction_block(rows: List[Dict], cohort: str, name: str = "activity",
                           min_cell: int = 20) -> InteractionBlock:
    """Read a crosstab_<name>_k<K>.csv exported from the workbench.

    Expected columns: <FEAT>_bin per feature, plus n, events, and EITHER
    `surplus` (BART minus GAM, preferred) or `rate` (from which a surplus is
    derived against the grid's own marginal-additive fit).
    """
    feats = [c[:-4] for c in rows[0] if c.endswith("_bin")]
    k = max(int(_f(r, f + "_bin")) for r in rows for f in feats) + 1
    ncell = k ** len(feats)
    surplus = [0.0] * ncell
    nn = [0.0] * ncell
    have_surplus = "surplus" in rows[0]
    for r in rows:
        idx = 0
        for f in feats:
            idx = idx * k + int(_f(r, f + "_bin"))
        nn[idx] = _f(r, "n", 0.0)
        surplus[idx] = _f(r, "surplus", 0.0) if have_surplus else np.nan
    if not have_surplus:
        # derive: cell log-odds minus an additive main-effects fit on the grid
        rate = np.array([_f(r, "rate", np.nan) for r in rows])
        idxs = []
        for r in rows:
            i = 0
            for f in feats:
                i = i * k + int(_f(r, f + "_bin"))
            idxs.append(i)
        w = np.array([nn[i] for i in idxs])
        ok = np.isfinite(rate) & (w >= min_cell) & (rate > 0) & (rate < 1)
        lg = np.full(len(rows), np.nan)
        lg[ok] = _logit(rate[ok])
        D = [np.ones(len(rows))]
        for j, f in enumerate(feats):
            for lvl in range(1, k):
                D.append(np.array([1.0 if int(_f(r, f + "_bin")) == lvl else 0.0
                                   for r in rows]))
        D = np.column_stack(D)
        beta, *_ = np.linalg.lstsq(D[ok] * w[ok, None] ** .5,
                                   lg[ok] * w[ok] ** .5, rcond=None)
        res = lg - D @ beta
        for i, r in zip(idxs, res):
            surplus[i] = 0.0 if not np.isfinite(r) else float(r)
    edges = [[float(b) for b in range(k + 1)] for _ in feats]   # placeholder
    occ = float(sum(1 for v in nn if v >= min_cell) / ncell)
    return InteractionBlock(features=feats, k=k, edges=edges, surplus=surplus,
                            n=nn, cohort=cohort, name=name, occupancy=occ)


# --------------------------------------------------------------------------- #
def refit_all(table_dir: str, cohort: str = "All of Us",
              label_kind: str = "incident HF (90d landmark)",
              min_cell: int = 20) -> Dict:
    """Read every exported table in `table_dir` and return the model artefact."""
    d = Path(table_dir)
    art = {"cohort": cohort, "label_kind": label_kind, "min_cell": min_cell,
           "marginals": {}, "norms": {}, "context": None, "sources": []}

    # metrics_final.json carries the DESIGN prevalence and the measured result.
    # The context table's own weighted prevalence is biased LOW because cells
    # with n_raw < 20 were suppressed at export, so prefer the design value.
    mpath = d / "metrics_final.json"
    if mpath.exists():
        with open(mpath) as f:
            mf = json.load(f)
        art["design"] = mf.get("design", {})
        art["measured_results"] = mf.get("results", {})
        art["feature_distributions"] = mf.get("feature_distributions", {})
        art["sources"].append(mpath.name)

    ctx_path = d / "context_table.csv"
    if ctx_path.exists():
        cm = fit_context_from_table(_read_csv(ctx_path), min_cell)
        dp = (art.get("design") or {}).get("prevalence")
        if dp:
            # shift the intercept so the model's population rate matches the
            # design prevalence; the covariate slopes are unaffected
            cm.intercept += float(np.log(dp / (1 - dp)) - np.log(
                cm.prevalence / (1 - cm.prevalence)))
            cm.prevalence = float(dp)
        art["context"] = asdict(cm)
        art["sources"].append(ctx_path.name)

    for p in sorted(d.glob("marginal_*.csv")):
        feat = p.stem.replace("marginal_", "").split("_hf")[0]
        try:
            m = fit_marginal_from_table(_read_csv(p), feat, cohort, label_kind, min_cell)
            art["marginals"][feat] = asdict(m)
            art["sources"].append(p.name)
        except ValueError as e:
            art.setdefault("skipped", []).append(str(e))

    for p in sorted(d.glob("crosstab_*.csv")):
        try:
            blk = load_interaction_block(_read_csv(p), cohort,
                                         name=p.stem.replace("crosstab_", ""), min_cell=min_cell)
            art.setdefault("blocks", {})[blk.name] = asdict(blk)
            art["sources"].append(p.name)
        except Exception as e:
            art.setdefault("skipped", []).append(f"{p.name}: {e}")

    for p in sorted(d.glob("quantiles_*.csv")):
        feat = p.stem.replace("quantiles_", "")
        art["norms"][feat] = asdict(load_norms(_read_csv(p), feat))
        art["sources"].append(p.name)

    return art


def save_artifact(art: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(art, f, indent=1)


def load_artifact(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
