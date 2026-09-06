"""
priors_v2.py — Stage-3 load-block priors, rebuilt.

WHAT CHANGED FROM priors.py, AND WHY
------------------------------------
1. Nocturnal_PTT_dipping and dPWV_reserve now have MARGINAL priors.
   Previously they existed ONLY inside INTERACTION_PRIORS with lam=0.0, so
   `mean = lam * z_i * z_j = 0` at every value. They contributed EXACTLY
   ZERO to the point estimate, always. That was the bug: they were encoded
   as modifiers of dtau rather than as measurements of their own physical
   quantity. They measure three DIFFERENT properties of one artery:

       dtau                   STRUCTURAL stiffness at operating pressure
       Nocturnal_PTT_dipping  PRESSURE-DEPENDENCE of stiffness (a_c) plus
                              autonomic circadian unloading
       dPWV_reserve           ACTIVE endothelial / vasodilatory reserve

2. Synergy priors are now ONE-SIDED (lam ~ HalfNormal). Claiming lam >= 0 is
   a far weaker claim than picking a value, and it is what the mechanism says:
   for these specific pairs the combination should be no BETTER than additive.
   A symmetric zero-mean prior instead widened the interval DOWNWARD, i.e. it
   made "much healthier than average" more plausible for a subject who is
   pathological on two axes. That is physiologically incoherent.

3. Total prior width is CAPPED and de-correlated. Seven pair terms at
   lam_sd=0.12 each, summed in quadrature at z=2, gave sd=1.27 -> a 95%
   interval of 0.6%-55% on a 4% base rate. The interval spanned the base
   rate: formally uninformative for exactly the subject the device exists to
   find. Two fixes, both stated: (a) lam_sd scales as 1/sqrt(n_pairs) because
   the pairs share features (dtau appears in four of them) and are therefore
   not independent draws of ignorance; (b) a hard ceiling on total prior sd.

PROVENANCE TAGS used on every constant below:
    MEASURED  a published effect estimate, cited, with its own CI
    DERIVED   arithmetic on a MEASURED value, derivation shown
    ANALOGY   magnitude borrowed from a related quantity; weakest class
    ASSUMED   a stated assumption with no source; sensitivity-tested
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# shared physics
# --------------------------------------------------------------------------- #
# MEASURED (canine gold standard, carried over from priors.py).
# Exponential tube law: E(P) = E0*exp(gamma*P) => c(P) = c0*exp(a_c*P),
# and since dtau ~ 1/c,  dtau(P) = dtau_ref * exp(-a_c*(P - P_ref)).
A_C_PRIOR = 0.0055          # mmHg^-1
P_REF_MMHG = 80.0           # ASSUMED diastolic reference

# MEASURED: 14 mm bilateral path-length difference. This resolves the spec's
# 7x length ambiguity in favour of the SHORT model (which predicted 2-5 ms).
# Consequence: dtau centre moves 18.35 -> 2.57 ms and SD 5.69 -> 0.80 ms. Every
# z-score, risk and tier is UNCHANGED, because the measurement and the centre
# scale together -- but the MEASUREMENT CHAIN gets 49x harder in variance terms.
DELTA_L_BAR_M = 0.014

# ASSUMED, and the single most important unknown left in the feature.
# dtau = dL / c. dL is a per-person ANATOMICAL asymmetry that nobody has
# measured. Assuming a population dL for an individual injects an error
#     dz ~ (dL_i/dL_bar - 1) / CV
# so a 15% anatomical spread is ~0.48 SD of z error -- enough to REVERSE the
# ordering of two people whose true stiffness differs by less than that.
# It is carried as extra width on the dtau prior, and `rank_guard()` refuses to
# assert an ordering inside it. The first pilot cohort with imaging replaces it.
DTAU_ANATOMICAL_CV = 0.15


class PriorUnavailable(RuntimeError):
    """Raised when a prior is requested but its required input is unsupplied."""


# --------------------------------------------------------------------------- #
# the generic marginal prior
# --------------------------------------------------------------------------- #
@dataclass
class MarginalPrior:
    """A literature-anchored log-odds marginal for ONE feature.

        contribution = beta * z(feature),   beta ~ N(beta_per_sd, beta_sd)

    Returns (mean_logit, sd_logit). A NaN feature returns (0, beta_sd): no
    shift, and the full prior width, which is the honest state of knowledge
    about an unmeasured subject.

    `sign` is +1 if HIGHER raw value = HIGHER risk, -1 if lower = higher.
    """
    feature: str
    pop_mean: float
    pop_sd: float
    beta_per_sd: float
    beta_sd: float
    sign: float                      # +1 or -1
    source: str                      # human-readable citation
    provenance: str                  # MEASURED | DERIVED | ANALOGY | ASSUMED
    units: str = ""

    def z(self, x: np.ndarray) -> np.ndarray:
        """Sign-oriented standardised deviation. Positive z = MORE pathological."""
        x = np.asarray(x, float)
        return self.sign * (x - self.pop_mean) / self.pop_sd

    def contribution(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ANALYTIC, not Monte Carlo. For beta ~ N(m,s) and fixed z, the
        product beta*z is exactly N(m*z, s*|z|)."""
        zz = self.z(x)
        known = np.isfinite(zz)
        z0 = np.nan_to_num(zz, nan=0.0)
        mean = np.where(known, self.beta_per_sd * z0, 0.0)
        sd = np.where(known, self.beta_sd * np.abs(z0), self.beta_sd)
        return mean, sd

    def describe(self) -> str:
        lo = self.beta_per_sd - 1.96 * self.beta_sd
        hi = self.beta_per_sd + 1.96 * self.beta_sd
        return (f"{self.feature:22s} [{self.provenance:8s}] "
                f"beta={self.beta_per_sd:+.3f}/SD  95%[{lo:+.3f},{hi:+.3f}]  "
                f"centre={self.pop_mean:.4g}{self.units} sd={self.pop_sd:.4g}")


# --------------------------------------------------------------------------- #
# 1. dtau — STRUCTURAL stiffness
# --------------------------------------------------------------------------- #
# MEASURED: carotid-femoral PWV -> hospitalisation for new-onset HF,
#   HR 1.29 per +1 m/s (95% CI 1.02-1.63).  ln(1.29) = 0.255; the CI implies
#   sd = 0.120 on the per-m/s log-hazard.
# DERIVED: carotid-path wave-speed population sd ~1.6 m/s
#   => 0.255 * 1.6 = 0.41 log-odds per SD.  Rounded DOWN to 0.35 because the
#   cfPWV -> inter-aural transfer is unvalidated; a prior must not exceed its
#   source. Width kept wide (0.25) because the source CI nearly touches null.
#
# BANNED: ePWV effect sizes (HR ~1.35/m/s). ePWV is COMPUTED from age and BP,
#   which is this model's own baseline. Using it would re-inject the baseline
#   under another name, inflate the index, and contribute ZERO delta-AUROC.
#   When the same quantity IS adjusted for age and SBP the slope collapses to
#   HR 1.13. `test_no_epwv_slope` pins this.
CFPWV_BETA_PER_MS = 0.255
CFPWV_BETA_SD = 0.120

DTAU_PRIOR = MarginalPrior(
    feature="dtau",
    pop_mean=DELTA_L_BAR_M / 5.45,               # DERIVED: dtau = dL/c at c=5.45 m/s
    pop_sd=0.31 * (DELTA_L_BAR_M / 5.45),        # ASSUMED: CV 0.31 from c range 4.5-8
    beta_per_sd=0.35,                            # DERIVED from CFPWV_BETA_PER_MS
    # widened for the anatomical dL unknown on top of the source CI: a
    # mis-assumed path length is indistinguishable from a stiffness difference
    beta_sd=float(np.sqrt(0.25**2 + (0.35 * DTAU_ANATOMICAL_CV / 0.31)**2)),
    sign=-1.0,                                   # SHORTER dtau = faster wave = stiffer
    source="cfPWV->new-onset HF, HR 1.29/m/s (95% CI 1.02-1.63), "
           "PWV-HF systematic review/meta-analysis",
    provenance="DERIVED",
    units=" s",
)

# --------------------------------------------------------------------------- #
# 2. Nocturnal_PTT_dipping — PRESSURE-DEPENDENCE + autonomic unloading
# --------------------------------------------------------------------------- #
# Feature definition:  ratio = dtau_night / dtau_day.
#   From the tube law, ratio = exp(a_c * dP_dip). A healthy dipper's BP falls
#   overnight, the artery unloads, the wave slows, dtau LENGTHENS -> ratio > 1.
#   A non-dipper's artery never unloads -> ratio ~ 1.
#
# MEASURED: study-level meta-analysis, 5 cohorts, 15,526 treated hypertensive
#   patients, 625 HF events. Percent nocturnal SBP fall: HR 0.81 (95% CI
#   0.75-0.88) per 1 SD increase, covariate-adjusted, robust to further
#   adjustment for clinic BP. Non-dipping status: HR 1.64 (95% CI 1.54-1.98).
#   Night-time BP outperformed clinic and daytime BP for predicting HF.
#
# DERIVED beta: risk RISES as the dip SHRINKS, so beta = -ln(0.81) = 0.211/SD.
#   This is natively per-SD on a HEART FAILURE endpoint with 625 events --
#   a better-sourced number than the dtau prior, which needed a unit
#   conversion off a hospitalisation endpoint.
# Source sd from the CI: (-ln(0.75) + ln(0.88))/2 / 1.96 = 0.041.
# TRANSFER WIDENING: BP-dip -> PTT-dip is unvalidated, so keep the mean and
#   widen the sd to 0.6*beta (same discipline as dtau), floored at source sd.
NOCT_BETA = 0.211
NOCT_BETA_SD = max(0.041, 0.6 * NOCT_BETA)       # = 0.127

# DERIVED population distribution: mean dip ratio = exp(a_c * dP_dip) with a
# normotensive MAP dip of ~12 mmHg -> 1.068. sd spans reverse-dippers (~0.99)
# to extreme dippers (~1.14). Override with device percentiles when they exist.
_NOCT_DP_DIP_MMHG = 12.0                          # ASSUMED normotensive MAP dip
NOCT_PRIOR = MarginalPrior(
    feature="Nocturnal_PTT_dipping",
    pop_mean=float(np.exp(A_C_PRIOR * _NOCT_DP_DIP_MMHG)),
    pop_sd=0.038,                                 # ASSUMED, sensitivity-tested
    beta_per_sd=NOCT_BETA,
    beta_sd=NOCT_BETA_SD,
    sign=-1.0,                                    # LOWER ratio (no dip) = higher risk
    source="Nocturnal SBP fall -> incident HF, HR 0.81/SD (95% CI 0.75-0.88); "
           "non-dipping HR 1.64 (1.54-1.98). 5-cohort meta-analysis, "
           "15,526 patients / 625 HF events.",
    provenance="MEASURED",
    units=" (night/day)",
)

# --------------------------------------------------------------------------- #
# 3. dPWV_reserve — ACTIVE endothelial / vasodilatory reserve
# --------------------------------------------------------------------------- #
# Feature definition (the PASSIVE prediction is subtracted, leaving the ACTIVE
# response):
#     reserve_index = ln(dtau_active / dtau_rest) + a_c * dP_activity
#   Pressure alone predicts dtau to shorten by exp(-a_c*dP). A healthy
#   endothelium dilates the conduit, so the artery stiffens LESS than pressure
#   predicts and the residual is POSITIVE. Endothelial dysfunction -> ~0.
#
# WHY THIS IS THE MOST HFpEF-SPECIFIC FEATURE IN THE SET: the Paulus-Tschope
#   paradigm makes microvascular endothelial dysfunction the mechanistic core
#   of HFpEF specifically (comorbidity -> inflammation -> reduced NO ->
#   reduced PKG -> titin hypophosphorylation -> stiff cardiomyocyte).
#   Arterial stiffness drives BOTH phenotypes; endothelial dysfunction is the
#   one that discriminates HFpEF from HFrEF.
#
# *** THE WEAKEST LINK IN THIS FILE. READ BEFORE PRESENTING. ***
# ANALOGY, not MEASURED. The right anchor exists -- MESA measured brachial FMD
# in a nested case-cohort of 3,496 participants without prevalent CVD (mean
# age 61), followed 149 incident HF events over a median 12 years, and split
# the analysis by HFpEF (LVEF>=45%) vs HFrEF -- but the per-SD hazard ratio was
# not extracted here, so the magnitude below is BORROWED from the dipping
# prior at a discount, with a width that spans zero.
#
# TO FIX: pull the per-SD HR from that paper and set the two lines marked
# `<-- REPLACE`. Nothing else changes. `sensitivity()` below reports how much
# the risk output moves across the plausible range in the meantime.
RESERVE_BETA = 0.20                               # <-- REPLACE with -ln(HR_perSD)
RESERVE_BETA_SD = 0.20                            # <-- REPLACE with the CI-derived sd

RESERVE_PRIOR = MarginalPrior(
    feature="dPWV_reserve",
    pop_mean=0.020,                               # ASSUMED, sensitivity-tested
    pop_sd=0.025,                                 # ASSUMED, sensitivity-tested
    beta_per_sd=RESERVE_BETA,
    beta_sd=RESERVE_BETA_SD,
    sign=-1.0,                                    # LOWER reserve = higher risk
    source="ANALOGY to the nocturnal-dipping slope, discounted. Intended "
           "anchor: MESA brachial FMD -> incident HF, nested case-cohort "
           "n=3,496, 149 events, median 12y, HFpEF/HFrEF-stratified. "
           "Per-SD HR NOT YET EXTRACTED.",
    provenance="ANALOGY",
    units=" (log residual)",
)

LOAD_PRIORS: Dict[str, MarginalPrior] = {
    p.feature: p for p in (DTAU_PRIOR, NOCT_PRIOR, RESERVE_PRIOR)
}


# --------------------------------------------------------------------------- #
# per-subject a_c: nocturnal dipping MEASURES the constant dtau normalisation
# ASSUMES. Use it when a nocturnal dP is available.
# --------------------------------------------------------------------------- #
def a_c_from_dipping(dip_ratio: float, dP_dip_mmHg: float) -> float:
    """a_c = ln(dtau_night/dtau_day) / dP_dip.

    `dtau_ref_from_dtau` currently applies the POPULATION constant A_C_PRIOR to
    pressure-normalise dtau. Nocturnal dipping measures that constant for THIS
    subject, which makes the primary feature more accurate as well as
    supplying a risk term of its own.

    CAVEAT, state it: the dip ratio confounds the MATERIAL component (a_c) with
    the AUTONOMIC one (whether BP fell at all). Separating them needs an
    INDEPENDENT nocturnal dP -- a PTT-derived BP cannot serve as the pressure
    axis for a PTT-vs-pressure fit without circularity. With no independent dP,
    use the composite ratio via NOCT_PRIOR and say so.
    """
    if not np.isfinite(dP_dip_mmHg) or abs(dP_dip_mmHg) < 1e-6:
        raise PriorUnavailable(
            "a_c requires an independent nocturnal pressure fall (mmHg). "
            "Without it, use NOCT_PRIOR on the composite dip ratio.")
    return float(np.log(dip_ratio) / dP_dip_mmHg)


def reserve_index(dtau_rest: float, dtau_active: float,
                  dP_activity_mmHg: float, a_c: float = A_C_PRIOR) -> float:
    """Active (endothelial) residual after removing the passive pressure effect."""
    return float(np.log(dtau_active / dtau_rest) + a_c * dP_activity_mmHg)


# --------------------------------------------------------------------------- #
# synergy priors — ONE-SIDED, de-correlated, capped
# --------------------------------------------------------------------------- #
@dataclass
class PairPrior:
    """Mechanistic synergy term for one feature pair:  contribution = lam*z_i*z_j.

    lam ~ HalfNormal(lam_sd), i.e. lam >= 0. That asserts only that the pair is
    no BETTER than additive -- which is what each mechanism note below claims --
    and never picks a magnitude. For a HalfNormal:
        E[lam]   = lam_sd * sqrt(2/pi)
        Var[lam] = lam_sd^2 * (1 - 2/pi)

    Contribution is applied ONLY when both features are deviant in the
    pathological direction (z_i > 0 and z_j > 0). A single deviant feature
    gets no synergy, and a protective value gets no protective synergy: the
    mechanism claims compounding of harm, not of benefit.
    """
    lam_sd: float
    mechanism: str
    one_sided: bool = True

    def contribution(self, z_i: np.ndarray, z_j: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        z_i, z_j = np.asarray(z_i, float), np.asarray(z_j, float)
        known = np.isfinite(z_i) & np.isfinite(z_j)
        p = np.nan_to_num(z_i, nan=0.0) * np.nan_to_num(z_j, nan=0.0)
        # SATURATION. The product z_i*z_j grows QUADRATICALLY, so without a
        # ceiling the one-sided synergy explodes for extreme subjects -- the
        # mirror image of the old symmetric-width bug. ASSUMED: synergy
        # saturates once both features are 2 SD out. Being 4 SD out on both
        # does not compound four times harder than being 2 SD out on both.
        p = np.clip(p, -SYNERGY_PRODUCT_CAP, SYNERGY_PRODUCT_CAP)
        if self.one_sided:
            # synergy only where BOTH are pathological
            active = known & (z_i > 0) & (z_j > 0)
            p = np.where(active, p, 0.0)
            mean = self.lam_sd * np.sqrt(2.0 / np.pi) * p
            sd = self.lam_sd * np.sqrt(1.0 - 2.0 / np.pi) * np.abs(p)
        else:
            mean = np.zeros_like(p)
            sd = self.lam_sd * np.abs(p)
        return (np.where(known, mean, 0.0),
                np.where(known, sd, self.lam_sd))


# DERIVED: base width = the uncertainty on the only directly-sourced slope in
# the model, so interaction ignorance is priced at the scale of a real effect.
LAM_SD_BASE = CFPWV_BETA_SD          # 0.120

# ASSUMED: synergy saturates at z_i = z_j = 2 SD.  See PairPrior.contribution.
SYNERGY_PRODUCT_CAP = 4.0

INTERACTION_PRIORS: Dict[Tuple[str, str], PairPrior] = {
    ("dtau", "HRR60"): PairPrior(LAM_SD_BASE,
        "Ventricular-arterial coupling. A stiff proximal aorta raises LV wall "
        "stress; blunted chronotropic recovery removes the compensatory "
        "reserve. Together they are the canonical HFpEF substrate."),
    ("dtau", "ATC95"): PairPrior(LAM_SD_BASE,
        "High stiffness with a preserved activity ceiling implies compensation; "
        "high stiffness with a collapsed ceiling is exhausted reserve."),
    ("dtau", "RR_irregularity"): PairPrior(LAM_SD_BASE,
        "AF removes the atrial kick a poorly-compliant ventricle depends on."),
    ("dPWV_reserve", "HRACS"): PairPrior(LAM_SD_BASE,
        "Vasodilatory reserve and chronotropic reserve are both sympathetically "
        "mediated; losing both signals global autonomic-vascular failure."),
    ("Nocturnal_PTT_dipping", "RR_irregularity"): PairPrior(LAM_SD_BASE,
        "Non-dipping and AF burden share a sustained-sympathetic-load driver."),
    ("Nocturnal_PTT_dipping", "dtau"): PairPrior(LAM_SD_BASE,
        "A structurally stiff vessel that also never unloads overnight has no "
        "functional reserve left."),
    ("dPWV_reserve", "dtau"): PairPrior(LAM_SD_BASE,
        "Structural stiffness with no active dilatory response: the artery "
        "cannot compensate at any operating point."),
}

# Hard ceiling on the TOTAL prior sd contributed to one prediction, in logits.
# ASSUMED, and the reason is stated: above ~0.6 the 95% interval on a 4% base
# rate already spans roughly 1.3%-11%, wide enough to express real ignorance.
# Beyond that the output stops being a risk estimate and becomes a shrug, and
# the perverse scaling kicks in (pair width grows as z^2 while the marginal
# signal grows as z, so the sicker the subject the less the model will say).
MAX_TOTAL_PRIOR_SD = 0.60

# DERIVED CEILING ON THE SYNERGY MEAN. Rule: an UNSOURCED mechanism claim must
# never move risk more than the WEAKEST SOURCED marginal in the model does at
# the same deviation. The weakest MEASURED slope is the dipping prior
# (0.211/SD), evaluated at z=2 -> 0.422 logits. Round to 0.40.
MAX_TOTAL_SYNERGY_LOGIT = 0.40

# DERIVED CEILING ON THE TOTAL PRIOR MEAN. The largest published arterial-
# stiffness -> incident-HFpEF effect is roughly 4x (top ePWV quartile,
# 12-year). The whole load block is unvalidated IN THIS INSTRUMENT, so it must
# not claim more than the literature's largest effect for a validated one.
# ln(5) = 1.61.
MAX_TOTAL_PRIOR_LOGIT = 1.60


def combine_prior_terms(means: List[np.ndarray], sds: List[np.ndarray],
                        n_pairs: int, cap: float = MAX_TOTAL_PRIOR_SD,
                        synergy_means: Optional[List[np.ndarray]] = None
                        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Sum prior means; combine sds in quadrature with a shared-feature
    de-correlation factor and a hard cap.

    De-correlation: the pair priors are NOT independent draws of ignorance --
    dtau appears in four of the seven -- so a plain quadrature sum
    over-counts. Scaling by 1/sqrt(n_pairs) is the standard correction for
    perfectly-shared uncertainty and is deliberately conservative.
    Returns (mean, sd, caps) where `caps` records which ceilings bound.
    """
    marg = np.sum(means, axis=0) if means else np.array([0.0])
    syn = np.sum(synergy_means, axis=0) if synergy_means else np.zeros_like(marg)
    syn_c = np.clip(syn, -MAX_TOTAL_SYNERGY_LOGIT, MAX_TOTAL_SYNERGY_LOGIT)
    total = marg + syn_c
    total_c = np.clip(total, -MAX_TOTAL_PRIOR_LOGIT, MAX_TOTAL_PRIOR_LOGIT)
    caps = {"synergy_capped": bool(np.any(np.abs(syn) > MAX_TOTAL_SYNERGY_LOGIT)),
            "total_mean_capped": bool(np.any(np.abs(total) > MAX_TOTAL_PRIOR_LOGIT))}
    if not sds:
        return total_c, np.zeros_like(total_c), {**caps, "sd_capped": False}
    var = np.sum(np.square(sds), axis=0)
    if n_pairs > 1:
        var = var / np.sqrt(n_pairs)
    sd = np.sqrt(var)
    caps["sd_capped"] = bool(np.any(sd > cap))
    return total_c, np.minimum(sd, cap), caps


# --------------------------------------------------------------------------- #
# sensitivity: how much does the risk output move across each prior's own CI?
# --------------------------------------------------------------------------- #
def sensitivity(feature: str, z: float = 2.0, base_prev: float = 0.04) -> Dict:
    """Risk at z, swept across the prior's own 95% interval on beta.
    Use this on any prior tagged ANALOGY or ASSUMED before quoting a number."""
    p = LOAD_PRIORS[feature]
    out = {}
    for lbl, b in (("lower_95", p.beta_per_sd - 1.96 * p.beta_sd),
                   ("point", p.beta_per_sd),
                   ("upper_95", p.beta_per_sd + 1.96 * p.beta_sd)):
        lg = np.log(base_prev / (1 - base_prev)) + b * z
        out[lbl] = float(1 / (1 + np.exp(-lg)))
    out["fold_range"] = out["upper_95"] / max(out["lower_95"], 1e-9)
    return out


def rank_guard(dtau_a: float, dtau_b: float, n_beats_a: int = 200,
               n_beats_b: int = 200, foot_sd_ms: float = 0.78) -> Dict:
    """Can these two subjects be ordered by arterial stiffness? Often: no.

    Two independent error sources sit between a dtau reading and a stiffness
    ranking:

      MEASUREMENT   per-beat dtau SD = foot_sd * sqrt(2); a steady-hold median
                    over n beats has SE = that / sqrt(n).
      ANATOMICAL    dtau = dL/c and dL is unknown per person. A DTAU_ANATOMICAL_CV
                    spread contributes dL_cv * dtau to each reading, and this
                    one does NOT average away with more beats -- it is a fixed
                    per-person offset.

    Returns can_rank=False whenever the gap is inside the combined 95% band.
    Reporting an order there would be a coin flip dressed as a measurement.
    """
    pb = foot_sd_ms * np.sqrt(2.0)
    se_a = np.sqrt((pb / np.sqrt(max(n_beats_a, 1))) ** 2
                   + (DTAU_ANATOMICAL_CV * dtau_a * 1000.0) ** 2)
    se_b = np.sqrt((pb / np.sqrt(max(n_beats_b, 1))) ** 2
                   + (DTAU_ANATOMICAL_CV * dtau_b * 1000.0) ** 2)
    diff = (dtau_a - dtau_b) * 1000.0
    se_d = float(np.sqrt(se_a ** 2 + se_b ** 2))
    mdd = 1.96 * se_d
    return {"diff_ms": float(diff), "se_ms": se_d,
            "min_detectable_diff_ms": float(mdd),
            "can_rank": bool(abs(diff) > mdd),
            "stiffer": (None if abs(diff) <= mdd else ("a" if diff < 0 else "b")),
            "anatomical_share": float((DTAU_ANATOMICAL_CV * dtau_a * 1000.0) ** 2
                                      / max(se_a ** 2, 1e-12)),
            "note": ("ordering supported" if abs(diff) > mdd else
                     "INDISTINGUISHABLE -- the gap is inside the combined "
                     "measurement + anatomical uncertainty")}


def prior_report() -> str:
    L = ["LOAD-BLOCK MARGINAL PRIORS (each moves the point estimate):"]
    for p in LOAD_PRIORS.values():
        L.append("  " + p.describe())
        L.append(f"      source: {p.source}")
    L.append("")
    L.append(f"SYNERGY PRIORS: {len(INTERACTION_PRIORS)} pairs, "
             f"lam ~ HalfNormal({LAM_SD_BASE}), one-sided, "
             f"total prior sd capped at {MAX_TOTAL_PRIOR_SD}")
    return "\n".join(L)
