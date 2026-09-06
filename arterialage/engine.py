"""
engine.py — the single source of truth for a prediction.

The CLI, the HTTP server and therefore the GUI all call `RiskEngine.predict`.
There is no second implementation of the maths anywhere, so the GUI cannot
drift from the model.

    logit(index) = context(age, sex, MAP)      <- learned, counted ONCE
                 + SUM learned marginals        <- refit from AoU tables
                 + SUM load-block priors        <- literature-sourced
                 + SUM one-sided synergy priors <- mechanism, magnitude-free

Separately and independently, an HMOD TIER is computed from age/sex
percentiles. The tier never uses a beta, which is why it survives the
uncertainties that make the index's absolute percentage unreliable.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .marginals import (ContextModel, InteractionBlock, LearnedMarginal,
                        Norms, load_artifact)
from .priors import (LOAD_PRIORS, INTERACTION_PRIORS, MAX_TOTAL_PRIOR_SD,
                     DTAU_ANATOMICAL_CV, combine_prior_terms, prior_report,
                     rank_guard)

# feature -> HMOD domain
DOMAIN = {
    "dtau": "arterial_load", "Nocturnal_PTT_dipping": "arterial_load",
    "dPWV_reserve": "arterial_load",
    "HR_RANGE": "exertional_reserve", "HRR60": "exertional_reserve",
    "HRACS": "exertional_reserve", "HR_rest": "exertional_reserve",
    "RR_irregularity": "atrial_electrical",
    "ATC95": "capacity",
}
DOMAIN_LABEL = {"arterial_load": "Arterial load",
                "exertional_reserve": "Exertional reserve",
                "atrial_electrical": "Atrial-electrical",
                "capacity": "Functional capacity"}
TIER_DOMAINS = ["arterial_load", "exertional_reserve", "atrial_electrical"]

# HIGHER raw value = worse?  -1 means LOWER is worse.
SIGN = {"dtau": -1, "Nocturnal_PTT_dipping": -1, "dPWV_reserve": -1,
        "HR_RANGE": -1, "HRR60": -1, "HRACS": -1, "ATC95": -1,
        "HR_rest": +1, "RR_irregularity": +1}

ABNORMAL_PCT = 95.0          # ESC/ESH cut-point convention


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


class RiskEngine:
    def __init__(self, artifact_path: str, base_prevalence: Optional[float] = None,
                 use_load_priors: bool = True, use_synergy: bool = True,
                 demo_mode: bool = False):
        art = load_artifact(artifact_path)
        self.art = art
        self.context = ContextModel(**art["context"]) if art.get("context") else None
        self.marginals = {k: LearnedMarginal(**v) for k, v in art.get("marginals", {}).items()}
        self.norms = {k: Norms(**v) for k, v in art.get("norms", {}).items()}
        self.blocks = {k: InteractionBlock(**v) for k, v in art.get("blocks", {}).items()}
        self._add_provisional_load_norms()
        self.base_prev = float(base_prevalence if base_prevalence is not None
                               else (self.context.prevalence if self.context else 0.04))
        self.use_load_priors = use_load_priors
        self.use_synergy = use_synergy
        self.demo_mode = demo_mode

    def _add_provisional_load_norms(self):
        """PROVISIONAL reference percentiles for the load block, from each
        prior's STATED population distribution.

        No device cohort exists yet, so these are not measured norms. They are
        flagged `provisional` everywhere they surface, and the first pilot
        percentiles replace them. Without them the arterial domain is
        permanently NOT ASSESSED and the tier can never exceed 2 -- which is
        correct but makes the mechanism untestable end to end.
        """
        from scipy.stats import norm as _sn
        grid = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.provisional = []
        for feat, pr in LOAD_PRIORS.items():
            if feat in self.norms:
                continue
            vals = {f"p{q}": float(pr.pop_mean + pr.pop_sd * _sn.ppf(q / 100.0))
                    for q in grid}
            self.norms[feat] = Norms(feature=feat, strata=[
                {"age_lo": 0.0, "age_hi": 200.0, "sex": s_, "n": 0.0, **vals}
                for s_ in (0.0, 1.0)])
            self.provisional.append(feat)

    # ------------------------------------------------------- demo tier -----
    def _demo_tier(self, fv: Dict[str, float]) -> Dict:
        """DEMO MODE: tier graded on ARTERIAL LOAD SEVERITY, 0-2.

        The clinical tier counts damaged domains out of three. In a live
        setting only some domains are measurable, so a count reads as a mild
        finding when it is actually a full finding on what was measured. This
        grades SEVERITY inside the arterial-load domain instead -- while still
        REPORTING all three domains, so nothing is hidden.

        Ceiling of 2 is structural, not cosmetic: Tier 3 means ventricular-
        arterial COUPLING failure, a statement about two domains together.
        One domain cannot establish it at any severity.

            below 84th    -> 0   within normal range      (z < 1.0)
            84th - 97.7th -> 1   elevated arterial load   (1.0 <= z < 2.0)
            97.7th and up -> 2   marked stiffening        (z >= 2.0)
        """
        T1_PCT, T2_PCT = 84.0, 97.7
        age, sex = fv.get("age", 60.0), fv.get("sex", 0.0)

        doms: Dict[str, Dict] = {}
        for feat, dom in DOMAIN.items():
            if dom not in TIER_DOMAINS:
                continue
            slot = doms.setdefault(dom, {"assessed": False, "features": [],
                                         "worst_pct": None, "driver": None})
            x = fv.get(feat)
            if _nan(x):
                slot["features"].append({"feature": feat, "state": "not measured"})
                continue
            n = self.norms.get(feat)
            pct = n.percentile_of(x, age, sex) if n else None
            if pct is None:
                slot["features"].append({"feature": feat, "value": x,
                                         "state": "no reference norms"})
                continue
            sev = pct if SIGN.get(feat, 1) > 0 else 100.0 - pct
            slot["assessed"] = True
            if slot["worst_pct"] is None or sev > slot["worst_pct"]:
                slot["worst_pct"], slot["driver"] = sev, feat
            slot["features"].append({
                "feature": feat, "value": x, "severity_pct": round(sev, 1),
                "provisional_norms": feat in getattr(self, "provisional", []),
                "state": ("marked" if sev >= T2_PCT else
                          "elevated" if sev >= T1_PCT else "normal")})

        load = doms.get("arterial_load", {})
        worst = load.get("worst_pct")
        if worst is None:
            tier, label = 0, "Arterial load not assessed"
        elif worst >= T2_PCT:
            tier, label = 2, "Marked arterial stiffening"
        elif worst >= T1_PCT:
            tier, label = 1, "Elevated arterial load"
        else:
            tier, label = 0, "Within normal range"

        def band(p):
            if p is None:
                return "not assessed"
            return "marked" if p >= T2_PCT else "elevated" if p >= T1_PCT else "normal"

        out_domains = {}
        for d in TIER_DOMAINS:
            v = doms.get(d, {"assessed": False, "features": [], "worst_pct": None,
                             "driver": None})
            out_domains[DOMAIN_LABEL[d]] = {
                **v, "state": band(v.get("worst_pct")),
                "severity_pct": v.get("worst_pct"),
                "score": (0 if v.get("worst_pct") is None else
                          2 if v["worst_pct"] >= T2_PCT else
                          1 if v["worst_pct"] >= T1_PCT else 0)}

        n_as = sum(1 for d in out_domains.values() if d["assessed"])
        return {
            "tier": tier, "max_tier": 2, "demo_mode": True, "label": label,
            "driver": load.get("driver"),
            "worst_severity_pct": None if worst is None else round(worst, 1),
            "band_pcts": [T1_PCT, T2_PCT],
            "n_abnormal": sum(1 for d in out_domains.values() if d["score"] >= 1),
            "n_assessed": n_as, "tier_is_floor": n_as < len(TIER_DOMAINS),
            "va_coupling_rule": False,
            "provisional_norms": sorted(getattr(self, "provisional", [])),
            "note": ("Tier graded on arterial-load severity, ceiling 2. Tier 3 "
                     "means ventricular-arterial coupling failure and needs two "
                     "domains, so it cannot be reached from arterial "
                     "measurements alone."),
            "domains": out_domains,
        }

    # ---------------------------------------------------------------- tier --
    def tier(self, fv: Dict[str, float]) -> Dict:
        if self.demo_mode:
            return self._demo_tier(fv)
        age, sex = fv.get("age", 60.0), fv.get("sex", 0.0)
        dom: Dict[str, Dict] = {}
        for feat, d in DOMAIN.items():
            if d not in TIER_DOMAINS:
                continue
            slot = dom.setdefault(d, {"assessed": False, "abnormal": False,
                                      "features": [], "worst_pct": None})
            x = fv.get(feat)
            if _nan(x):
                slot["features"].append({"feature": feat, "state": "not measured"})
                continue
            n = self.norms.get(feat)
            pct = n.percentile_of(x, age, sex) if n else None
            if pct is None:
                slot["features"].append({"feature": feat, "value": x,
                                         "state": "NOT SCOREABLE (no reference norms)"})
                continue
            slot["assessed"] = True
            prov = feat in getattr(self, "provisional", [])
            sev = pct if SIGN.get(feat, 1) > 0 else 100.0 - pct
            ab = sev >= ABNORMAL_PCT
            slot["abnormal"] |= ab
            slot["worst_pct"] = sev if slot["worst_pct"] is None else max(slot["worst_pct"], sev)
            slot["features"].append({"feature": feat, "value": x,
                                     "severity_pct": round(sev, 1),
                                     "provisional_norms": prov,
                                     "state": "abnormal" if ab else "normal"})

        n_ab = sum(1 for d in TIER_DOMAINS if dom.get(d, {}).get("abnormal"))
        n_as = sum(1 for d in TIER_DOMAINS if dom.get(d, {}).get("assessed"))
        va = bool(dom.get("arterial_load", {}).get("abnormal")
                  and dom.get("exertional_reserve", {}).get("abnormal"))
        tier = 3 if va else n_ab
        return {
            "tier": tier, "n_abnormal": n_ab, "n_assessed": n_as,
            "tier_is_floor": n_as < len(TIER_DOMAINS),
            "provisional_norms": sorted(getattr(self, "provisional", [])),
            "va_coupling_rule": va,
            "label": ["No damaged domain", "One damaged domain",
                      "Two damaged domains",
                      "Ventricular-arterial coupling failure"][tier],
            "domains": {DOMAIN_LABEL[d]: {
                **dom.get(d, {"assessed": False, "abnormal": False, "features": []}),
                "state": ("not assessed" if not dom.get(d, {}).get("assessed")
                          else "abnormal" if dom.get(d, {}).get("abnormal") else "normal")
            } for d in TIER_DOMAINS},
        }

    # --------------------------------------------------------------- index --
    def predict(self, fv: Dict[str, float]) -> Dict:
        age = float(fv.get("age", 60.0))
        sex = float(fv.get("sex", 0.0))
        MAP = float(fv.get("MAP", 92.0))
        terms: List[Dict] = []

        if self.context is not None:
            total = self.context.logit(age, sex, MAP)
            terms.append({"term": "context (age, sex, MAP)", "kind": "context",
                          "logit": round(total, 4), "sd": 0.0,
                          "note": "learned, counted ONCE - this is also the baseline"})
        else:
            total = math.log(self.base_prev / (1 - self.base_prev))
            terms.append({"term": "base prevalence", "kind": "context",
                          "logit": round(total, 4), "sd": 0.0,
                          "note": "no context model fitted"})
        var = 0.0

        for feat, m in self.marginals.items():
            x = fv.get(feat)
            c = m.contribution(x)
            s = m.se(x)
            observed = not _nan(x)
            if observed:
                total += c
                var += s * s
            terms.append({"term": f"marginal({feat})", "kind": "learned",
                          "logit": round(c, 4), "sd": round(s, 4),
                          "observed": observed, "cohort": m.cohort,
                          "label": m.label_kind,
                          "note": "learned from data" if observed else "NOT MEASURED"})

        # ---- learned NON-ADDITIVE block corrections (BART distillate) ----
        for bname, blk in self.blocks.items():
            c, s_ = blk.correction(fv)
            if c != 0.0 or s_ != 0.0:
                total += c
                var += s_ * s_
            terms.append({
                "term": f"interaction({bname})", "kind": "learned",
                "logit": round(c, 4), "sd": round(s_, 4),
                "observed": c != 0.0 or s_ != 0.0,
                "cohort": blk.cohort,
                "note": (f"non-additive surplus, BART-distilled {blk.k}^{len(blk.features)} "
                         f"grid over {'+'.join(blk.features)}")
                        if (c != 0.0 or s_ != 0.0) else "block not fully observed"})

        pm, ps, sm, zmap = [], [], [], {}
        if self.use_load_priors:
            for feat, pr in LOAD_PRIORS.items():
                x = fv.get(feat)
                arr = np.array([np.nan if _nan(x) else float(x)])
                z = pr.z(arr)
                zmap[feat] = z
                mn, sd = pr.contribution(arr)
                pm.append(mn); ps.append(sd)
                terms.append({
                    "term": f"load_prior({feat})", "kind": "prior",
                    "provenance": pr.provenance, "source": pr.source,
                    "z": None if not np.isfinite(z[0]) else round(float(z[0]), 2),
                    "logit": round(float(mn[0]), 4), "sd": round(float(sd[0]), 4),
                    "observed": bool(np.isfinite(z[0])),
                    "note": f"PRIOR ({pr.provenance}) - not learned from any cohort"})

        npairs = 0
        if self.use_synergy:
            for (a, b), pp in INTERACTION_PRIORS.items():
                za, zb = zmap.get(a), zmap.get(b)
                if za is None or zb is None:
                    continue
                mn, sd = pp.contribution(za, zb)
                if float(mn[0]) == 0.0 and float(sd[0]) == 0.0:
                    continue
                npairs += 1
                sm.append(mn); ps.append(sd)
                if float(mn[0]) != 0.0:
                    terms.append({"term": f"synergy({a} x {b})", "kind": "prior",
                                  "provenance": "MECHANISM", "source": pp.mechanism,
                                  "logit": round(float(mn[0]), 4),
                                  "sd": round(float(sd[0]), 4), "observed": True,
                                  "note": "one-sided, magnitude-free"})

        caps = {}
        if pm or sm:
            p_mean, p_sd, caps = combine_prior_terms(pm, ps, max(npairs, 1),
                                                     synergy_means=sm)
            total += float(p_mean[0])
            var += float(p_sd[0]) ** 2

        sd_tot = math.sqrt(var)
        risk = 1 / (1 + math.exp(-total))
        lo = 1 / (1 + math.exp(-(total - 1.96 * sd_tot)))
        hi = 1 / (1 + math.exp(-(total + 1.96 * sd_tot)))

        aa = None
        if self.context is not None:
            aa_age = self.context.age_for_logit(total, sex, MAP)
            aa = {"arterial_age": round(aa_age, 1), "chronological_age": age,
                  "gap_years": round(aa_age - age, 1)}

        learned_abs = sum(abs(t["logit"]) for t in terms if t["kind"] == "learned")
        prior_abs = sum(abs(t["logit"]) for t in terms if t["kind"] == "prior")

        # ---- ranking guard: can this subject be ordered against another? ----
        dt = fv.get("dtau")
        rankg = None
        if not _nan(dt):
            nb = int(fv.get("n_beats", 200) or 200)
            pb = 0.78 * math.sqrt(2.0)
            se_meas = pb / math.sqrt(max(nb, 1))
            se_anat = DTAU_ANATOMICAL_CV * float(dt) * 1000.0
            se = math.sqrt(se_meas ** 2 + se_anat ** 2)
            rankg = {
                "dtau_ms": float(dt) * 1000.0,
                "se_ms": se, "n_beats": nb,
                "se_measurement_ms": se_meas, "se_anatomical_ms": se_anat,
                "anatomical_share": se_anat ** 2 / max(se ** 2, 1e-12),
                "min_separation_ms": 1.96 * math.sqrt(2.0) * se,
                "band_lo_ms": float(dt) * 1000.0 - 1.96 * se,
                "band_hi_ms": float(dt) * 1000.0 + 1.96 * se,
                "note": ("dtau = dL/c and dL is a per-person anatomy nobody has "
                         "measured. Two subjects closer than min_separation_ms "
                         "CANNOT be ordered by stiffness -- the gap is inside "
                         "the anatomical unknown, which does not shrink with "
                         "more beats."),
            }

        return {
            "rank_guard": rankg,
            "risk_index": risk, "lo": lo, "hi": hi,
            "logit": total, "logit_sd": sd_tot,
            "base_prevalence": self.base_prev,
            "multiple_of_population": risk / self.base_prev,
            "calibrated": False,
            "prior_caps": caps,
            "prior_share": (prior_abs / (learned_abs + prior_abs)
                            if (learned_abs + prior_abs) > 0 else 0.0),
            "arterial_age": aa,
            "tier": self.tier(fv),
            "provenance": terms,
            "note": ("Risk INDEX, not a calibrated probability. The MULTIPLE of "
                     "population rate, the ranking and the tier are supported; "
                     "the absolute percentage is not, until a cohort co-observes "
                     "every feature with the outcome."),
        }

    def describe(self) -> str:
        L = [f"cohort            : {self.art.get('cohort')}",
             f"label             : {self.art.get('label_kind')}",
             f"base prevalence   : {self.base_prev:.4f}",
             f"context fitted    : {'yes' if self.context else 'NO'}",
             f"demo mode         : {self.demo_mode}",
             f"learned marginals : {sorted(self.marginals)}",
             f"learned blocks    : "
             + (", ".join(f"{k} ({b.k}^{len(b.features)} cells, "
                          f"{100*b.occupancy:.0f}% occupied)"
                          for k, b in self.blocks.items()) or "none"),
             f"reference norms   : {sorted(self.norms)}", "", prior_report()]
        return "\n".join(L)
