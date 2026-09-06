#!/usr/bin/env python3
"""
refit.py — rebuild the model from exported aggregate tables.

    python refit.py --tables ./tables --out model/arterialage.json

`--tables` is the unzipped aou_training_tables.zip. Add MC-MED's
marginal_RR_irregularity.csv to the same folder and it is picked up too.

    python refit.py --demo          # synthesise tables matching the measured
                                    # AoU result, so everything is runnable now
"""
import argparse, json, os, numpy as np
from pathlib import Path
from arterialage import refit_all, save_artifact, RiskEngine

MEASURED = {  # from the AoU landmark-incident run; used only by --demo
 "prevalence":0.0166,"ATC95":{"mean":9000,"sd":3400,"beta":-0.145},
 "HR_RANGE":{"mean":41,"sd":9,"beta":-0.115},"HR_rest":{"mean":60.5,"sd":8.5,"beta":0.075},
}

def demo_tables(out="tables"):
    """Aggregate tables with the SAME SHAPE the workbench exports, calibrated so
    the refit reproduces the measured effect sizes. Replace with the real zip."""
    Path(out).mkdir(exist_ok=True); rng=np.random.default_rng(0)
    P=MEASURED["prevalence"]; b0=np.log(P/(1-P))
    bands=[(0,50),(50,60),(60,70),(70,200)]
    for f,spec in [(k,v) for k,v in MEASURED.items() if k!="prevalence"]:
        rows=["feature,age_lo,age_hi,sex,lo,hi,n_raw,n_weighted,events_weighted,rate"]
        edges=np.linspace(spec["mean"]-2.6*spec["sd"],spec["mean"]+2.6*spec["sd"],16)
        for lo,hi in bands:
            for s in (0,1):
                agec=(lo+min(hi,90))/2
                for i in range(15):
                    m=(edges[i]+edges[i+1])/2; z=(m-spec["mean"])/spec["sd"]
                    lg=b0+spec["beta"]*z+0.055*(agec-58)+0.30*s
                    p=1/(1+np.exp(-lg)); n=int(rng.integers(40,420))
                    rows.append(f"{f},{lo},{hi},{s},{edges[i]:.4f},{edges[i+1]:.4f},"
                                f"{n},{n},{p*n:.3f},{p:.6f}")
        Path(f"{out}/marginal_{f}.csv").write_text("\n".join(rows))
        g=[1,5,10,25,50,75,90,95,99]
        qr=["feature,age_lo,age_hi,sex,n,"+",".join(f"p{q}" for q in g)]
        for lo,hi in bands:
            for s in (0,1):
                from scipy.stats import norm as _n
                v=[spec["mean"]+spec["sd"]*_n.ppf(q/100) for q in g]
                qr.append(f"{f},{lo},{hi},{s},2000,"+",".join(f"{x:.4f}" for x in v))
        Path(f"{out}/quantiles_{f}.csv").write_text("\n".join(qr))
    cr=["age_lo,age_hi,sex,map_lo,map_hi,age_mid,map_mid,n_raw,n_weighted,events_weighted,rate"]
    for lo,hi in bands:
        for s in (0,1):
            for mq in [(70,85),(85,92),(92,99),(99,107),(107,140)]:
                agec=(lo+min(hi,90))/2; mm=(mq[0]+mq[1])/2
                lg=b0+0.069*(agec-58)+0.0012*(agec-58)**2/10+0.42*s+0.016*(mm-92)
                p=1/(1+np.exp(-lg)); n=int(rng.integers(300,1600))
                cr.append(f"{lo},{hi},{s},{mq[0]},{mq[1]},{agec:.1f},{mm:.1f},{n},{n},{p*n:.3f},{p:.6f}")
    Path(f"{out}/context_table.csv").write_text("\n".join(cr))
    print(f"demo tables -> {out}/  (SYNTHETIC — replace with aou_training_tables.zip)")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--tables",default="tables"); ap.add_argument("--out",default="model/arterialage.json")
    ap.add_argument("--demo",action="store_true")
    ap.add_argument("--cohort",default="All of Us")
    ap.add_argument("--label",default="incident HF (90d landmark)")
    a=ap.parse_args()
    if a.demo: demo_tables(a.tables)
    art=refit_all(a.tables,cohort=a.cohort,label_kind=a.label)
    art["synthetic_demo_tables"]=bool(a.demo)
    save_artifact(art,a.out)
    print(f"\nrefit from {len(art['sources'])} tables -> {a.out}")
    for f,m in art["marginals"].items():
        sw=max(m["knots_logit"])-min(m["knots_logit"])
        print(f"  marginal {f:22} {m['n_bins']:3d} bins  logit swing {sw:+.3f}  se {m['logit_se']:.3f}")
    if art.get("context"):
        c=art["context"]; print(f"  context  age {c['coef']['age']:+.4f}/yr  sex {c['coef']['sex']:+.3f}  "
                                f"MAP {c['coef']['MAP']:+.4f}/mmHg  prev {c['prevalence']:.4f}"
                                + ("  (design)" if art.get("design",{}).get("prevalence") else ""))
    if art.get("measured_results"):
        m=art["measured_results"]
        print(f"  measured  baseline {m['base_auroc']:.4f} -> full {m['full_auroc']:.4f}  "
              f"dAUROC {m['delta']:+.4f} {m['delta_ci']}")
        print(f"            HFpEF {m['hfpef_delta']:+.4f} {m['hfpef_ci']}  |  "
              f"within-age {m['within_age_auroc']:.3f}  |  decile OR {m['decile_or_base']:.0f}->{m['decile_or_full']:.0f}")
    for s in art.get("skipped",[]): print("  SKIPPED:",s)
    print("\n"+RiskEngine(a.out).describe())
