#!/usr/bin/env python3
"""
predict.py — score one or many feature vectors from the terminal.

    # inline
    python predict.py --set age=71 sex=1 MAP=104 dtau=0.0072 ATC95=5200 HR_RANGE=31

    # from a file
    python predict.py --json examples/subject_hfpef.json

    # FROM YOUR SIGNAL-PROCESSING PIPELINE  (this is the integration point)
    python sigproc.py --emit-features | python predict.py --stdin --format json

    # many subjects, one JSON array in / one array out
    cat batch.json | python predict.py --stdin --format json

FEATURE VECTOR: a flat JSON object. Missing or null features are treated as
NOT MEASURED -- no shift, full prior width, and the tier becomes a floor.
Aggregate your signal-processing output over many beats/nights BEFORE calling
this: one number per feature per subject.
"""
import argparse, json, sys, math
from arterialage import RiskEngine, DOMAIN, LOAD_PRIORS

FEATURES = ["age","sex","MAP","dtau","Nocturnal_PTT_dipping","dPWV_reserve",
            "ATC95","HR_RANGE","HR_rest","HRR60","HRACS","RR_irregularity"]

def parse_set(pairs):
    fv={}
    for p in pairs:
        if "=" not in p: raise SystemExit(f"bad --set '{p}', expected key=value")
        k,v=p.split("=",1)
        fv[k]=None if v.lower() in ("","nan","null","none","na") else float(v)
    return fv

def bar(v, lo=-1.2, hi=1.2, w=34):
    mid=w//2; n=int(round(abs(v)/max(hi,1e-9)*mid)); n=min(n,mid)
    s=[" "]*w
    if v>=0: s[mid:mid+n]="#"*n
    else:    s[mid-n:mid]="#"*n
    s[mid]="|"
    return "".join(s)

def render(r, fv, verbose=True):
    L=[]
    t=r["tier"]
    L.append("="*74)
    if t.get("demo_mode"): L.append("  ** DEMO MODE — arterial load only, tier capped at 2 **")
    L.append(f"  TIER {t['tier']}{'/2' if t.get('demo_mode') else ''}   {t['label']}"
             + ("   [FLOOR - domain not assessed]" if t["tier_is_floor"] else ""))
    L.append(f"  RISK INDEX  {100*r['risk_index']:.2f}%   "
             f"95% [{100*r['lo']:.2f}%, {100*r['hi']:.2f}%]")
    L.append(f"  {r['multiple_of_population']:.2f}x the {100*r['base_prevalence']:.2f}% "
             f"population rate    <- the supported claim")
    if r.get("arterial_age"):
        a=r["arterial_age"]
        L.append(f"  ARTERIAL AGE {a['arterial_age']:.0f} vs chronological "
                 f"{a['chronological_age']:.0f}   gap {a['gap_years']:+.0f} years")
    L.append(f"  calibrated: {r['calibrated']}   prior share of |logit|: "
             f"{100*r['prior_share']:.0f}%")
    L.append("="*74)
    if verbose:
        L.append("\n  WHERE THE RISK CAME FROM        [#=learned  o=prior]")
        for tm in r["provenance"]:
            if tm["logit"]==0 and not tm.get("observed",True):
                L.append(f"    {tm['term']:34} {'not measured':>10}")
                continue
            if tm["logit"]==0 and tm["kind"]!="context": continue
            g=bar(tm["logit"])
            if tm["kind"]=="prior": g=g.replace("#","o")
            L.append(f"    {tm['term']:34} {tm['logit']:+7.3f} {g}")
        L.append("\n  DOMAINS")
        for d,v in t["domains"].items():
            sev=f"  worst {v['worst_pct']:.0f}th pct" if v.get("worst_pct") else ""
            L.append(f"    {d:24} {v['state']:>14}{sev}")
        caps=[k for k,v in (r.get("prior_caps") or {}).items() if v]
        if caps: L.append(f"\n  ! prior ceilings hit: {', '.join(caps)}")
    L.append(f"\n  {r['note']}")
    return "\n".join(L)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="model/arterialage.json")
    ap.add_argument("--set",nargs="*",default=[])
    ap.add_argument("--json"); ap.add_argument("--stdin",action="store_true")
    ap.add_argument("--format",choices=["text","json"],default="text")
    ap.add_argument("--no-priors",action="store_true")
    ap.add_argument("--demo",action="store_true",
                    help="dtau-only severity tier, capped at 2")
    ap.add_argument("--quiet",action="store_true")
    a=ap.parse_args()

    if a.stdin: payload=json.load(sys.stdin)
    elif a.json: payload=json.load(open(a.json))
    elif a.set:  payload=parse_set(a.set)
    else: raise SystemExit("give --set, --json or --stdin.  -h for examples")

    eng=RiskEngine(a.model, use_load_priors=not a.no_priors, demo_mode=a.demo)
    many=isinstance(payload,list)
    vecs=payload if many else [payload]
    out=[eng.predict({k:v for k,v in fv.items() if k in FEATURES}) for fv in vecs]

    if a.format=="json":
        json.dump(out if many else out[0], sys.stdout, indent=1, default=float)
        sys.stdout.write("\n")
    else:
        for fv,r in zip(vecs,out):
            miss=[f for f in FEATURES if fv.get(f) is None]
            print(render(r,fv,verbose=not a.quiet))
            if miss: print(f"\n  not measured: {', '.join(miss)}")
