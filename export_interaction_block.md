# Workbench cell — export the BART interaction surface (landmark-incident label)

This was missing from the final export. Run it in the Workbench after Cell 4,
with `cc`, `yy`, `w`, `WEAR` and `FEAT` still in memory. ~3 min.

```python
# ============ STAGE 2: BART non-additive surplus -> exportable grid ============
import numpy as np, pandas as pd, itertools, json, os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
EXP="./aou_export_final"; os.makedirs(EXP,exist_ok=True)
FEAT=["ATC95","HR_RANGE","HR_rest"]; WEAR=["age","age2","sex","MAP","log_n_days","log_days_total"]
yy=cc.y.values.astype(int); w=cc.w.values.astype(float)

def cvp(cols, depth, seed=0):
    """depth=1 -> ADDITIVE twin (no interactions). depth=3 -> full."""
    X=cc[cols].to_numpy(dtype=float,na_value=np.nan); p=np.zeros(len(yy))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,yy):
        m=HistGradientBoostingClassifier(max_depth=depth,max_iter=250,learning_rate=.05,
            interaction_cst="no_interactions" if depth==1 else None, random_state=seed)
        m.fit(X[tr],yy[tr],sample_weight=w[tr]); p[te]=m.predict_proba(X[te])[:,1]
    return np.clip(p,1e-6,1-1e-6)

# --- OPTIONAL: true BART. Slower (~3-8 min) but it is the spec'd model. ---
USE_BART=False        # set True if pymc-bart is installed in the Workbench
if USE_BART:
    import pymc as pm, pymc_bart as pmb
    X=cc[WEAR+FEAT].to_numpy(dtype=float,na_value=np.nan)
    X=np.nan_to_num(X,nan=np.nanmedian(X,axis=0))
    with pm.Model() as md:
        mu=pmb.BART("mu",X,yy,m=50)
        pm.Bernoulli("y",p=pm.math.sigmoid(mu),observed=yy)
        idata=pm.sample(500,tune=500,chains=2,random_seed=0)
    p_full=np.clip(pm.math.sigmoid(idata.posterior["mu"].mean(("chain","draw"))).eval(),1e-6,1-1e-6)
else:
    p_full=cvp(WEAR+FEAT,3)
p_add=cvp(WEAR+FEAT,1)

# --- DOES THE NON-ADDITIVITY ACTUALLY HELP OUT OF SAMPLE? ---
# The old "18.8% non-additive" figure was an IN-SAMPLE sum-of-squares share with
# no null reference. This is the number that matters.
a_full,a_add=roc_auc_score(yy,p_full,sample_weight=w),roc_auc_score(yy,p_add,sample_weight=w)
r=np.random.default_rng(0); ip,ing=np.where(yy==1)[0],np.where(yy==0)[0]; d=[]
for _ in range(400):
    i=np.concatenate([r.choice(ip,len(ip),True),r.choice(ing,len(ing),True)])
    try: d.append(roc_auc_score(yy[i],p_full[i],sample_weight=w[i])
                 -roc_auc_score(yy[i],p_add[i],sample_weight=w[i]))
    except ValueError: pass
lo,hi=np.percentile(d,[2.5,97.5])
print(f"additive twin {a_add:.4f}  full {a_full:.4f}  INTERACTION dAUROC {a_full-a_add:+.4f} [{lo:+.4f},{hi:+.4f}]")
print("  CI excluding zero = the interactions earn their place. Otherwise report the null.")

# --- the surplus, and its distillate ---
surplus=np.log(p_full/(1-p_full))-np.log(p_add/(1-p_add))
for K in (3,5):
    bins,edges=[],[]
    for f in FEAT:
        v=cc[f].to_numpy(float); e=np.unique(np.nanquantile(v,np.linspace(0,1,K+1)))
        edges.append([float(x) for x in e])
        bins.append(np.clip(np.digitize(np.nan_to_num(v,nan=np.nanmedian(v)),e[1:-1]),0,K-1))
    B=np.column_stack(bins); rows=[]; kept=0
    for combo in itertools.product(range(K),repeat=len(FEAT)):
        sel=(B==np.array(combo)).all(1); n=float(w[sel].sum()); nraw=int(sel.sum())
        if nraw<20: continue
        kept+=1
        rows.append({**{f"{f}_bin":c for f,c in zip(FEAT,combo)},
                     "n":n,"n_raw":nraw,"events":float(w[sel][yy[sel]==1].sum()),
                     "rate":float(w[sel][yy[sel]==1].sum()/max(n,1e-9)),
                     "surplus":float(np.average(surplus[sel],weights=w[sel]))})
    pd.DataFrame(rows).to_csv(f"{EXP}/crosstab_activity_k{K}.csv",index=False)
    json.dump({"features":FEAT,"k":K,"edges":edges},
              open(f"{EXP}/crosstab_activity_k{K}_edges.json","w"))
    sp=[r["surplus"] for r in rows]
    print(f"  k={K}: {kept}/{K**len(FEAT)} cells kept ({100*kept/K**len(FEAT):.0f}% occupancy), "
          f"surplus range {min(sp):+.3f}..{max(sp):+.3f}")
print(f"\n-> {EXP}/crosstab_activity_k3.csv, _k5.csv (+ edges json)")
print("   drop these into tables/ and re-run refit.py")
```

## Reading it

**The ΔAUROC line is the one that matters.** `logit(full) − logit(additive)` is the
surplus by construction, but whether it *generalises* is an out-of-sample
question the old export never asked. If the CI includes zero, the honest move
is to report the interactions as null and ship marginals only — the grid still
exports, it just carries a surplus indistinguishable from noise.

**Occupancy at k=5 will be worse than k=3.** 125 cells against ~1,015 events is
~8 events per cell; many will fall under the n≥20 floor. Use k=3 (27 cells) for
the shipped model and k=5 only as a sensitivity check.

**Cells under 20 are suppressed and the engine returns (0, 0) for them** — the
block makes no claim where it has no data, rather than interpolating a guess.
