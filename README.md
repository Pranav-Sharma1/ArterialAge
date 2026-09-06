# ArterialAge — trained model, CLI and GUI

Everything below runs offline. The GUI, the terminal CLI and the HTTP API all
call **one** `RiskEngine`, so a feature vector scored in the browser and the
same vector scored in the terminal return identical numbers by construction.

**The model in `model/arterialage.json` is already trained on the real All of
Us aggregate tables** (`tables/`). `refit.py --demo` exists only to regenerate
synthetic tables if you want to test the pipeline without them.

```bash
pip install -r requirements.txt
python refit.py --tables tables    # 1. rebuild from the real tables
python serve.py               # 2. open http://localhost:8000  -> "waiting for a vector"

# 3. in a SECOND terminal, act as the device:
python push.py --json examples/subject_hfpef.json
python push.py --replay examples/session.jsonl --interval 2
```

**The GUI has no inputs.** It is a monitor: it renders whatever the last
feature vector produced. Vectors arrive from the device / signal-processing
pipeline via `POST /ingest`; `push.py` is the stand-in until the device is
wired up. The browser holds no risk maths at all, so it cannot drift from the
model.

---

## 1. Retrain from the real All of Us exports

Unzip `aou_training_tables.zip` into `tables/`, overwriting the demo files:

```bash
rm -f tables/*.csv
unzip aou_training_tables.zip -d tables/
python refit.py --tables tables --out model/arterialage.json \
    --cohort "All of Us" --label "incident HF (90d landmark)"
```

Expected files:

| file | what it teaches |
|---|---|
| `context_table.csv` | the age·sex·MAP context term — **and the baseline** |
| `marginal_ATC95.csv` | activity-ceiling marginal |
| `marginal_HR_RANGE.csv` | HR-range marginal |
| `marginal_HR_rest.csv` | resting-HR marginal |
| `quantiles_*.csv` | age/sex reference percentiles → the HMOD tier |

`refit.py` prints each marginal's logit swing and standard error. Sanity check:
swings should be roughly 0.3–0.8 and SEs 0.02–0.08. An SE pinned at 0.5 means
the stratum centring failed — check that `age_lo`/`age_hi`/`sex` survived the
export.

**Why tables and not rows.** All of Us forbids exporting models fitted on
participant data but permits models fitted on summary data with no cell under
20 people. A univariate curve refit from ~15 weighted bins is statistically
near-identical to one fit on the rows.

## 2. Stage-2 interactions, and RR_irregularity

`export_interaction_block.md` has the Workbench cell that exports BART's
non-additive surplus as a 3x3x3 grid. **BART never ships** — it is a posterior
over tree ensembles, a training-time object. What ships is the distillate: 27
cells and a lookup, ~500 bytes. `refit.py` picks up any `crosstab_*.csv` in
`tables/` automatically.

That cell also reports the number the original export never did: the
**out-of-sample ΔAUROC of the full model over its additive twin**. If that CI
includes zero, the interactions are null and you ship marginals only.

### RR_irregularity (skipped)

```bash
python train_rr_irregularity.py --data /path/to/mc-med --out tables/
python refit.py --tables tables --out model/arterialage.json   # picks it up
```

It writes the same aggregate format, so no code changes. It prints an
unadjusted AUC and labels it **associative only** — MC-MED is a prevalent ED
label, and the All of Us landmark analysis showed 96% of a prevalent-label
effect was reverse causation.

---

## 3. Scoring from the terminal

```bash
# inline
python predict.py --set age=71 sex=1 MAP=104 dtau=0.0072 ATC95=5200 HR_RANGE=31

# from a file
python predict.py --json examples/subject_hfpef.json

# machine-readable
python predict.py --json examples/subject_normal.json --format json

# ablate the priors, to see what the learned block alone says
python predict.py --json examples/subject_hfpef.json --no-priors
```

Output shows the tier, the multiple of population rate, the arterial-age gap,
and a **provenance ladder** — every term marked learned (`#`) or prior (`o`).

## 4. Connecting your signal-processing pipeline

The integration point is a flat JSON object, **one number per feature per
subject**, aggregated over many beats/nights upstream. One line of JSON per
timepoint.

```bash
# LIVE: sig-proc streams onto the display, one JSON object per line
python sigproc.py --emit-features | python push.py --stdin --follow

# a recorded session, replayed at 2 s per timepoint
python push.py --replay examples/session.jsonl --interval 2

# straight HTTP, from anything
curl -s localhost:8000/ingest -H 'Content-Type: application/json' -d @fv.json
```

`--follow` reads stdin forever, so a long-running process streams
continuously. Each vector updates the display and extends the session trend.

```python
# in-process, no server
from arterialage import RiskEngine
eng = RiskEngine("model/arterialage.json")
r = eng.predict({"age": 68, "sex": 1, "MAP": 99, "dtau": 0.0088})
print(r["tier"]["tier"], r["multiple_of_population"])
```

### Endpoints

| endpoint | purpose |
|---|---|
| `POST /ingest` | **the device endpoint** — score a vector and put it on the display |
| `GET /stream` | server-sent events; the GUI subscribes, no polling |
| `GET /current` | the latest prediction, or `{"waiting": true}` |
| `GET /history` | the session so far, for the trend strip |
| `POST /predict` | stateless scoring — does **not** touch the display |
| `POST /reset` | clear the session |

```python
# in-process
from arterialage import RiskEngine
eng = RiskEngine("model/arterialage.json")
r = eng.predict({"age": 68, "sex": 1, "MAP": 99, "dtau": 0.0088})
print(r["tier"]["tier"], r["multiple_of_population"])
```

**Units.** `dtau` in **seconds** (0.0088 = 8.8 ms). `Nocturnal_PTT_dipping` is
the ratio `dtau_night / dtau_day` (>1 = healthy dipper).
`dPWV_reserve` is `ln(dtau_active/dtau_rest) + a_c·ΔP` (positive = healthy
endothelium). `sex` 0 = female, 1 = male.

**Missing features.** Omit them or pass `null`. An unmeasured feature produces
no shift, the full prior width, and marks the tier a floor. That is the honest
state of knowledge about that subject, and it is worth demonstrating.

---

## 5. What the outputs mean

| output | status | what it supports |
|---|---|---|
| **Tier 0–3** | threshold-based against percentiles; **never uses a β** | the clinical read — same logic as HMOD scoring |
| **× population rate** | robust to every β being wrong by 2× | triage and ranking |
| **risk index %** | **`calibrated: false`** | not for decisions until pilot recalibration |
| **arterial-age gap** | ranking-based | the product claim: "you're 55, your arteries are 70" |
| **provenance ladder** | every term stamped | which part is evidence, which is assumption |

Perturbation testing: halving or doubling **every** β leaves the subject
ranking identical (ρ = 1.000) while moving the absolute percentage by up to 47
points. **Magnitudes barely affect who ranks above whom; a wrong sign destroys
it.** That is why `calibrated: false` and why the tier is the clinical output.

### Provisional norms

No device cohort exists yet, so the arterial domain has no measured
percentiles. The engine substitutes **provisional** norms from each prior's
stated population distribution and flags them everywhere they surface. Without
them the arterial domain is permanently `NOT ASSESSED` and the tier can never
exceed 2. The first pilot percentiles replace them.

---

## 6. Layout

```
arterialage/
  priors.py       load-block priors, one-sided synergy, three ceilings
  marginals.py    refit learned curves from aggregate tables
  engine.py       composite + HMOD tier — THE single source of truth
refit.py          tables -> model/arterialage.json
train_rr_irregularity.py   MC-MED -> tables/
predict.py        terminal CLI — stateless scoring, prints the ladder
push.py           send a vector to the monitor (device stand-in)
serve.py          display server: /ingest, /stream, /current, /history
gui/index.html    the MONITOR — no inputs, no risk maths, SSE-driven
examples/session.jsonl   12 monthly timepoints, one subject deteriorating
```

## 7. Known state

| item | status |
|---|---|
| AoU marginals — ATC95, HR_RANGE, HR_rest | **trained** on the real export: 100/92/94 bins, logit swings 0.98/1.05/0.77, SE 0.25/0.16/0.11 |
| context (age·sex·MAP) | **trained**, counted once; age +0.060/yr, sex +0.24, MAP +0.0066/mmHg, prevalence 1.66% |
| measured result carried in the artefact | baseline 0.7995 → 0.8116, ΔAUROC +0.0121 [+0.0067, +0.0182]; HFpEF +0.0091; within-age 0.752; decile OR 81→171 |
| Stage-2 interaction block (BART distillate) | **not yet exported on the incident label** — run `export_interaction_block.md` in the Workbench, drop the crosstab into `tables/`, re-run `refit.py`. Loader and engine wiring are live and tested |
| RR_irregularity | **skipped.** MC-MED numerics are nurse-charted vitals hours apart (one visit showed two HR values 3h49m apart); MIMIC chartevents is hourly; Ext-PPG never downloaded. AF irregularity is a ~1 Hz beat-to-beat property, so no accessible dataset samples fast enough. Atrial domain reads NOT ASSESSED |
| dtau prior | DERIVED from cfPWV→HF, HR 1.29/m·s⁻¹ |
| Nocturnal_PTT_dipping prior | **MEASURED** — HR 0.81/SD, 625 HF events |
| dPWV_reserve prior | **ANALOGY** — replace two lines marked `<-- REPLACE` in `priors.py` with the MESA FMD per-SD HR |
| HRR60, HRACS | in the spec, **untested** — not estimable from Fitbit's smoothed HR |
| device reference norms | **missing** — provisional norms in use |
| calibration | `false` — needs pilot outcomes |
