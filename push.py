#!/usr/bin/env python3
"""
push.py — send a feature vector to the running monitor.

This is what the DEVICE will do. Until the device is wired up, use it by hand.

    # one vector, from a file
    python push.py --json examples/subject_hfpef.json

    # one vector, inline
    python push.py --set age=71 sex=1 MAP=104 dtau=0.0072 ATC95=5200 HR_RANGE=31

    # replay a recorded session, one line of JSON per timepoint
    python push.py --replay examples/session.jsonl --interval 2

    # LIVE from your signal-processing pipeline: one JSON object per line
    python sigproc.py --emit-features | python push.py --stdin --follow

`--follow` keeps reading stdin forever, pushing each line as it arrives, so a
long-running sig-proc process streams straight onto the display.

UNITS: dtau in SECONDS (0.0088 = 8.8 ms). sex 0=female 1=male.
Missing features: omit them, or pass null. They produce no shift, the full
prior width, and mark the tier a floor.
"""
import argparse, json, sys, time, urllib.request, urllib.error

FEATURES = ["age","sex","MAP","dtau","Nocturnal_PTT_dipping","dPWV_reserve",
            "ATC95","HR_RANGE","HR_rest","HRR60","HRACS","RR_irregularity"]


def post(url, obj, source):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json",
                                          "X-Source": source})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {url} — is serve.py running?  ({e})")


def parse_set(pairs):
    fv = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"bad --set '{p}', expected key=value")
        k, v = p.split("=", 1)
        if k not in FEATURES:
            raise SystemExit(f"unknown feature '{k}'\n  known: {', '.join(FEATURES)}")
        fv[k] = None if v.lower() in ("", "nan", "null", "none", "na") else float(v)
    return fv


def show(fv, r):
    seen = [k for k in FEATURES if fv.get(k) is not None]
    print(f"  #{r['seq']:<4} tier {r['tier']}   {r['multiple_of_population']:6.2f}x population"
          f"   [{len(seen)}/{len(FEATURES)} features]  {r['at'][11:19]}Z")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/ingest")
    ap.add_argument("--json"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--follow", action="store_true",
                    help="keep reading stdin, one JSON object per line")
    ap.add_argument("--replay"); ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--source", default="manual")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    if a.reset:
        post(a.url.replace("/ingest", "/reset"), {}, a.source)
        print("session cleared"); sys.exit(0)

    if a.replay:
        lines = [l for l in open(a.replay) if l.strip() and not l.startswith("#")]
        print(f"replaying {len(lines)} timepoints at {a.interval}s intervals "
              f"— watch the display")
        for i, l in enumerate(lines):
            fv = json.loads(l)
            show(fv, post(a.url, fv, "replay"))
            if i < len(lines) - 1:
                time.sleep(a.interval)
        sys.exit(0)

    if a.stdin and a.follow:
        print("following stdin — one JSON object per line, Ctrl-C to stop")
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                fv = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"  skipped bad line: {e}\n"); continue
            show(fv, post(a.url, fv, "sigproc"))
        sys.exit(0)

    if a.stdin:   payload = json.load(sys.stdin)
    elif a.json:  payload = json.load(open(a.json))
    elif a.set:   payload = parse_set(a.set)
    else:         raise SystemExit("give --json, --set, --stdin or --replay.  -h for examples")

    if isinstance(payload, list):
        for fv in payload:
            show(fv, post(a.url, fv, a.source))
    else:
        show(payload, post(a.url, payload, a.source))
