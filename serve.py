#!/usr/bin/env python3
"""
serve.py — the display server.

The GUI is a MONITOR, not a form. It has no inputs. Feature vectors arrive
from outside -- from the signal-processing pipeline, or from `push.py` while
the device is not connected -- and the screen renders whatever arrived last.

    python serve.py                       # http://localhost:8000

ENDPOINTS
  GET  /            the monitor
  GET  /model       artefact metadata
  GET  /current     latest prediction (or {"waiting": true})
  GET  /stream      server-sent events; the GUI subscribes, no polling
  GET  /history     the session so far, for the trend strip
  POST /ingest      {feature: value, ...}  <- THE DEVICE / SIG-PROC ENDPOINT
  POST /predict     stateless scoring, does not touch the display
  POST /reset       clear the session

INTEGRATION
    python sigproc.py --emit-features | python push.py --stdin --follow
    curl -s localhost:8000/ingest -H 'Content-Type: application/json' -d @fv.json
"""
import argparse, json, os, queue, sys, threading, time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arterialage import RiskEngine, LOAD_PRIORS

ROOT = Path(__file__).parent
ENGINE = None
FEATURES = ["age","sex","MAP","dtau","Nocturnal_PTT_dipping","dPWV_reserve",
            "ATC95","HR_RANGE","HR_rest","HRR60","HRACS","RR_irregularity"]

LOCK = threading.Lock()
# Live waveform ring. ~3 s at 32 Hz per ear is enough for the display window.
WAVE_MAX = 96
WAVE = {"l": [], "r": [], "fl": None, "fr": None, "seq": 0, "hz": 32}

STATE = {"current": None, "history": [], "n": 0, "last_at": None, "source": None}
SUBS = []          # live SSE subscribers
MAX_HISTORY = 240


def model_meta():
    a = ENGINE.art
    return {"cohort": a.get("cohort"), "label_kind": a.get("label_kind"),
            "synthetic_demo_tables": a.get("synthetic_demo_tables", False),
            "base_prevalence": ENGINE.base_prev, "sources": a.get("sources", []),
            "learned": {k: {"cohort": m.cohort, "label": m.label_kind,
                            "n_bins": m.n_bins, "logit_se": m.logit_se}
                        for k, m in ENGINE.marginals.items()},
            "priors": {k: {"beta": p.beta_per_sd, "beta_sd": p.beta_sd,
                           "provenance": p.provenance, "source": p.source,
                           "pop_mean": p.pop_mean, "pop_sd": p.pop_sd,
                           "sign": p.sign, "units": p.units}
                       for k, p in LOAD_PRIORS.items()},
            "provisional_norms": sorted(getattr(ENGINE, "provisional", [])),
            "demo_mode": ENGINE.demo_mode,
            "features": FEATURES}


def clean(fv):
    return {k: (None if fv.get(k) is None else float(fv[k]))
            for k in FEATURES if k in fv}


def broadcast(evt, payload):
    msg = f"event: {evt}\ndata: {json.dumps(payload, default=float)}\n\n"
    dead = []
    for q in list(SUBS):
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        SUBS.remove(q)


def ingest(fv, source="unknown"):
    """Score a vector and make it the current display state."""
    cv = clean(fv)
    r = ENGINE.predict(cv)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOCK:
        STATE["n"] += 1
        STATE["last_at"] = now
        STATE["source"] = source
        rec = {"seq": STATE["n"], "at": now, "source": source,
               "features": cv, "result": r}
        STATE["current"] = rec
        STATE["history"].append({
            "seq": STATE["n"], "at": now,
            "risk": r["risk_index"], "lo": r["lo"], "hi": r["hi"],
            "mult": r["multiple_of_population"], "tier": r["tier"]["tier"],
            "gap": (r["arterial_age"] or {}).get("gap_years"),
            "dtau": cv.get("dtau"),
            "rank_se_ms": (r.get("rank_guard") or {}).get("se_ms")})
        if len(STATE["history"]) > MAX_HISTORY:
            STATE["history"] = STATE["history"][-MAX_HISTORY:]
        hist = list(STATE["history"])
    broadcast("subject", {"current": rec, "history": hist,
                          "compare": _compare(hist)})
    return rec


def _compare(hist):
    """Order the session's readings by stiffness, GROUPING any that cannot be
    separated. Two subjects inside each other's band are shown as a tie, not
    ranked -- the whole point of the guard."""
    pts = [h for h in hist if h.get("dtau") and h.get("rank_se_ms")]
    pts.sort(key=lambda h: h["dtau"])          # shortest dtau = stiffest first
    out, group = [], []
    for h in pts:
        if group:
            prev = group[-1]
            sep = 1.96 * ((prev["rank_se_ms"] ** 2 + h["rank_se_ms"] ** 2) ** .5)
            if abs(h["dtau"] * 1000 - prev["dtau"] * 1000) <= sep:
                group.append(h); continue
            out.append(group); group = [h]
        else:
            group = [h]
    if group:
        out.append(group)
    return [{"rank": i + 1, "tied": len(g) > 1,
             "members": [{"seq": x["seq"], "dtau_ms": x["dtau"] * 1000,
                          "se_ms": x["rank_se_ms"], "tier": x["tier"],
                          "gap": x.get("gap")} for x in g]}
            for i, g in enumerate(out)]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=float).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            # demo.html is the DEFAULT at / — it is the display that gets
            # shown, so it should not require remembering a flag. The clinical
            # monitor stays reachable at /clinical, and --demo still works.
            f = "demo.html"
            return self._send(200, (ROOT / "gui" / f).read_text(),
                              "text/html; charset=utf-8")
        if p == "/waveform":
            return self._send(200, {"l": WAVE["l"], "r": WAVE["r"],
                                    "fl": WAVE["fl"], "fr": WAVE["fr"],
                                    "seq": WAVE["seq"], "hz": WAVE["hz"]})
        if p in ("/demo", "/demo.html"):
            return self._send(200, (ROOT / "gui" / "demo.html").read_text(),
                              "text/html; charset=utf-8")
        if p in ("/clinical", "/clinical.html"):
            return self._send(200, (ROOT / "gui" / "index.html").read_text(),
                              "text/html; charset=utf-8")
        if p == "/model":
            return self._send(200, model_meta())
        if p == "/current":
            with LOCK:
                c = STATE["current"]
            return self._send(200, c or {"waiting": True})
        if p == "/compare":
            with LOCK:
                hist = list(STATE["history"])
            return self._send(200, {"subjects": _compare(hist)})
        if p == "/history":
            with LOCK:
                return self._send(200, {"history": list(STATE["history"])})
        if p == "/health":
            return self._send(200, {"ok": True, "n": STATE["n"]})
        if p == "/stream":
            return self._stream()
        self._send(404, {"error": "not found"})

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = queue.Queue(maxsize=64)
        SUBS.append(q)
        try:
            with LOCK:
                c, hist = STATE["current"], list(STATE["history"])
            init = ({"current": c, "history": hist} if c else {"waiting": True})
            self.wfile.write(f"event: subject\ndata: {json.dumps(init, default=float)}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(q.get(timeout=15).encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if q in SUBS:
                SUBS.remove(q)

    def do_POST(self):
        if self.path.split("?")[0] == "/waveform":
            # Live PPG from the device. Kept OUT of STATE["current"] so it
            # cannot be mistaken for a feature vector — it is a display feed,
            # not a measurement, and it must not bump the sequence number or
            # trigger a re-score.
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "bad json"})
            w = d.get("w") or []
            WAVE["l"].extend(w[0::2])
            WAVE["r"].extend(w[1::2])
            if len(WAVE["l"]) > WAVE_MAX: WAVE["l"] = WAVE["l"][-WAVE_MAX:]
            if len(WAVE["r"]) > WAVE_MAX: WAVE["r"] = WAVE["r"][-WAVE_MAX:]
            if d.get("fl") is not None: WAVE["fl"] = d["fl"]
            if d.get("fr") is not None: WAVE["fr"] = d["fr"]
            WAVE["seq"] += 1
            return self._send(200, {"ok": True, "n": len(WAVE["l"])})

        p = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad JSON: {e}"})
        src = self.headers.get("X-Source", "http")
        try:
            if p == "/ingest":
                if isinstance(payload, list):
                    return self._send(200, {"ingested": [ingest(fv, src)["seq"]
                                                         for fv in payload]})
                rec = ingest(payload, src)
                return self._send(200, {"seq": rec["seq"], "at": rec["at"],
                                        "tier": rec["result"]["tier"]["tier"],
                                        "multiple_of_population":
                                            rec["result"]["multiple_of_population"]})
            if p == "/predict":
                many = isinstance(payload, list)
                out = [ENGINE.predict(clean(fv)) for fv in (payload if many else [payload])]
                return self._send(200, out if many else out[0])
            if p == "/reset":
                with LOCK:
                    STATE.update(current=None, history=[], n=0, last_at=None, source=None)
                broadcast("subject", {"waiting": True})
                return self._send(200, {"reset": True})
        except Exception as e:
            return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *a):
        if os.environ.get("AA_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="model/arterialage.json")
    ap.add_argument("--demo",action="store_true",
                    help="dtau-only severity tier, capped at 2")
    a = ap.parse_args()
    if not Path(a.model).exists():
        raise SystemExit(f"no model at {a.model}\n  run:  python refit.py --demo")
    ENGINE = RiskEngine(a.model, demo_mode=a.demo)
    m = model_meta()
    print("=" * 68)
    print("  ArterialAge  —  monitor")
    print("=" * 68)
    print(f"  model    {a.model}")
    print(f"  cohort   {m['cohort']}  |  {m['label_kind']}")
    print(f"  learned  {sorted(m['learned'])}")
    print(f"  priors   {sorted(m['priors'])}")
    if m["synthetic_demo_tables"]:
        print("  ! SYNTHETIC demo tables — swap in aou_training_tables.zip")
    if m["provisional_norms"]:
        print(f"  ! provisional norms: {m['provisional_norms']}")
    if ENGINE.demo_mode:
        print("  ** DEMO MODE — arterial-load severity tier, capped at 2 **")
    print(f"\n  CLINICAL  http://localhost:{a.port}/clinical")
    print(f"  DISPLAY   http://localhost:{a.port}      (no inputs — it renders what arrives)")
    print(f"  DEVICE    POST http://localhost:{a.port}/ingest")
    print(f"\n  no device yet? push a vector by hand:")
    print(f"    python push.py --json examples/subject_hfpef.json")
    print(f"    python push.py --replay examples/session.jsonl --interval 2")
    print(f"    python sigproc.py --emit-features | python push.py --stdin --follow\n")
    ThreadingHTTPServer(("", a.port), H).serve_forever()
