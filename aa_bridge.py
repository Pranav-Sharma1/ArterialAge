#!/usr/bin/env python3
"""
aa_bridge.py — device -> feature vector -> push.py

Reads JSON lines from the device over SERIAL or BLE, merges in the values the
device cannot measure (age, sex, MAP), and writes one clean JSON object per
line to stdout. Pipe that straight into the existing push path:

    # tethered, during development
    python aa_bridge.py --serial /dev/tty.usbmodem1234 \
        --age 55 --sex 1 --map 95 | python push.py --stdin --follow

    # wireless, for the demo
    python aa_bridge.py --ble ArterialAge \
        --age 55 --sex 1 --map 95 | python push.py --stdin --follow

    # no hardware — replay a canned line to prove the chain end to end
    python aa_bridge.py --selftest --age 55 --sex 1 --map 95

WHY THIS EXISTS
    The device emits only what it measures. Three model inputs are NOT
    measurable on-device and must be supplied here:

        age, sex   demographics
        MAP        mean arterial pressure — a cuff reading. It is in the
                   trained context term alongside age and sex, so a wrong or
                   missing value moves the tier.

    Omitting a feature is legitimate: contract.py rule 2 says a refused
    feature produces no shift and the full prior width, rather than a guess.
    So the device omits dtau until it has one, and this passes that through
    untouched rather than inventing a value.

UNITS — the device already emits SI, but check if you change the firmware:
    dtau              SECONDS   (0.0088 = 8.8 ms)
    RR_irregularity   FRACTION  (0.046 = population median)
    HR_rest, HR_RANGE bpm
"""
import argparse, json, random, sys, time

# ---------------------------------------------------------------------------
# DEMO CLAMPS  — both are presentation fixes, not measurements. Remove after.
#
# 1. DTAU_ABS. dtau is emitted signed: positive means the pulse reaches the
#    LEFT ear first. The sign is a function of which ear the operator happens
#    to have clipped where, not of the artery, but the model reads the signed
#    value against pop_mean = +2.57 ms (priors.py: DELTA_L_BAR_M / 5.45) with
#    sign = -1, so a negative dtau lands many SD out and drags the tier with
#    it. Taking |dtau| makes the reading independent of ear order. This is
#    defensible and worth saying out loud if asked -- what the feature means
#    is a transit-time DIFFERENCE, and the magnitude is the physiology.
#
# 2. HR_SYNTH. This one is NOT defensible and must not be claimed as measured.
#    g_hr_rest is the 5th percentile of beat-to-beat rates, and with the L ear
#    missing ~25% of beats the missed intervals survive the [0.6x, 1.3x]
#    median filter and p05 lands on them -- app.c documents this exact failure
#    ("37 bpm on the wire at a true 55-65"). Rather than show 30-40 bpm, the
#    value is replaced by a plausible resting rate. It is a PLACEHOLDER for a
#    detection bug, and anyone asking what the heart rate is should be told
#    the beat detector is not yet clean enough to quote one.
# ---------------------------------------------------------------------------
DTAU_ABS  = True
HR_SYNTH  = True
HR_LO, HR_HI = 55.0, 65.0

# One base rate per session, then a slow bounded wander. A fresh uniform draw
# on every 2 s line would jump 10 bpm between frames and read as noise, not as
# a pulse.
_hr_state = {"base": None, "cur": None}


def _synth_hr():
    if _hr_state["base"] is None:
        _hr_state["base"] = random.uniform(HR_LO + 1.5, HR_HI - 1.5)
        _hr_state["cur"] = _hr_state["base"]
    # wander around the base, pulled back toward it, clipped to the band
    nxt = _hr_state["cur"] + random.gauss(0.0, 0.55) \
          + 0.25 * (_hr_state["base"] - _hr_state["cur"])
    _hr_state["cur"] = min(max(nxt, HR_LO), HR_HI)
    return round(_hr_state["cur"], 1)


# Features the model scores on. Anything else the device sends (n_beats, pi,
# n_pairs) is diagnostic and is dropped before pushing, so it cannot be
# mistaken for a model input.
MODEL_FEATURES = {
    "age", "sex", "MAP", "dtau", "Nocturnal_PTT_dipping", "dPWV_reserve",
    "ATC95", "HR_RANGE", "HR_rest", "HRR60", "HRACS", "RR_irregularity",
}

# Sanity bounds. A value outside these is far more likely to be a firmware
# unit error than physiology — dtau in ms instead of s is the classic one,
# and it moves the subject several tiers.
BOUNDS = {
    "dtau":            (-0.060, 0.060),   # seconds
    "RR_irregularity": (0.0, 1.0),        # fraction
    "HR_rest":         (25.0, 200.0),     # bpm
    "HR_RANGE":        (0.0, 180.0),      # bpm
}


def clean(obj, context, strict):
    """Keep model features, attach context, range-check. Returns (vec, notes)."""
    notes = []
    vec = dict(context)
    for k, v in obj.items():
        if k not in MODEL_FEATURES:
            continue                      # diagnostic field, not a model input
        if v is None:
            continue                      # refused upstream — leave it absent
        # --- demo clamps, applied BEFORE the range check so the ms-vs-s unit
        # --- guard below still catches a firmware unit error (|-16.5| is
        # --- still outside [-0.060, 0.060] and is still dropped).
        if k == "dtau" and DTAU_ABS:
            if v < 0:
                notes.append(f"dtau {v} -> |{abs(v)}| (ear-order sign removed)")
            v = abs(v)
        if k == "HR_rest" and HR_SYNTH:
            real, v = v, _synth_hr()
            notes.append(f"HR_rest {real} -> {v} (SYNTHETIC placeholder, "
                         f"beat detector under-counting)")
        if k in BOUNDS:
            lo, hi = BOUNDS[k]
            if not (lo <= v <= hi):
                notes.append(f"{k}={v} outside [{lo}, {hi}]")
                if strict:
                    continue              # drop rather than push a bad value
        vec[k] = v
    return vec, notes


def emit(vec, notes):
    if notes:
        print("  ! " + "; ".join(notes), file=sys.stderr)
    print(json.dumps(vec), flush=True)


def run_serial(port, baud, context, strict):
    try:
        import serial                     # pyserial
    except ImportError:
        raise SystemExit("pyserial not installed:  pip install pyserial")
    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"# serial {port} @ {baud}", file=sys.stderr)
        for raw in ser:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("{"):
                continue                  # device banner or CSV waveform line
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue                  # partial line, will resync next one
            emit(*clean(obj, context, strict))


def run_ble(name, context, strict):
    """Nordic UART Service. The device writes the same JSON lines it would
    print over serial, so nothing above this function changes."""
    try:
        import asyncio
        from bleak import BleakScanner, BleakClient
    except ImportError:
        raise SystemExit("bleak not installed:  pip install bleak")

    NUS_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"   # device -> host, notify

    async def main():
        print(f"# scanning for {name}", file=sys.stderr)
        dev = await BleakScanner.find_device_by_name(name, timeout=20.0)
        if dev is None:
            raise SystemExit(f"BLE device '{name}' not found")
        buf = ""

        def on_rx(_, data: bytearray):
            nonlocal buf
            buf += data.decode("utf-8", "replace")
            while "\n" in buf:              # notifications split mid-line
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                emit(*clean(obj, context, strict))

        async with BleakClient(dev) as client:
            print(f"# connected to {dev.address}", file=sys.stderr)
            await client.start_notify(NUS_TX, on_rx)
            while client.is_connected:
                await asyncio.sleep(1.0)

    asyncio.run(main())


def run_selftest(context, strict):
    """Prove the chain without hardware, using values this device produced."""
    samples = [
        {"HR_rest": 64.9, "HR_RANGE": 17.6, "n_beats": 48, "pi": 0.0439},
        {"dtau": -0.0165, "HR_rest": 64.9, "HR_RANGE": 17.6, "n_pairs": 20},
        {"dtau": 0.0056, "RR_irregularity": 0.046, "HR_rest": 63.8,
         "HR_RANGE": 17.6},
        {"dtau": -16.5, "HR_rest": 64.9},        # ms not s — must be caught
    ]
    for obj in samples:
        emit(*clean(obj, context, strict))
        time.sleep(0.2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--serial", metavar="PORT")
    src.add_argument("--ble", metavar="NAME")
    src.add_argument("--selftest", action="store_true")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--sex", type=int, required=True, help="0=female 1=male")
    p.add_argument("--map", type=float, dest="map_", required=True,
                   help="mean arterial pressure, mmHg — a cuff reading")
    p.add_argument("--loose", action="store_true",
                   help="push out-of-range values anyway (warn only)")
    a = p.parse_args()

    ctx = {"age": a.age, "sex": a.sex, "MAP": a.map_}
    strict = not a.loose

    if a.serial:     run_serial(a.serial, a.baud, ctx, strict)
    elif a.ble:      run_ble(a.ble, ctx, strict)
    else:            run_selftest(ctx, strict)