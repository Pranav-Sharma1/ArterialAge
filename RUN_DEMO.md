# Running the demo

Three things run at once. Two terminals plus a browser tab.

## 1. Server  (terminal 1)

    python serve.py

Holds state and serves the GUI. Leave it running.

## 2. Bridge  (terminal 2)

Reads the device, attaches the values it cannot measure, feeds push.py.

    # no hardware — proves the whole host chain
    python3 aa_bridge.py --selftest --age 55 --sex 1 --map 95 \
        | python push.py --stdin --follow

    # tethered, device streaming JSON over VCOM
    python3 aa_bridge.py --serial /dev/tty.usbmodemXXXX --age 55 --sex 1 --map 95 \
        | python push.py --stdin --follow

    # wireless
    python3 aa_bridge.py --ble ArterialAge --age 55 --sex 1 --map 95 \
        | python push.py --stdin --follow

`--follow` is required: it makes push.py read one JSON object per line rather
than expecting a single object. Without it you get a JSONDecodeError.

## 3. GUI

Open `gui/demo.html`.

---

## age, sex and MAP are yours to supply

The device measures dtau, HR_rest and HR_RANGE. It cannot measure age, sex or
mean arterial pressure — MAP is a cuff reading, and it sits in the trained
context term alongside age and sex, so a wrong value moves the tier. Pass the
real ones.

## Units

    dtau              SECONDS    (0.0114 = 11.4 ms)
    RR_irregularity   FRACTION   (0.046 = population median)
    HR_rest/HR_RANGE  bpm

aa_bridge.py range-checks these and DROPS anything implausible rather than
pushing it — dtau arriving in milliseconds instead of seconds is the easy
mistake and it moves a subject several tiers.

## Features not measured

RR_irregularity is omitted when the device does not produce one. That is
deliberate: contract.py rule 2 refuses rather than guesses, the engine returns
`tier_is_floor: true`, and the GUI shows the Atrial domain as "not measured".
A fabricated value would be indistinguishable on screen from a real one.
