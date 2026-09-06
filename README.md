# ArterialAge

To access the core sig proc and bluetooth file, switch from main to workspace a in top left hand corner

To access server/Demo GUI, switch from main to workspace b in to left hand corner

Operating commands for demo:

KILL OLD SERVER:
lsof -ti:8000 | xargs kill -9


START SERVER:
python3 serve.py


RUN COMMAND:

python3 aa_bridge.py --ble "Empty Ex" --age 55 --sex 1 --map 95 \
    | python3 push.py --stdin --follow


Since BLE program pushed to device, just need to switch device on before executing server and leptop bluetooth picks up the device once commands above ran.
