#!/usr/bin/env python3
"""Post one line to Slack via an incoming webhook (SLACK_WEBHOOK in /opt/bench/env). Always also
appends to /opt/bench/notify.log, so nothing is lost when the webhook is unset or Slack is down."""
import json, os, sys, time, urllib.request
msg = " ".join(sys.argv[1:]).strip()
if not msg: sys.exit(0)
env = {}
try:
    for ln in open("/opt/bench/env"):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("=", 1); env[k] = v.strip().strip('"')
except FileNotFoundError:
    pass
host = os.uname().nodename
line = "[%s %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), host, msg)
with open("/opt/bench/notify.log", "a") as f: f.write(line + "\n")
url = env.get("SLACK_WEBHOOK") or os.environ.get("SLACK_WEBHOOK")
if url:
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": line}).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        with open("/opt/bench/notify.log", "a") as f: f.write("  (slack post failed: %s)\n" % e)
