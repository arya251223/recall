#!/usr/bin/env bash
# Runs unit/integration tests and the frontend smoke test against a live server.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -t . 2>&1 | grep -vE "ResourceWarning|tracemalloc|if method|^$" | tail -5
if command -v node >/dev/null; then
python3 - <<'PY'
import subprocess, threading
from recall.config import Settings
from recall.server import create_server
s = Settings(); s.rate_limit_per_min = 1000
srv, _ = create_server(s, port=0)
threading.Thread(target=srv.serve_forever, daemon=True).start()
r = subprocess.run(["node", "tests/frontend_smoke.mjs", f"http://127.0.0.1:{srv.server_address[1]}"], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()); raise SystemExit(r.returncode)
PY
else echo "node not found: skipping frontend smoke test"; fi
