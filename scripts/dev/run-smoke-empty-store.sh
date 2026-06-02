#!/usr/bin/env bash
# Fresh-install smoke harness: run examples/quick_start.py against a throwaway
# EMPTY store, on a host that already runs a populated production server.
#
# Isolation (two independent guarantees):
#   * The temp server uses a THROWAWAY data dir (/tmp) — production's data/ha is
#     never touched.
#   * The temp server binds a DIFFERENT PORT (8421), and the smoke targets that
#     port via PARAMEM_URL.  So even an orphaned smoke client can NEVER reach
#     production on 8420.  (A prior version shared port 8420; on a botched
#     teardown an orphaned smoke hit production and polluted it.)
#
# Lifecycle:
#   1. Build a temp config: paths.{data,sessions,debug} → /tmp; debug:true;
#      procedural OFF (speed); refresh_cadence "" (no full cycle → both stages
#      are interim cycles → 2 consolidation runs total, demonstrating
#      accumulation); port 8421.
#   2. Stop production (frees the GPU; port is already isolated).
#   3. Launch the temp server on 8421 against the empty store.
#   4. Run the smoke (PARAMEM_URL=:8421), tee-ing output to a log file.
#   5. On SUCCESS: delete the throwaway dir.  On FAILURE: PRESERVE it.
#   6. ALWAYS restore production (trap on EXIT) — killing the temp server AND any
#      orphaned smoke client first.
#
# Usage:  bash scripts/dev/run-smoke-empty-store.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PARAMEM_PYTHON:-$HOME/miniforge3/envs/paramem/bin/python}"
PORT=8421                                  # temp server port — NOT 8420 (production)
PROD_URL=http://localhost:8420
TEMP_URL=http://localhost:$PORT
STORE_ROOT=/tmp/paramem-smoke
TEMP_CONFIG=/tmp/smoke-config.yaml
SMOKE_LOG="$STORE_ROOT/smoke-run.log"
SERVER_LOG="$STORE_ROOT/temp-server.log"
TOKEN="$(grep -E '^PARAMEM_API_TOKEN=' .env | cut -d= -f2-)"
TEMP_PID=""

log() { echo "[harness] $*"; }

restore_production() {
  log "teardown: killing temp server + any orphaned smoke client BEFORE restoring production ..."
  [ -n "$TEMP_PID" ] && kill "$TEMP_PID" 2>/dev/null
  pkill -f "examples/quick_start.py" 2>/dev/null
  pkill -f "paramem.server.app --config $TEMP_CONFIG" 2>/dev/null
  sleep 3
  # Confirm the smoke client is actually dead before production reclaims 8420.
  for _ in $(seq 1 10); do
    pgrep -f "examples/quick_start.py" >/dev/null 2>&1 || break
    sleep 1
  done
  log "restoring production server ..."
  systemctl --user start paramem-server
  for _ in $(seq 1 40); do
    sleep 5
    curl -s --max-time 3 "$PROD_URL/health" 2>/dev/null | grep -q '"status":"ok"' && { log "production back up"; return; }
  done
  log "WARNING: production not healthy within timeout — check 'systemctl --user status paramem-server'"
}
trap restore_production EXIT

# --- 1. temp config + empty dirs --------------------------------------------
log "building temp empty-store config (data=$STORE_ROOT, port=$PORT, refresh_cadence='') ..."
rm -rf "$STORE_ROOT"; mkdir -p "$STORE_ROOT"/{data,sessions,debug}
sed -E \
  -e "s|^(  data:) +data/ha.*|\1 $STORE_ROOT/data|" \
  -e "s|^(  sessions:) +data/ha/sessions.*|\1 $STORE_ROOT/sessions|" \
  -e "s|^(  debug:) +data/ha/debug.*|\1 $STORE_ROOT/debug|" \
  configs/server.yaml > "$TEMP_CONFIG"
# procedural OFF (smoke facts are episodic); refresh_cadence "" (no full cycle —
# both stages stay interim); port 8421 (isolation).  debug/early-stop/SOTA stay
# as the shipped config has them.
"$PY" - "$TEMP_CONFIG" "$PORT" <<'PY'
import re, sys, pathlib
cfg = pathlib.Path(sys.argv[1]); port = sys.argv[2]
lines = cfg.read_text().splitlines(); out = []; in_proc = False
for ln in lines:
    if re.match(r'^  procedural:', ln): in_proc = True
    elif re.match(r'^  \w', ln) and not ln.startswith('  procedural'): in_proc = False
    if in_proc and re.match(r'^    enabled:\s*true', ln):
        ln = ln.replace('true', 'false', 1)
    if re.match(r'^  refresh_cadence:', ln):
        ln = '  refresh_cadence: ""'
    if re.match(r'^  port:\s*8420', ln):
        ln = f'  port: {port}'
    out.append(ln)
cfg.write_text("\n".join(out) + "\n")
PY

# --- 2. stop production (free the GPU; port already isolated) ----------------
cons="$(curl -s --max-time 5 -H "Authorization: Bearer $TOKEN" "$PROD_URL/status" 2>/dev/null | "$PY" -c 'import sys,json;print(json.load(sys.stdin).get("consolidating"))' 2>/dev/null || echo unknown)"
log "production consolidating=$cons — stopping it ..."
systemctl --user stop paramem-server
for _ in $(seq 1 15); do systemctl --user is-active paramem-server | grep -q inactive && break; sleep 1; done

# --- 3. launch temp server on $PORT -----------------------------------------
log "launching temp empty-store server on port $PORT ..."
export $(grep -v '^#' .env | xargs) 2>/dev/null
nohup "$PY" -m paramem.server.app --config "$TEMP_CONFIG" > "$SERVER_LOG" 2>&1 &
TEMP_PID=$!
up=0
for i in $(seq 1 40); do
  sleep 5
  if curl -s --max-time 3 "$TEMP_URL/health" 2>/dev/null | grep -q '"status":"ok"'; then up=1; log "temp server up after ~$((i*5))s"; break; fi
done
[ "$up" = 0 ] && { log "FAIL: temp server did not come up — see $SERVER_LOG"; exit 1; }
keys="$(curl -s --max-time 5 -H "Authorization: Bearer $TOKEN" "$TEMP_URL/status" | "$PY" -c 'import sys,json;print(json.load(sys.stdin).get("keys_count"))' 2>/dev/null)"
log "empty-store check (port $PORT): keys_count=$keys"

# --- 4. run the smoke against the TEMP port (tee to log) ---------------------
log "running examples/quick_start.py against $TEMP_URL (log → $SMOKE_LOG) ..."
PARAMEM_URL="$TEMP_URL" PARAMEM_API_TOKEN="$TOKEN" "$PY" -u examples/quick_start.py 2>&1 | tee "$SMOKE_LOG"
SMOKE_EXIT=${PIPESTATUS[0]}

# --- 5. preserve-on-fail / delete-on-success --------------------------------
kill "$TEMP_PID" 2>/dev/null; TEMP_PID=""; sleep 2
if [ "$SMOKE_EXIT" = 0 ]; then
  log "SMOKE PASSED (exit 0) — deleting throwaway store $STORE_ROOT"
  rm -rf "$STORE_ROOT" "$TEMP_CONFIG"
else
  log "SMOKE FAILED (exit $SMOKE_EXIT) — PRESERVING for analysis:"
  log "   smoke log:     $SMOKE_LOG"
  log "   server log:    $SERVER_LOG"
  log "   debug dir:     $STORE_ROOT/debug"
fi

# trap restores production
exit "$SMOKE_EXIT"
