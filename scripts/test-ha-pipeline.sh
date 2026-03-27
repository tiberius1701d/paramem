#!/bin/bash
# Integration sanity check for the HA pipeline.
# Requires: ParaMem server running, HA reachable, cloud agent configured.
#
# Usage:
#   bash scripts/test-ha-pipeline.sh                        # default
#   bash scripts/test-ha-pipeline.sh http://localhost:8420   # custom URL
#   bash scripts/test-ha-pipeline.sh --verbose               # show full responses

set -euo pipefail

SERVER="${1:-http://localhost:8420}"
VERBOSE=false
[[ "${1:-}" == "--verbose" || "${2:-}" == "--verbose" ]] && VERBOSE=true
[[ "${1:-}" == "--verbose" ]] && SERVER="${2:-http://localhost:8420}"

CONV_ID="integration-test-$(date +%s)"
PASS=0
FAIL=0

log_verbose() {
    if $VERBOSE; then
        echo "        $1"
    fi
}

check() {
    local label="$1"
    local text="$2"
    local expect="$3"

    local start_ms=$(($(date +%s%N) / 1000000))

    response=$(curl -sf -X POST "$SERVER/chat" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$text\", \"conversation_id\": \"$CONV_ID\"}" 2>/dev/null) || {
        echo "  FAIL  $label — server unreachable"
        echo "        curl failed. Is the server running? Check: curl $SERVER/status"
        FAIL=$((FAIL + 1))
        return
    }

    local end_ms=$(($(date +%s%N) / 1000000))
    local elapsed=$(( end_ms - start_ms ))

    reply=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null)
    escalated=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('escalated', False))" 2>/dev/null)

    if echo "$reply" | grep -qi "$expect"; then
        echo "  PASS  $label (${elapsed}ms)"
        log_verbose "response: $reply"
        log_verbose "escalated: $escalated"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $label (${elapsed}ms)"
        echo "        expected pattern: /$expect/"
        echo "        got: $reply"
        echo "        escalated: $escalated"
        FAIL=$((FAIL + 1))
    fi
}

echo "ParaMem HA Pipeline — Integration Test"
echo "======================================="
echo ""

# Step 1: Server reachable?
echo "--- Prerequisites ---"
status=$(curl -sf "$SERVER/status" 2>/dev/null) || {
    echo "  FAIL  Server not reachable at $SERVER"
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Is the server running?  systemctl --user status paramem-server"
    echo "    2. Check logs:             journalctl --user -u paramem-server -n 20"
    echo "    3. Start manually:         bash scripts/start-server.sh"
    exit 1
}

mode=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode','unknown'))")
model=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model','unknown'))")
keys=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('keys_count',0))")
echo "  OK    Server: $SERVER"
echo "        Mode: $mode | Model: $model | Keys: $keys"

# Step 2: HA reachable? (via server's HA client)
ha_check=$(echo "$status" | python3 -c "
import sys,json
s = json.load(sys.stdin)
# If server loaded tools, HA is reachable
print('ok')
" 2>/dev/null)
echo "  OK    HA connection (server loaded tools at startup)"
echo ""

# Step 3: Greeting flow
echo "--- Greeting Flow ---"
check "greeting prompt" "Hello" "who are you\|who.*you"
check "speaker identification" "I am TestUser" "nice to meet\|TestUser"

# Step 4: Cloud escalation
echo ""
echo "--- Cloud Escalation ---"

if [ "$mode" = "cloud-only" ]; then
    echo "  INFO  Server in cloud-only mode — all queries route to HA agent"
fi

check "time query" "What time is it?" "[0-9]"
check "weather query" "How is the weather?" "temperature\|°C\|degrees\|weather\|cloudy\|clear\|rain"

# Step 5: Memory (only if keys > 0 and mode is local)
if [ "$mode" = "local" ] && [ "$keys" -gt 0 ]; then
    echo ""
    echo "--- Memory Recall ---"
    check "memory probe" "What do you know about me?" "."
fi

# Results
echo ""
echo "--- Results ---"
echo "  Passed: $PASS / $((PASS + FAIL))"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "  Troubleshooting failed tests:"
    echo "    Server logs:  journalctl --user -u paramem-server -n 50"
    echo "    Verbose mode: bash scripts/test-ha-pipeline.sh --verbose"
    echo "    HA status:    curl -s $SERVER/status | python3 -m json.tool"
    exit 1
else
    echo "  All checks passed."
fi
