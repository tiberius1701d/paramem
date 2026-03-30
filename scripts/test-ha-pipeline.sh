#!/bin/bash
# Integration test for the HA pipeline and tri-path routing.
# Tests: greeting flow, HA path, SOTA path, fallback chains, /refresh-ha.
# Requires: ParaMem server running, HA reachable, SOTA agent configured.
#
# Usage:
#   bash scripts/test-ha-pipeline.sh                        # default
#   bash scripts/test-ha-pipeline.sh http://localhost:8420   # custom URL
#   bash scripts/test-ha-pipeline.sh --verbose               # show full responses

set -uo pipefail

SERVER="${1:-http://localhost:8420}"
VERBOSE=false
[[ "${1:-}" == "--verbose" || "${2:-}" == "--verbose" ]] && VERBOSE=true
[[ "${1:-}" == "--verbose" ]] && SERVER="${2:-http://localhost:8420}"

PASS=0
FAIL=0
SKIP=0

log_verbose() {
    if $VERBOSE; then
        echo "        $1"
    fi
}

# check <label> <text> <expect_pattern> [conversation_id]
check() {
    local label="$1"
    local text="$2"
    local expect="$3"
    local conv_id="${4:-integration-test-$$}"

    local start_ms=$(($(date +%s%N) / 1000000))

    local payload
    payload=$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1], 'conversation_id': sys.argv[2]}))" "$text" "$conv_id")

    response=$(curl -sf -X POST "$SERVER/chat" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null) || {
        echo "  FAIL  $label — server unreachable"
        echo "        curl failed. Is the server running? Check: curl $SERVER/status"
        FAIL=$((FAIL + 1))
        return 1
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
        return 0
    else
        echo "  FAIL  $label (${elapsed}ms)"
        echo "        expected pattern: /$expect/"
        echo "        got: $reply"
        echo "        escalated: $escalated"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

# check_escalated <label> <text> <expect_escalated> <expect_pattern> [conversation_id]
check_escalated() {
    local label="$1"
    local text="$2"
    local expect_esc="$3"
    local expect="$4"
    local conv_id="${5:-integration-test-$$}"

    local start_ms=$(($(date +%s%N) / 1000000))

    local payload
    payload=$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1], 'conversation_id': sys.argv[2]}))" "$text" "$conv_id")

    response=$(curl -sf -X POST "$SERVER/chat" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null) || {
        echo "  FAIL  $label — server unreachable"
        FAIL=$((FAIL + 1))
        return 1
    }

    local end_ms=$(($(date +%s%N) / 1000000))
    local elapsed=$(( end_ms - start_ms ))

    reply=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null)
    escalated=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('escalated', False))" 2>/dev/null)

    local esc_ok=true
    local pat_ok=true

    if [ "$escalated" != "$expect_esc" ]; then
        esc_ok=false
    fi
    if ! echo "$reply" | grep -qi "$expect"; then
        pat_ok=false
    fi

    if $esc_ok && $pat_ok; then
        echo "  PASS  $label (${elapsed}ms)"
        log_verbose "response: $reply"
        log_verbose "escalated: $escalated"
        PASS=$((PASS + 1))
        return 0
    else
        echo "  FAIL  $label (${elapsed}ms)"
        $esc_ok || echo "        expected escalated=$expect_esc, got $escalated"
        $pat_ok || echo "        expected pattern: /$expect/"
        echo "        got: $reply"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

skip() {
    echo "  SKIP  $1 — $2"
    SKIP=$((SKIP + 1))
}

# check_routed <label> <text> <route> <expect_pattern> [conversation_id]
# Forces routing via the "route" parameter (ha, sota)
check_routed() {
    local label="$1"
    local text="$2"
    local route="$3"
    local expect="$4"
    local conv_id="${5:-integration-routed-$$}"

    local start_ms=$(($(date +%s%N) / 1000000))

    # Use python3 to build JSON payload safely (no shell interpolation issues)
    local payload
    payload=$(python3 -c "import json,sys; print(json.dumps({'text': sys.argv[1], 'conversation_id': sys.argv[2], 'route': sys.argv[3]}))" "$text" "$conv_id" "$route")

    response=$(curl -sf -X POST "$SERVER/chat" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null) || {
        echo "  FAIL  $label — server unreachable"
        FAIL=$((FAIL + 1))
        return 1
    }

    local end_ms=$(($(date +%s%N) / 1000000))
    local elapsed=$(( end_ms - start_ms ))

    reply=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null)
    escalated=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('escalated', False))" 2>/dev/null)

    # Fail on error responses — "unavailable" or "couldn't" means the provider is down
    if echo "$reply" | grep -qi "unavailable\|couldn't"; then
        echo "  FAIL  $label (${elapsed}ms, route=$route) — provider error"
        echo "        got: $reply"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if echo "$reply" | grep -qi "$expect"; then
        echo "  PASS  $label (${elapsed}ms, route=$route)"
        log_verbose "response: $reply"
        log_verbose "escalated: $escalated"
        PASS=$((PASS + 1))
        return 0
    else
        echo "  FAIL  $label (${elapsed}ms, route=$route)"
        echo "        expected pattern: /$expect/"
        echo "        got: $reply"
        FAIL=$((FAIL + 1))
        return 1
    fi
}

echo "ParaMem HA Pipeline — Integration Test"
echo "======================================="
echo ""

# ========================================================================
# Prerequisites
# ========================================================================
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
echo ""

# ========================================================================
# 1. Greeting Flow
# ========================================================================
CONV_MAIN="integration-main-$(date +%s)"
echo "--- 1. Greeting Flow ---"
check "greeting prompt" "Hello" "who are you\|who.*you" "$CONV_MAIN"
check "speaker identification" "I am TestUser" "nice to meet\|TestUser" "$CONV_MAIN"

# ========================================================================
# 2. Real-time via HA agent (tools: SerpAPI, HA sensors)
# ========================================================================
echo ""
echo "--- 2. Real-time via HA Agent ---"
check_routed "HA: time query" "What's the current time in Germany?" "ha" \
    "[0-9].*:[0-9]" "$CONV_MAIN"
check_routed "HA: weather query" "What's the weather like in Berlin right now?" "ha" \
    "temperature\|°C\|degrees\|weather\|cloudy\|clear\|rain\|wind\|sun" "$CONV_MAIN"

# ========================================================================
# 3. Real-time via SOTA agents (web search per provider)
# ========================================================================
echo ""
echo "--- 3a. SOTA: Anthropic (Claude web search) ---"
check_routed "Anthropic: time query" "What's the current time in Germany?" "sota:anthropic" \
    "time\|clock\|[0-9].*:[0-9]\|CET\|CEST\|UTC" "$CONV_MAIN"
check_routed "Anthropic: weather query" "What's the weather like in Berlin right now?" "sota:anthropic" \
    "temperature\|°C\|°F\|degrees\|weather\|cloudy\|clear\|rain\|wind\|sun" "$CONV_MAIN"

echo ""
echo "--- 3b. SOTA: OpenAI (web search preview) ---"
check_routed "OpenAI: time query" "What's the current time in Germany?" "sota:openai" \
    "time\|clock\|[0-9].*:[0-9]\|CET\|CEST\|UTC" "$CONV_MAIN"
check_routed "OpenAI: weather query" "What's the weather like in Berlin right now?" "sota:openai" \
    "temperature\|°C\|°F\|degrees\|weather\|cloudy\|clear\|rain\|wind\|sun" "$CONV_MAIN"

echo ""
echo "--- 3c. SOTA: Google Gemini (search grounding) ---"
check_routed "Gemini: time query" "What's the current time in Germany?" "sota:google" \
    "time\|clock\|[0-9].*:[0-9]\|CET\|CEST\|UTC" "$CONV_MAIN"
check_routed "Gemini: weather query" "What's the weather like in Berlin right now?" "sota:google" \
    "temperature\|°C\|°F\|degrees\|weather\|cloudy\|clear\|rain\|wind\|sun" "$CONV_MAIN"

# ========================================================================
# 4. Auto-routing (normal flow: local model → escalation)
# ========================================================================
echo ""
echo "--- 4. Auto-routing (real-time + reasoning) ---"
check_escalated "auto: real-time escalation" \
    "What's the current time in Germany?" "True" \
    "time\|clock\|[0-9]\|CET\|CEST\|UTC" "$CONV_MAIN"
check_escalated "auto: reasoning (no graph match)" \
    "Explain the difference between deductive and inductive reasoning" \
    "True" "deduct\|induct\|reason\|logic\|premis\|conclusion\|general" "$CONV_MAIN"
check_escalated "auto: math reasoning" \
    "If a train travels 120km in 2 hours, what is its average speed?" \
    "True" "60" "$CONV_MAIN"

# ========================================================================
# 5. Memory Recall (local mode only, keys > 0)
# ========================================================================
echo ""
echo "--- 5. Memory Recall ---"
if [ "$mode" = "local" ] && [ "$keys" -gt 0 ]; then
    check "memory probe" "What do you know about me?" "." "$CONV_MAIN"
else
    skip "memory probe" "requires local mode with keys (mode=$mode, keys=$keys)"
fi

# ========================================================================
# 6. Imperative HA Command
# ========================================================================
echo ""
echo "--- 6. Imperative HA Command ---"
check_escalated "imperative: turn on lights" \
    "Turn on the kitchen lights" \
    "True" "." "$CONV_MAIN"

# ========================================================================
# 7. /refresh-ha endpoint
# ========================================================================
echo ""
echo "--- 7. HA Graph Refresh ---"
refresh=$(curl -sf -X POST "$SERVER/refresh-ha" 2>/dev/null) || {
    echo "  FAIL  /refresh-ha — endpoint unreachable"
    FAIL=$((FAIL + 1))
    refresh=""
}

if [ -n "$refresh" ]; then
    r_status=$(echo "$refresh" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    r_entities=$(echo "$refresh" | python3 -c "import sys,json; print(json.load(sys.stdin).get('entities',0))" 2>/dev/null)
    r_verbs=$(echo "$refresh" | python3 -c "import sys,json; print(json.load(sys.stdin).get('verbs',0))" 2>/dev/null)

    if [ "$r_status" = "refreshed" ] && [ "$r_entities" -gt 0 ]; then
        echo "  PASS  HA graph refresh (${r_entities} entities, ${r_verbs} verbs)"
        PASS=$((PASS + 1))
    elif [ "$r_status" = "not_configured" ]; then
        skip "HA graph refresh" "HA not configured"
    else
        echo "  FAIL  HA graph refresh — status=$r_status, entities=$r_entities"
        FAIL=$((FAIL + 1))
    fi
fi

# ========================================================================
# 8. /status endpoint fields
# ========================================================================
echo ""
echo "--- 8. Status Endpoint ---"
status_ok=$(echo "$status" | python3 -c "
import sys, json
s = json.load(sys.stdin)
required = ['model', 'mode', 'adapter_loaded', 'keys_count', 'pending_sessions', 'consolidating']
missing = [k for k in required if k not in s]
print('ok' if not missing else 'missing: ' + ', '.join(missing))
" 2>/dev/null)

if [ "$status_ok" = "ok" ]; then
    echo "  PASS  /status has all required fields"
    PASS=$((PASS + 1))
else
    echo "  FAIL  /status — $status_ok"
    FAIL=$((FAIL + 1))
fi

# ========================================================================
# Results
# ========================================================================
echo ""
echo "======================================="
total=$((PASS + FAIL))
echo "  Passed: $PASS / $total (skipped: $SKIP)"

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
