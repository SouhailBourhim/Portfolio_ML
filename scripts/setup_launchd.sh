#!/usr/bin/env bash
# setup_launchd.sh — install dagster-daemon + dagster-webserver as macOS
# LaunchAgents so the Phase 1 pipeline schedule keeps running across logins
# without a terminal open. Run again any time to pick up a moved project path.
#
# Usage:
#   ./scripts/setup_launchd.sh            # install + (re)start both agents
#   ./scripts/setup_launchd.sh --uninstall # stop + remove both agents
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$PROJECT_ROOT/.venv/bin"
DAGSTER_HOME="$PROJECT_ROOT/.dagster_home"
AGENTS_DIR="$HOME/Library/LaunchAgents"
UID_GUI="gui/$(id -u)"

DAEMON_LABEL="com.portfolioml.dagster-daemon"
WEBSERVER_LABEL="com.portfolioml.dagster-webserver"
DAEMON_PLIST="$AGENTS_DIR/$DAEMON_LABEL.plist"
WEBSERVER_PLIST="$AGENTS_DIR/$WEBSERVER_LABEL.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "Stopping and removing Dagster LaunchAgents..."
    launchctl bootout "$UID_GUI/$DAEMON_LABEL" 2>/dev/null || true
    launchctl bootout "$UID_GUI/$WEBSERVER_LABEL" 2>/dev/null || true
    rm -f "$DAEMON_PLIST" "$WEBSERVER_PLIST"
    echo "Done. (.dagster_home/ data was left untouched.)"
    exit 0
fi

if [[ ! -x "$VENV_BIN/dagster-daemon" ]]; then
    echo "error: $VENV_BIN/dagster-daemon not found — run 'pip install -r requirements.txt' in .venv first." >&2
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "error: $PROJECT_ROOT/.env not found — create it with FRED_API_KEY=... first." >&2
    exit 1
fi

# launchd does not source .env itself, and FRED_API_KEY must reach the
# daemon's spawned code-server subprocess reliably — read it directly
# rather than depend on dotenv auto-load propagating through that subprocess.
FRED_API_KEY="$(grep -E '^FRED_API_KEY=' "$PROJECT_ROOT/.env" | head -1 | cut -d'=' -f2-)"
if [[ -z "$FRED_API_KEY" ]]; then
    echo "error: FRED_API_KEY not found in $PROJECT_ROOT/.env" >&2
    exit 1
fi

mkdir -p "$DAGSTER_HOME/logs" "$AGENTS_DIR"

# Telemetry off; everything else (run/event/schedule storage) defaults to
# local SQLite under DAGSTER_HOME, created automatically on first run.
cat > "$DAGSTER_HOME/dagster.yaml" <<EOF
telemetry:
  enabled: false
EOF

cat > "$DAEMON_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$DAEMON_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_BIN/dagster-daemon</string>
        <string>run</string>
        <string>-w</string>
        <string>workspace.yaml</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DAGSTER_HOME</key>
        <string>$DAGSTER_HOME</string>
        <key>FRED_API_KEY</key>
        <string>$FRED_API_KEY</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$DAGSTER_HOME/logs/dagster-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>$DAGSTER_HOME/logs/dagster-daemon.err.log</string>
</dict>
</plist>
EOF

cat > "$WEBSERVER_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$WEBSERVER_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VENV_BIN/dagster-webserver</string>
        <string>-w</string>
        <string>workspace.yaml</string>
        <string>-h</string>
        <string>127.0.0.1</string>
        <string>-p</string>
        <string>3000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_ROOT</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DAGSTER_HOME</key>
        <string>$DAGSTER_HOME</string>
        <key>FRED_API_KEY</key>
        <string>$FRED_API_KEY</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$DAGSTER_HOME/logs/dagster-webserver.log</string>
    <key>StandardErrorPath</key>
    <string>$DAGSTER_HOME/logs/dagster-webserver.err.log</string>
</dict>
</plist>
EOF

plutil -lint "$DAEMON_PLIST" "$WEBSERVER_PLIST"

# Re-bootstrap cleanly: bootout is a no-op (and harmless) if not loaded yet.
launchctl bootout "$UID_GUI/$DAEMON_LABEL" 2>/dev/null || true
launchctl bootout "$UID_GUI/$WEBSERVER_LABEL" 2>/dev/null || true
launchctl bootstrap "$UID_GUI" "$DAEMON_PLIST"
launchctl bootstrap "$UID_GUI" "$WEBSERVER_PLIST"

echo "Installed and started:"
echo "  $DAEMON_LABEL"
echo "  $WEBSERVER_LABEL"
echo
echo "UI:    http://127.0.0.1:3000"
echo "Logs:  $DAGSTER_HOME/logs/"
echo "Status check (replace daemon with webserver to check the other):"
echo "  launchctl print $UID_GUI/$DAEMON_LABEL"
echo "Uninstall:"
echo "  ./scripts/setup_launchd.sh --uninstall"
