#!/bin/bash
# Install tiny wrapper-stubs at ~/.local/bin/ that dispatch to canonical
# scripts in this repo. Stubs survive atomic-rename Write operations that
# would follow symlinks and corrupt the real script.
#
# Run after a fresh clone (or when stubs drift):
#   bash scripts/dev/install-stubs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$HOME/.local/bin"
mkdir -p "$DEST"

# Exec stub — executed on PATH, dispatches with exec. Heredoc is unquoted so
# $REPO_ROOT substitutes into a fixed absolute path at generation time; "$@"
# is escaped so it stays a runtime expansion in the written stub.
cat > "$DEST/paramem-status.sh" <<EOF
#!/bin/bash
# paramem-stub v1 — dispatches to canonical script in repo.
# Regenerate: $REPO_ROOT/scripts/dev/install-stubs.sh
exec bash "$REPO_ROOT/scripts/dev/paramem-status.sh" "\$@"
EOF
chmod +x "$DEST/paramem-status.sh"

# Source stub — sourced by ~/.bashrc, loads functions into current shell.
cat > "$DEST/training-control.sh" <<EOF
# paramem-stub v1 — sourced by ~/.bashrc to expose training_{pause,resume,status}.
# Regenerate: $REPO_ROOT/scripts/dev/install-stubs.sh
[ -r "$REPO_ROOT/scripts/dev/training-control.sh" ] && \\
    . "$REPO_ROOT/scripts/dev/training-control.sh"
EOF

echo "Installed stubs in $DEST:"
ls -la "$DEST/paramem-status.sh" "$DEST/training-control.sh"
echo
echo "Canonical scripts live in $REPO_ROOT/scripts/dev/"
