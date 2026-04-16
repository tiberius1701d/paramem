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

# Exec stub — executed on PATH, dispatches with exec.
cat > "$DEST/paramem-status.sh" <<'EOF'
#!/bin/bash
# paramem-stub v1 — dispatches to canonical script in repo.
# Regenerate: ~/projects/paramem/scripts/dev/install-stubs.sh
exec bash "$HOME/projects/paramem/scripts/dev/paramem-status.sh" "$@"
EOF
chmod +x "$DEST/paramem-status.sh"

# Source stub — sourced by ~/.bashrc, loads functions into current shell.
cat > "$DEST/training-control.sh" <<'EOF'
# paramem-stub v1 — sourced by ~/.bashrc to expose training_{pause,resume,status}.
# Regenerate: ~/projects/paramem/scripts/dev/install-stubs.sh
[ -r "$HOME/projects/paramem/scripts/dev/training-control.sh" ] && \
    . "$HOME/projects/paramem/scripts/dev/training-control.sh"
EOF

echo "Installed stubs in $DEST:"
ls -la "$DEST/paramem-status.sh" "$DEST/training-control.sh"
echo
echo "Canonical scripts live in $REPO_ROOT/scripts/dev/"
