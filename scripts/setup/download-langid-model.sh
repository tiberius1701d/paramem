#!/usr/bin/env bash
# Idempotent fetch of fastText lid.176.bin (126 MB) into the paramem cache.
#
# Usage:
#   bash scripts/setup/download-langid-model.sh
#
# The model identifies 176 languages from text and powers the text-side
# language detection on the /chat endpoint when no STT-derived language
# signal is available.  Once present, the file is reused indefinitely;
# re-running the script is a no-op when the destination is already valid.
#
# Optional env:
#   LANGID_MODEL_SHA256  — override the pinned sha256 to verify against.
#                          Defaults to the value observed at first download
#                          (2026-05-10) so subsequent installs reproduce
#                          the exact bytes.

set -euo pipefail

DEST_DIR="${PARAMEM_LANGID_DIR:-$HOME/.cache/paramem/lang_id}"
DEST="${DEST_DIR}/lid.176.bin"
URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
EXPECTED_BYTES=131266198  # ~126 MiB; published Facebook CDN size
EXPECTED_SHA256="${LANGID_MODEL_SHA256:-7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e}"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]]; then
    actual_bytes=$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")
    if [[ "$actual_bytes" == "$EXPECTED_BYTES" ]]; then
        echo "lang_id model already present at $DEST (${actual_bytes} bytes)"
        exit 0
    fi
    echo "lang_id model at $DEST has unexpected size ${actual_bytes} (expected ${EXPECTED_BYTES}); re-downloading"
    rm -f "$DEST"
fi

echo "Downloading lid.176.bin from $URL ..."
curl --fail --location --show-error --progress-bar -o "$DEST.partial" "$URL"
mv "$DEST.partial" "$DEST"

actual_sha=$(sha256sum "$DEST" | cut -d' ' -f1)
echo "lang_id model installed at $DEST"
echo "  sha256: $actual_sha"

if [[ "$actual_sha" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: sha256 mismatch (expected $EXPECTED_SHA256)" >&2
    rm -f "$DEST"
    exit 1
fi
echo "  sha256 verified against pinned value"
