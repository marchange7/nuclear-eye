#!/usr/bin/env bash
# check-model-manifest.sh
#
# CI gate for nuclear-eye (Sentinelle perception runtime).
# Identical logic to sentinelle/scripts/check-model-manifest.sh.
#
# Exit 0 on clean. Exit 1 on any violation.
# Doctrine: os/70 §2 + model-manifest.yml in repo root.

set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="model-manifest.yml"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: $MANIFEST not found at repo root" >&2
    exit 1
fi

VIOLATIONS=0

FORBIDDEN_GLOBS=("*.mlmodelc" "*.mlpackage" "*.safetensors" "*.pt")

SCAN_DIRS=()
for d in models services src; do [[ -d "$d" ]] && SCAN_DIRS+=("$d"); done

if [[ ${#SCAN_DIRS[@]} -gt 0 ]]; then
    for pat in "${FORBIDDEN_GLOBS[@]}"; do
        hits=$(find "${SCAN_DIRS[@]}" -name "$pat" 2>/dev/null | head -5)
        if [[ -n "$hits" ]]; then
            echo "VIOLATION (forbidden-glob '$pat'):"
            echo "$hits"
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done

    for frag in "arianne" "companion" "emile"; do
        hits=$(find "${SCAN_DIRS[@]}" -iname "*${frag}*" 2>/dev/null | head -5)
        if [[ -n "$hits" ]]; then
            echo "VIOLATION (forbidden product artifact '$frag'):"
            echo "$hits"
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done
fi

if [[ -d models ]]; then
    while IFS= read -r -d '' artifact; do
        rel="${artifact#./}"
        if ! grep -q "bundle_path: ${rel}" "$MANIFEST" 2>/dev/null; then
            echo "VIOLATION (unlisted ONNX artifact): $rel not declared in $MANIFEST"
            VIOLATIONS=$((VIOLATIONS + 1))
        fi
    done < <(find models/ -name "*.onnx" 2>/dev/null | sort -z)
fi

if [[ $VIOLATIONS -gt 0 ]]; then
    echo ""
    echo "check-model-manifest: $VIOLATIONS violation(s). See model-manifest.yml (os/70 §2)." >&2
    exit 1
fi

echo "OK: model manifest clean. ($(grep -c 'bundle_path:' "$MANIFEST") declared artifact(s))"
exit 0
