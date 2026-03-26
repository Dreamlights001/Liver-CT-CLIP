#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_PY="${ROOT_DIR}/scripts/run_zero_shot.py"
TRAIN_PY="${ROOT_DIR}/scripts/ct_lipro_train.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/transformer_maskgit:${ROOT_DIR}/CT_CLIP:${PYTHONPATH:-}"

if [[ ! -f "${RUN_PY}" ]]; then
  echo "Error: cannot find ${RUN_PY}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Error: cannot find ${TRAIN_PY}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: python command not found: ${PYTHON_BIN}" >&2
  exit 1
fi

for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      echo "Usage: scripts/run_hcc_3_templates.sh [common regression args]"
      echo "This script runs 3 CT-image templates: arterial_only, arterial_portal, all_features."
      echo "Each template runs: train (ct_lipro_train.py) then test (run_zero_shot.py --stage test)."
      echo "Necrosis mode is forced to group_only (binary classification)."
      echo "Example:"
      echo "  conda activate test"
      echo "  scripts/run_hcc_3_templates.sh --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish"
      echo
      exit 0
      ;;
    --prompt-template|--prompt-template=*)
      echo "Error: do not pass --prompt-template, this script runs all 3 templates automatically." >&2
      exit 1
      ;;
    --run-name|--run-name=*)
      echo "Error: do not pass --run-name, it may cause 3 template runs to write to the same folder." >&2
      exit 1
      ;;
    --necrosis-mode|--necrosis-mode=*)
      echo "Error: do not pass --necrosis-mode, this script is fixed to --necrosis-mode group_only." >&2
      exit 1
      ;;
    --no-auto-out-subdir)
      echo "Error: do not pass --no-auto-out-subdir, otherwise outputs may overwrite each other." >&2
      exit 1
      ;;
    --stage|--stage=*)
      echo "Error: do not pass --stage, this script controls stage internally (train then test)." >&2
      exit 1
      ;;
    --load-model|--load-model=*)
      echo "Error: do not pass --load-model, this script auto-loads regressor.pt from each template output." >&2
      exit 1
      ;;
  esac
done

TEMPLATES=(
  "arterial_only"
  "arterial_portal"
  "all_features"
)

echo "Running 3 prompt templates with shared args: $*"
echo "Output: each template will be written to a different subdirectory under --out-dir."
echo

for tmpl in "${TEMPLATES[@]}"; do
  echo "========== template: ${tmpl} =========="
  echo "[1/2] training ..."
  "${PYTHON_BIN}" "${TRAIN_PY}" \
    --task regression \
    --necrosis-mode group_only \
    --auto-out-subdir \
    --prompt-template "${tmpl}" \
    "$@"
  echo "[2/2] testing ..."
  "${PYTHON_BIN}" "${RUN_PY}" \
    --task regression \
    --necrosis-mode group_only \
    --auto-out-subdir \
    --stage test \
    --load-model regressor.pt \
    --split-file split_manifest.json \
    --prompt-template "${tmpl}" \
    "$@"
  echo
done

echo "All 3 template runs completed."
