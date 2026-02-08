#!/usr/bin/env bash
# startup.sh — Initialize the research environment inside the container.
#
# Called after the Apptainer instance starts. Sets up:
#   1. Ray cluster (head + workers on local GPUs)
#   2. Agent workspace with the paper's repo
#   3. Budget tracking
#   4. Baseline metrics (run once, cached)
#
# Environment variables (set by the orchestrator):
#   REPO_URL          - GitHub URL of the paper's code
#   REPO_BRANCH       - Branch to checkout (default: main)
#   PAPER_PATH        - Path to paper markdown (bind-mounted)
#   GPU_BUDGET_HOURS  - Total GPU-hours budget (default: 1.0)
#   MAX_CONCURRENT    - Max concurrent Ray jobs (default: 4)
#   BASELINE_CMD      - Command to run baseline (e.g., "python train.py --config base")
#   BASELINE_METRIC   - JSONPath to the metric in baseline output (e.g., ".eval_loss")
#   NUM_GPUS          - Number of GPUs (auto-detected if unset)
#   TASK_JSON         - Path to task definition JSON

set -euo pipefail

WORKSPACE="${HOME}/workspace"
RESULTS_DIR="${HOME}/results"
REPO_DIR="${WORKSPACE}/repo"
BUDGET_FILE="${HOME}/.budget.json"
BASELINE_FILE="${HOME}/.baseline_metrics.json"

echo "=== Research Environment Startup ==="

# ----------------------------------------------------------------
# 1. Detect GPUs
# ----------------------------------------------------------------
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(python3 -c "
try:
    import subprocess
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    print(len([l for l in out.strip().split('\n') if l.strip()]))
except Exception:
    print(0)
")
fi
echo "[1/5] Detected ${NUM_GPUS} GPUs"

# ----------------------------------------------------------------
# 2. Start Ray cluster
# ----------------------------------------------------------------
echo "[2/5] Starting Ray cluster..."
ray stop --force 2>/dev/null || true

ray start --head \
    --num-gpus="${NUM_GPUS}" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats \
    --include-dashboard=true \
    2>&1 | tail -3

# Verify Ray is up
python3 -c "
import ray
ray.init(address='auto')
resources = ray.cluster_resources()
print(f'  Ray cluster: {resources.get(\"CPU\", 0):.0f} CPUs, {resources.get(\"GPU\", 0):.0f} GPUs')
ray.shutdown()
"

# ----------------------------------------------------------------
# 3. Set up workspace
# ----------------------------------------------------------------
echo "[3/5] Setting up workspace..."
mkdir -p "${WORKSPACE}" "${RESULTS_DIR}"

if [ -n "${REPO_URL:-}" ] && [ ! -d "${REPO_DIR}/.git" ]; then
    BRANCH="${REPO_BRANCH:-main}"
    echo "  Cloning ${REPO_URL} (branch: ${BRANCH})..."
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}" 2>&1 | tail -2

    # Install repo dependencies if pyproject.toml or requirements.txt exists
    if [ -f "${REPO_DIR}/pyproject.toml" ]; then
        echo "  Installing repo (pyproject.toml)..."
        cd "${REPO_DIR}" && pip install -e "." 2>&1 | tail -3
    elif [ -f "${REPO_DIR}/requirements.txt" ]; then
        echo "  Installing repo (requirements.txt)..."
        pip install -r "${REPO_DIR}/requirements.txt" 2>&1 | tail -3
    fi
    cd "${HOME}"
fi

# Copy paper if available
if [ -n "${PAPER_PATH:-}" ] && [ -f "${PAPER_PATH}" ]; then
    cp "${PAPER_PATH}" "${WORKSPACE}/paper.md"
    echo "  Paper available at ${WORKSPACE}/paper.md"
fi

# Copy task definition if available
if [ -n "${TASK_JSON:-}" ] && [ -f "${TASK_JSON}" ]; then
    cp "${TASK_JSON}" "${HOME}/task.json"
fi

# Git init the workspace so agent changes are diffable
cd "${REPO_DIR}" 2>/dev/null && git add -A && git commit -m "baseline" --allow-empty 2>/dev/null || true
cd "${HOME}"

# ----------------------------------------------------------------
# 4. Initialize budget tracking
# ----------------------------------------------------------------
echo "[4/5] Initializing budget tracker..."
GPU_BUDGET="${GPU_BUDGET_HOURS:-1.0}"
MAX_JOBS="${MAX_CONCURRENT:-4}"

python3 -c "
import json
budget = {
    'total_gpu_seconds': ${GPU_BUDGET} * 3600,
    'used_gpu_seconds': 0.0,
    'remaining_gpu_seconds': ${GPU_BUDGET} * 3600,
    'max_concurrent_jobs': ${MAX_JOBS},
    'num_gpus': ${NUM_GPUS},
    'jobs': []
}
with open('${BUDGET_FILE}', 'w') as f:
    json.dump(budget, f, indent=2)
print(f'  Budget: {${GPU_BUDGET}:.1f} GPU-hours, {${MAX_JOBS}} max concurrent jobs')
"

# ----------------------------------------------------------------
# 5. Run baseline (if configured)
# ----------------------------------------------------------------
if [ -n "${BASELINE_CMD:-}" ]; then
    echo "[5/5] Running baseline..."
    echo "  Command: ${BASELINE_CMD}"

    BASELINE_START=$(date +%s)

    # Run baseline from repo directory, capture output
    cd "${REPO_DIR}"
    eval "${BASELINE_CMD}" > "${HOME}/.baseline_stdout.txt" 2>&1
    BASELINE_RC=$?
    cd "${HOME}"

    BASELINE_END=$(date +%s)
    BASELINE_DURATION=$((BASELINE_END - BASELINE_START))
    echo "  Baseline completed in ${BASELINE_DURATION}s (exit code: ${BASELINE_RC})"

    if [ ${BASELINE_RC} -eq 0 ]; then
        # Extract metric from stdout or results file
        python3 -c "
import json, sys

# Try to load from a results file first
results_file = '${REPO_DIR}/results.json'
stdout_file = '${HOME}/.baseline_stdout.txt'
metric_path = '${BASELINE_METRIC:-eval_loss}'

metrics = {}
try:
    with open(results_file) as f:
        data = json.load(f)
    # Navigate JSONPath-like metric_path (simple dot notation)
    val = data
    for key in metric_path.lstrip('.').split('.'):
        val = val[key]
    metrics['primary_metric'] = float(val)
    metrics['metric_name'] = metric_path
except Exception:
    # Fall back: try to parse last JSON line from stdout
    try:
        with open(stdout_file) as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                data = json.loads(line.strip())
                if metric_path.lstrip('.') in data:
                    metrics['primary_metric'] = float(data[metric_path.lstrip('.')])
                    metrics['metric_name'] = metric_path
                    break
            except (json.JSONDecodeError, KeyError):
                continue
    except Exception as e:
        print(f'  Warning: Could not extract baseline metric: {e}', file=sys.stderr)

metrics['baseline_duration_seconds'] = ${BASELINE_DURATION}
metrics['baseline_command'] = '${BASELINE_CMD}'

with open('${BASELINE_FILE}', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  Baseline metric: {metrics}')
"
    else
        echo "  WARNING: Baseline failed. Agent will need to fix the setup."
    fi
else
    echo "[5/5] No baseline command configured, skipping."
fi

# ----------------------------------------------------------------
# Compute integrity hashes for anti-tampering
# ----------------------------------------------------------------
python3 -c "
import hashlib, json, pathlib, os

hashes = {}
# Hash eval data directory
data_dir = pathlib.Path('/data')
if data_dir.exists():
    for f in sorted(data_dir.rglob('*')):
        if f.is_file() and f.stat().st_size < 100_000_000:  # skip files > 100MB
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            hashes[str(f)] = h

# Hash baseline metrics
baseline = pathlib.Path('${BASELINE_FILE}')
if baseline.exists():
    hashes['baseline_metrics'] = hashlib.sha256(baseline.read_bytes()).hexdigest()

with open('${HOME}/.integrity_hashes.json', 'w') as f:
    json.dump(hashes, f, indent=2)
print(f'  Integrity hashes computed for {len(hashes)} files')
"

echo ""
echo "=== Environment Ready ==="
echo "  Workspace:  ${WORKSPACE}"
echo "  Repo:       ${REPO_DIR}"
echo "  Results:    ${RESULTS_DIR}"
echo "  Budget:     ${BUDGET_FILE}"
echo "  Baseline:   ${BASELINE_FILE}"
echo ""
echo "  Use 'ray job submit' to submit experiments."
echo "  Use 'budget' to check remaining GPU-hours."
echo "  Write final results to ${RESULTS_DIR}/final_result.json"
