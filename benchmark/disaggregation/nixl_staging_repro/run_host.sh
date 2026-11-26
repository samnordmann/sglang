#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 || ( $1 != 0 && $1 != 1 ) ]]; then
    echo "Usage: run_host.sh ROLE(0=prefill,1=decode) HOST_A HOST_B SHARED_OUTPUT" >&2
    exit 2
fi
role=$1
host_a=$2
host_b=$3
out=$(realpath "$4")
bench=$(cd "$(dirname "$0")" && pwd)
: "${SGLANG_REPO:?path of local git repository with both source revisions}"
: "${PATCH_REF:?the candidate commit or PR branch, fetched in SGLANG_REPO}"
: "${MODEL:?path of the pinned Qwen3-0.6B model downloaded with hf --local-dir}"
base_ref=${BASE_REF:-0730305dbee380a841640c50510ebe3e9e1d9302}
test -f "$out/serving-plan.json"
mkdir "$out/role-$role.started"
# Build independent, clean source copies. Only the two result hooks change.
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
for arm in baseline patched; do
    ref=$base_ref
    [[ "$arm" != patched ]] || ref=$PATCH_REF
    rev=$(git -C "$SGLANG_REPO" rev-parse --verify "${ref}^{commit}")
    printf '%s\n' "$rev" > "$out/revision-$role-$arm.txt"
    mkdir "$scratch/$arm"
    git -C "$SGLANG_REPO" archive "$rev" | tar -xf - -C "$scratch/$arm"
    python "$bench/install_endpoints.py" "$scratch/$arm" > "$out/hooks-$role-$arm.json"
done
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
export SGLANG_RUST_SERVER=0 UCX_PROTO_INFO=y UCX_LOG_LEVEL=info
export NO_PROXY="127.0.0.1,localhost,$host_a,$host_b" no_proxy="127.0.0.1,localhost,$host_a,$host_b"
unset UCX_LOG_FILE KV_MINIMAL_ENDPOINTS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python "$bench/inventory.py" "$MODEL" "$role" "$host_a" "$host_b" > "$out/node-$role.json"
if [[ $role == 1 ]]; then
    mapfile -t options < <(python "$bench/plan.py" oracle "$out")
    PYTHONPATH="$scratch/baseline/python:$bench${PYTHONPATH:+:$PYTHONPATH}" python "$bench/serve_trial.py" oracle "$out/oracle" --model "$MODEL" "${options[@]}"
    touch "$out/oracle.done"
fi
mapfile -t runs < <(python "$bench/plan.py" runs "$out")
for run in "${runs[@]}"; do
    arm=baseline
    [[ $run != *-patched ]] || arm=patched
    PYTHONPATH="$scratch/$arm/python:$bench${PYTHONPATH:+:$PYTHONPATH}" KV_MINIMAL_ENDPOINTS=1 python "$bench/role_minimal.py" "$role" "$bench" "$MODEL" "$out" "$run" "$host_a" "$host_b"
done
