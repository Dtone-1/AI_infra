#!/usr/bin/env bash
set -euo pipefail
 
MODE="${1:-graph}"
MODEL_PATH="${MODEL_PATH:-/data/models/WeLMV4.5_YARN}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
 
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-1,2,3,4}"
export SGLANG_ENABLE_SPEC_V2=1
 
ARGS=(
  --model-path "${MODEL_PATH}"
  --device mlu
  --dtype bfloat16
  --tp-size 4
  --ep-size 4
  --moe-runner-backend mlu
  --moe-a2a-backend none
  --welm-shared-embedding-policy disabled
  --enable-welm-kv-mirror-opt
  --disable-hybrid-swa-memory
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
)

ARGS+=(
  --speculative-algorithm EAGLE
  --speculative-draft-model-path "${MODEL_PATH}"
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)
 
case "${MODE}" in
  eager)
    ARGS+=(--disable-cuda-graph)
    ;;
  graph)
    ;;
  *)
    echo "Usage: $0 [eager|graph]" >&2
    exit 2
    ;;
esac
 
echo "Starting WeLM service"
echo "  mode:    ${MODE}"
echo "  model:   ${MODEL_PATH}"
echo "  devices: ${MLU_VISIBLE_DEVICES}"
echo "  address: http://${HOST}:${PORT}"
 
LOG_FILE="server_$(date +%Y%m%d_%H%M%S).log"
exec python -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
