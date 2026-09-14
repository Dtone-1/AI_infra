#!/usr/bin/env bash
set -euo pipefail
 
MODE="${1:-eager}"
MODEL_PATH="${MODEL_PATH:-/data/models/WeLMV4.5_YARN}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
 
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-4,5,6,7}"
 
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
  --context-length 16384
  --chunked-prefill-size 8192
  --max-prefill-tokens 8192
  --disable-hybrid-swa-memory
  --host "${HOST}"
  --port "${PORT}"
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
 
exec python -m sglang.launch_server "${ARGS[@]}"