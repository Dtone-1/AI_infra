#!/bin/bash
 
set -o pipefail
 
cd /workspace/vllm_mlu
unset RAY_ADDRESS
unset RAY_DISABLE_CLUSTER_DISCOVERY
export GLOO_SOCKET_IFNAME=$(ip addr | awk '/inet 10.98\./ {print $NF}')
export RAY_EXPERIMENTAL_NOSET_MLU_VISIBLE_DEVICES=true
export PYTORCH_MLU_ALLOC_CONF="expandable_segments:True"
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_CGRAPH_get_timeout=300000
export CNCL_CNPX_DOMAIN_ENABLE=0
export VLLM_LATENCY_DEBUG=false
export VLLM_USE_V1=1
export NOSET_MLU_VISIBLE_DEVICES_ENV_VAR=1
export VLLM_V1_UNCHUNK_SCHED_LOG=1
export VLLM_ENGINE_READY_TIMEOUT_S=360000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=600000
export VLLM_ENABLE_DATA_PARALLEL_SERVING_METRICS=1
source tools/config_env.sh
 
MODEL_PATH="${MODEL_PATH:-/data1/models/hf_models/zai-org/GLM-5-Next-0808/compressed_tensors/GLM-5-Next-0808-ct-intw4a8/}"
SERVER_PORT="${SERVER_PORT:-20000}"
SERVER_LOG="${SERVER_LOG:-log_glm5_next_server.txt}"
if [[ -z "${SPECULATIVE_CONFIG+x}" ]]; then
    SPECULATIVE_CONFIG='{"method": "mtp", "num_speculative_tokens": 2}'
fi

USE_GRAPH="${USE_GRAPH:-false}"
GRAPH_OPTION="--enforce-eager"
unset VLLM_USE_BREAKABLE_CUDAGRAPH

if [[ "${USE_GRAPH}" == "true" ]]; then
    # export VLLM_USE_BREAKABLE_CUDAGRAPH=1
    GRAPH_OPTION=""
fi

# Keep the default runtime shape aligned with the H20 reference run. Override
# MODEL_LEN only for long-context leaderboards that require the 256K window.
MODEL_LEN="${MODEL_LEN:-210000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
 
#     --enforce-eager \
#     --default-chat-template-kwargs '{"thinking":false}' \
#    --enable-auto-tool-choice \
#    --tool-call-parser glm5_next \
#    --reasoning-parser glm5_next \
#    --enable-prefix-caching --mamba-cache-mode align
#    --no-enable-prefix-caching \
#    -sc '{"method": "mtp", "num_speculative_tokens": 2}' \

vllm serve "${MODEL_PATH}" \
    --served-model-name glm5 \
    --distributed-executor-backend mp \
    --port "${SERVER_PORT}" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --tokenizer-mode glm5_next \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --max-num-batched-tokens 16384 \
    --max-model-len "${MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --block-size "${BLOCK_SIZE}" \
    --enable-expert-parallel \
    --enable-prefix-caching --mamba-cache-mode align \
    --enable-chunked-prefill \
    --disable-custom-all-reduce \
    --distributed-timeout-seconds 6000 \
    --cpu-distributed-timeout-seconds 6000 \
    --async-scheduling \
    ${GRAPH_OPTION} \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --skip-mm-profiling \
    -sc "${SPECULATIVE_CONFIG}" \
    2>&1 | tee "${SERVER_LOG}"
