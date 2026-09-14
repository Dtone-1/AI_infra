#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/data/models/WeLMV4.5_YARN"
MODEL_ID="WeLMV4.5_YARN"
API_URL="http://127.0.0.1:30000/v1/chat/completions"

evalscope eval \
  --model "${MODEL_PATH}" \
  --model-id "${MODEL_ID}" \
  --datasets gsm8k \
  --eval-type openai_api \
  --api-url "${API_URL}" \
  --generation-config '{"timeout":7200000,"batch_size":32,"max_tokens":20000,"top_p":1.0,"temperature":0.0,"seed":42,"do_sample":false,"stream":true,"extra_body":{"chat_template_kwargs":{"enable_thinking":false,"thinking":false}}}' \
  --eval-batch-size 32\
  --seed 42 \
  --no-timestamp \
  --collect-perf \
  --work-dir /projs/solutionsdk/mahao/outputs/gsm8k_after_mtp1