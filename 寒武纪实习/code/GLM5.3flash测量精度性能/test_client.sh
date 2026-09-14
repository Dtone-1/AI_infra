#!/bin/bash

set -o pipefail

# --- knobs (override via env) -------------------------------------------
SERVER_PORT="${SERVER_PORT:-20000}"
# AIME-2026 avg@32 is sampled 32x per problem; HLE/MMMU-Pro use a single sample.
AIME_N_SAMPLES="${AIME_N_SAMPLES:-32}"
# Max output tokens for the long-reasoning text benchmarks (AIME/HLE) and MMMU-Pro.
MAX_TOKENS="${MAX_TOKENS:-200000}"
# LLM judge endpoint for HLE (HLE is graded by an external LLM judge). Two injection
# routes are supported:
#   explicit: JUDGE_API_URL + JUDGE_API_KEY (judge model via JUDGE_MODEL, default
#             gpt-5-mini-2025-08-07 per the leaderboard recipe), e.g.
#             export JUDGE_API_URL="<judge openai-compatible url>"
#             export JUDGE_API_KEY="<judge key>"
#   ModelScope: export MODELSCOPE_API_BASE="<endpoint>" (model via JUDGE_MODEL or its
#             own MODELSCOPE_JUDGE_LLM env fallback)
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini-2025-08-07}"
JUDGE_API_URL="${JUDGE_API_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

EVAL_CHECK="--api-url http://127.0.0.1:${SERVER_PORT}/v1 --api-key EMPTY --eval-type openai_api --model glm5"

# --- 0) gsm8k: fast smoke to confirm the server is serving correctly -----
evalscope eval \
${EVAL_CHECK} \
--datasets gsm8k \
--generation-config '{"temperature": 0, "max_tokens": 1024, "seed": 1, "extra_body": {"chat_template_kwargs": {"thinking": false, "enable_thinking": false}}}' \
--dataset-args '{"gsm8k": {"few_shot_num": 4, "few_shot_random": false}}' \
--eval-batch-size 64 \
--timeout 18000 \
2>&1 | tee log_glm5_client_gsm8k.txt

# --- 1) AIME 2026 (text, avg@32, temp 1.0 / top_p 0.95 / top_k 2) --------
# --repeats AIME_N_SAMPLES duplicates each 0-shot sample at dataset load and the report
# aggregator averages the group -> exactly the leaderboard avg@32. Each duplicate runs
# one independent inference at temp 1.0. (generation-config "n" maps to the same repeats
# mechanism but is a deprecated evalscope API that emits a warning; --repeats is current.)
evalscope eval \
${EVAL_CHECK} \
--datasets aime26 \
--repeats ${AIME_N_SAMPLES} \
--generation-config "{\"do_sample\": true, \"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 2, \"max_tokens\": ${MAX_TOKENS}, \"seed\": 42, \"extra_body\": {\"chat_template_kwargs\": {\"thinking\": true, \"enable_thinking\": true}}}" \
--eval-batch-size 64 \
--timeout 600000 \
2>&1 | tee log_glm5_client_aime26.txt

# --- 2) HLE (text, temp 1.0 / top_p 0.95 / top_k 2, LLM-judge graded) ----
# HLE's adapter forces an external LLM judge (GRADE C/I). The judge may be injected either
# explicitly (JUDGE_API_URL/JUDGE_API_KEY; gpt-5-mini per the leaderboard recipe) or via
# ModelScope env (MODELSCOPE_API_BASE, judge model via JUDGE_MODEL). HLE grading is
# impossible without a reachable judge, so skip when none is configured instead of burning
# a multi-hour generation then scoring 0.
HLE_JUDGE_READY=0
if [ -n "${JUDGE_API_URL}" ] && [ -n "${JUDGE_API_KEY}" ]; then
  HLE_JUDGE_READY=1
elif [ -n "${MODELSCOPE_API_BASE}" ]; then
  HLE_JUDGE_READY=1
fi

if [ "${HLE_JUDGE_READY}" = "1" ]; then
JUDGE_ARGS="{\"model_id\": \"${JUDGE_MODEL}\""
[ -n "${JUDGE_API_URL}" ] && JUDGE_ARGS="${JUDGE_ARGS}, \"api_url\": \"${JUDGE_API_URL}\""
[ -n "${JUDGE_API_KEY}" ] && JUDGE_ARGS="${JUDGE_ARGS}, \"api_key\": \"${JUDGE_API_KEY}\""
JUDGE_ARGS="${JUDGE_ARGS}}"
evalscope eval \
${EVAL_CHECK} \
--datasets hle \
--judge-strategy llm \
--judge-model-args "${JUDGE_ARGS}" \
--generation-config "{\"do_sample\": true, \"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 2, \"max_tokens\": ${MAX_TOKENS}, \"seed\": 42, \"extra_body\": {\"chat_template_kwargs\": {\"thinking\": true, \"enable_thinking\": true}}}" \
--eval-batch-size 16 \
--timeout 600000 \
2>&1 | tee log_glm5_client_hle.txt
fi

# --- 3) MMMU-Pro (multimodal, temp 1.0 / top_p 0.95 / top_k 10) ----------
# Images ride through the server /v1 endpoint; graded by exact answer letter.
evalscope eval \
${EVAL_CHECK} \
--datasets mmmu_pro \
--generation-config "{\"do_sample\": true, \"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 10, \"max_tokens\": ${MAX_TOKENS}, \"seed\": 42, \"extra_body\": {\"chat_template_kwargs\": {\"thinking\": true, \"enable_thinking\": true}}}" \
--eval-batch-size 8 \
--timeout 600000 \
2>&1 | tee log_glm5_client_mmmu_pro.txt

# --- 4) MMLU-Pro (text baseline kept from the original configuration) ----
evalscope eval \
${EVAL_CHECK} \
--datasets mmlu_pro \
--generation-config '{"do_sample": true, "temperature": 1.0, "top_p": 1.0, "max_tokens": '"${MAX_TOKENS}"', "seed": 1, "extra_body": {"chat_template_kwargs": {"thinking": true, "enable_thinking": true}}}' \
--eval-batch-size 64 \
--dataset-args '{"mmlu_pro": {"filters": {"remove_until": " response"}}}' \
--timeout 360000 \
2>&1 | tee log_glm5_client_mmlu_pro.txt
