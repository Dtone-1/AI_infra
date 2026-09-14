#!/bin/bash
 
MODEL_NAME=WeLMV4.5_YARN-W8A8
tokenizer_path=/data/models/WeLMV4.5_YARN-W8A8
 
#====gr1===
input_len=11108
output_len=89
 
# input_len=3000
# output_len=300
 
TIME_STAMP=$(date '+%Y%m%d_%H%M%S')
 
# Loop through rate and parallel combinations (rate:parallel pairs)
declare -A rate_parallel=(
    ["0.3"]=3
    ["0.5"]=5
    ["1.0"]=10
    ["2.0"]=20
    ["4.0"]=40
    ["6.0"]=60
    ["8.0"]=80
    ["20.0"]=200
)
 
# Iterate through sorted keys to maintain order
for rate in $(echo "${!rate_parallel[@]}" | tr ' ' '\n' | sort -n); do
    parallel=${rate_parallel[$rate]}
    echo "Running with rate=$rate, parallel=$parallel"
 
    evalscope perf \
            --url http://127.0.0.1:30000/v1/chat/completions \
            --model ${MODEL_NAME} \
            --dataset random \
            --api-key "" \
            --parallel ${parallel} \
            --rate "${rate}"  \vim bench_client.sh
            --number 200 \
            --temperature 0.0 \
            --min-prompt-length 11000 \
            --max-prompt-length 11000 \
            --min-tokens 100 \
            --max-tokens 100 \
            --prefix-length 0 \
            --tokenizer-path "${tokenizer_path}" \
            --name ${MODEL_NAME} \
            --warmup-num 10 \
            --outputs-dir WeLM-mlu-outputs
    echo "Finished rate=$rate, parallel=$parallel, num=200"
    echo "-----------------------------------"
done
 
            #--max-prompt-length "${input_len}" \
            #--min-prompt-length "${input_len}" \
            #--max-tokens "${output_len}" \
            #--min-tokens "${output_len}" \
echo "All runs completed!"



# find /workspace -type f | grep -E "json|csv|log|result|output"