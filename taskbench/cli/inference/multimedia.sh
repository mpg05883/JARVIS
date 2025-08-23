#!/bin/bash
mkdir -p logs
source ./cli/utils/common.sh
activate_task_bench_env
load_env_variables

log_info "Starting inference..."

# Debug: Check if API key is loaded
if [ -z "$TOGETHER_API_KEY" ]; then
    log_error "TOGETHER_API_KEY is not set. Please check your .env file."
    exit 1
else
    log_info "Together AI API key loaded successfully (length: ${#TOGETHER_API_KEY})"
fi

MODELS=("meta-llama/Llama-3.3-70B-Instruct-Turbo")
DATA_DIRS=("data_multimedia")
NUM_RUNS=$((${#MODELS[@]} * ${#DATA_DIRS[@]}))

SEED=42
WAIT_TIME_SECONDS=3.0  

RUN_INDEX=0
for MODEL in "${MODELS[@]}"; do
    for DATA_DIR in "${DATA_DIRS[@]}"; do
        printf '%*s\n' 160 '' | tr ' ' '-'
        log_info "Run $((RUN_INDEX+1))/${NUM_RUNS} | model=$MODEL, data_dir=$DATA_DIR"

        if python inference.py \
            --llm "$MODEL" \
            --data_dir "$DATA_DIR" \
            --api_addr api.together.xyz \
            --api_port 443 \
            --api_key "$TOGETHER_API_KEY" \
            --seed $SEED \
            --wait_time $WAIT_TIME_SECONDS \
            --multiworker 1; then
            log_info "Successfully finished inference for model: $MODEL on data directory: $DATA_DIR"
        else
            log_error "Inference failed for model: $MODEL on data directory: $DATA_DIR"
            exit 1
        fi
    done
done

printf '%*s\n' 160 '' | tr ' ' '-'
log_info "Finished all ${NUM_RUNS} runs!"