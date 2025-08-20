#!/bin/bash
mkdir -p logs
source ./cli/utils.sh
activate_task_bench_env
load_env_variables

log_info "Starting inference..."

# Debug: Check if API key is loaded
if [ -z "$OPENAI_API_KEY" ]; then
    log_error "OPENAI_API_KEY is not set. Please check your .env file."
    exit 1
else
    log_info "API key loaded successfully (length: ${#OPENAI_API_KEY})"
fi

MODELS=("gpt-4")
DATA_DIRS=("data_multimedia" "data_huggingface" "data_dailylifeapis")
NUM_RUNS=$((${#MODELS[@]} * ${#DATA_DIRS[@]}))

RUN_INDEX=0
for MODEL in "${MODELS[@]}"; do
    for DATA_DIR in "${DATA_DIRS[@]}"; do
        printf '%*s\n' 160 '' | tr ' ' '-'
        log_info "Run $((RUN_INDEX+1))/${NUM_RUNS} | model=$MODEL, data_dir=$DATA_DIR"

        if python inference.py \
            --llm $MODEL \
            --data_dir $DATA_DIR \
            --api_addr api.openai.com \
            --api_port 443 \
            --api_key $OPENAI_API_KEY; then
            log_info "Successfully finished inference for model: $MODEL on data directory: $DATA_DIR"
        else
            log_error "Inference failed for model: $MODEL on data directory: $DATA_DIR"
            exit 1
        fi
    done
done

printf '%*s\n' 160 '' | tr ' ' '-'
log_info "Finished all ${NUM_RUNS} runs!"