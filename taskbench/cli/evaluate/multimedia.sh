#!/bin/bash
mkdir -p logs
source ./cli/utils/common.sh
activate_task_bench_env

log_info "Starting evaluation..."
MODELS=("meta-llama/Llama-3.3-70B-Instruct-Turbo")
DATA_DIRS=("data_multimedia")
NUM_RUNS=$((${#MODELS[@]} * ${#DATA_DIRS[@]}))

RUN_INDEX=0
for MODEL in "${MODELS[@]}"; do
    for DATA_DIR in "${DATA_DIRS[@]}"; do
        printf '%*s\n' 160 '' | tr ' ' '-'
        log_info "Run $((RUN_INDEX+1))/${NUM_RUNS} | model=$MODEL, data_dir=$DATA_DIR"

        if python evaluate.py \
            --data_dir "$DATA_DIR" \
            --prediction_dir "predictions_use_demos_2_reformat_by_self" \
            --llm "$MODEL"; then
            log_info "Successfully finished inference for model: $MODEL on data directory: $DATA_DIR"
        else
            log_error "Inference failed for model: $MODEL on data directory: $DATA_DIR"
            exit 1
        fi
    done
done

printf '%*s\n' 160 '' | tr ' ' '-'
log_info "Finished all ${NUM_RUNS} runs!"