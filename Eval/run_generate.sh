#!/bin/bash

project_path=$(cd $(dirname $0)/..; pwd)

CUSTOM_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo $CUSTOM_CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

MODEL_PATH=""
MODEL_NAME=""


declare -A TEST_FILES=(
    ["NLP4LP"]="$project_path/eval/benchmark/NLP4LP.jsonl"
    ["NL4Opt"]="$project_path/eval/benchmark/NL4Opt.jsonl"
    ["MamoEasy"]="$project_path/eval/benchmark/MamoEasy.jsonl"
    ["MamoComplex"]="$project_path/eval/benchmark/MamoComplex.jsonl"
    ["ComplexOR"]="$project_path/eval/benchmark/ComplexOR.jsonl"
    ["IndustryOR"]="$project_path/eval/benchmark/IndustryOR.jsonl"
    ["OptiBench"]="$project_path/eval/benchmark/OptiBench.jsonl"
    ["OptMATH"]="$project_path/eval/benchmark/OptMATH.jsonl"
)


DECODING_METHOD="greedy"
PASSK=1

OUTPUT_DIR="$project_path/Eval/$MODEL_NAME"
mkdir -p "$OUTPUT_DIR"

MASTER_LOG="$OUTPUT_DIR/generation.log"
: > "$MASTER_LOG"


echo "=============================================="
echo " Model: $MODEL_NAME"
echo " GPUs: $CUSTOM_CUDA_VISIBLE_DEVICES"
echo " Output Dir: $OUTPUT_DIR"
echo "==============================================" | tee -a "$MASTER_LOG"


for BENCHMARK_NAME in "${!TEST_FILES[@]}"; do
    TEST_FILE=${TEST_FILES[$BENCHMARK_NAME]}

    if [ ! -f "$TEST_FILE" ]; then
        echo "Warning: Test file not found: $TEST_FILE" | tee -a "$MASTER_LOG"
        continue
    fi

    echo "----------------------------------------------" | tee -a "$MASTER_LOG"
    echo "Running dataset: $BENCHMARK_NAME" | tee -a "$MASTER_LOG"
    echo "Input file: $TEST_FILE" | tee -a "$MASTER_LOG"

    CUDA_VISIBLE_DEVICES=$CUSTOM_CUDA_VISIBLE_DEVICES python $project_path/Eval/generate.py \
        --model_name_or_path "$MODEL_PATH" \
        --tensor_parallel_size "$NUM_GPUS" \
        --data_file "$TEST_FILE" \
        --max_tokens 65536 \
        --topk 1 \
        --decoding_method "$DECODING_METHOD" \
        --passk "$PASSK" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$MASTER_LOG"

    echo "Finished dataset: $BENCHMARK_NAME" | tee -a "$MASTER_LOG"
done

echo "=============================================="
echo " All datasets completed."
echo " Logs saved to $MASTER_LOG"
echo "==============================================" | tee -a "$MASTER_LOG"
