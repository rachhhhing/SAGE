#!/bin/bash

project_path=$(cd $(dirname $0); pwd)

echo "======== Current Path ========"
echo $project_path
echo "================"
cd $project_path

PYTHON_BIN="python3"

INPUT_BASE_DIR="$project_path/outputs"
OUTPUT_BASE_DIR="$project_path/outputs/results"
mkdir -p "$OUTPUT_BASE_DIR"

echo "Starting local execution evaluation"
echo "Input directory: $INPUT_BASE_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "========================================"

input_files=($INPUT_BASE_DIR/*.jsonl)
if [ ${#input_files[@]} -eq 0 ]; then
  echo "Error: No JSONL files found in $INPUT_BASE_DIR"
  exit 1
fi

echo "Found input files:"
for file in "${input_files[@]}"; do
  echo "$file"
done
echo ""

for input_file in "${input_files[@]}"; do
  filename=$(basename "$input_file" .jsonl)
  echo "========================================"
  echo "Processing dataset: $filename"
  echo "Input file: $input_file"
  start_time=$(date)
  echo "Start time: $start_time"
  output_dir="$OUTPUT_BASE_DIR/$filename"
  mkdir -p "$output_dir"

  echo "Using question_field: en_question, answer_field: en_answer"
  if [ -n "$MODEL_NAME" ]; then
    echo "Model name: $MODEL_NAME"
  fi
  echo "Starting execution..."
  echo "Command: $PYTHON_BIN $project_path/execute.py --input_file $input_file --output_dir $output_dir --question_field en_question --answer_field en_answer --timeout 600 --max_workers 4 --passk 1 --use_percentage_err_tolerance true --verbose"
  
  $PYTHON_BIN "$project_path/execute.py" \
      --input_file "$input_file" \
      --output_dir "$output_dir" \
      --question_field en_question \
      --answer_field en_answer \
      --timeout 100 \
      --max_workers 4 \
      --passk 1 \
      --use_percentage_err_tolerance true
  
  if [ $? -ne 0 ]; then
    echo "Execution failed for $filename"
    continue
  fi

  if [ ! -f "$output_dir/evaluation_report.json" ]; then
    echo "Warning: No evaluation_report.json found"
  else
    echo "Evaluation completed for $filename"
  fi

  echo "Completion time: $(date)"
  echo "----------------------------------------"
done

echo "========================================"
echo "All local executions completed"
echo "========================================"

SUMMARY_FILE="$OUTPUT_BASE_DIR/summary_report.txt"
echo "Generating summary report..."
echo "Local Execution Summary Report" > $SUMMARY_FILE
echo "Generation time: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

for input_file in "${input_files[@]}"; do
  filename=$(basename "$input_file" .jsonl)
  output_dir="$OUTPUT_BASE_DIR/$filename"
  if [ -f "$output_dir/evaluation_report.json" ]; then
    acc=$(jq '.accuracy' "$output_dir/evaluation_report.json" 2>/dev/null)
    echo "- $filename: Completed (accuracy=${acc})" >> $SUMMARY_FILE
  else
    echo "- $filename: Failed" >> $SUMMARY_FILE
  fi
done

echo "Summary report generated: $SUMMARY_FILE"
echo "========================================"
