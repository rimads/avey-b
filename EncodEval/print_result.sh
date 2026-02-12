#!/bin/bash

# Model name
MODEL="avey"

benchmarks=(
  "./results/main/results/${MODEL}/SC/mnli_m"
  "./results/main/results/${MODEL}/SC/qqp"
  "./results/main/results/${MODEL}/SC/sst2"
  "./results/main/results/${MODEL}/TC/conll2003_en"
  "./results/main/results/${MODEL}/TC/ontonotes"
  "./results/main/results/${MODEL}/TC/uner_en"
  "./results/main/results/${MODEL}/QA/record"
  "./results/main/results/${MODEL}/QA/squad"
  "./results/main/results/${MODEL}/QA/squad_v2"
  "./results/main/results/${MODEL}/IR/mldr_en/msmarco_pairs"
  "./results/main/results/${MODEL}/IR/msmarco/msmarco_pairs"
  "./results/main/results/${MODEL}/IR/nq/msmarco_pairs"
  "./results/main/results/${MODEL}/QA/niah1"
  "./results/main/results/${MODEL}/QA/niah2"
)

lrs=("2e-05" "6e-05" "1e-04" "5e-04")

for lr in "${lrs[@]}"; do
  echo "### Learning rate: $lr"
  for seed in {0..9}; do
      row=()
      for bm in "${benchmarks[@]}"; do
          # Determine the results.json path
          if [[ "$bm" == *"/IR/"* ]]; then
              # IR tasks: LR is in the filename
              file="${bm}_lr${lr}_sd${seed}/results.json"
          else
              # Other tasks: LR is in the directory
              file="${bm}/lr${lr}_sd${seed}/results.json"
          fi

          if [[ -f "$file" ]]; then
              val=$(jq -r '.test.score // empty' "$file")
              if [[ -n "$val" ]]; then
                  val=$(printf "%.2f" "$(echo "$val * 100" | bc -l)")
              else
                  val="NA"
              fi
          else
              val="NA"
          fi
          row+=("$val")
      done
      (IFS=,; echo "=SPLIT(\"${row[*]}\", \",\")")
  done
  echo
done
