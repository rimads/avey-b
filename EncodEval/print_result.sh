#!/bin/bash

# Model name variables
MODEL1="avey"
MODEL2="avey-model"

benchmarks=(
  "./results/main/results/${MODEL1}/SC/mnli_m"
  "./results/main/results/${MODEL1}/SC/qqp"
  "./results/main/results/${MODEL1}/SC/sst2"
  "./results/main/results/${MODEL1}/TC/conll2003_en"
  "./results/main/results/${MODEL1}/TC/ontonotes"
  "./results/main/results/${MODEL1}/TC/uner_en"
  "./results/main/results/${MODEL1}/QA/record"
  "./results/main/results/${MODEL1}/QA/squad"
  "./results/main/results/${MODEL1}/QA/squad_v2"
  "./results/main/results/${MODEL2}/IR/mldr_en/msmarco_pairs"
  "./results/main/results/${MODEL2}/IR/msmarco/msmarco_pairs"
  "./results/main/results/${MODEL2}/IR/nq/msmarco_pairs"
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
      (IFS=,; echo "SPLIT(\"${row[*]}\", \",\")")
  done
  echo
done
