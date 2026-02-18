#!/bin/bash

# Model name
MODEL="avey-b1-base-exp"

benchmarks=(
  "./${MODEL}/results/SC/mnli_m"
  "./${MODEL}/results/SC/qqp"
  "./${MODEL}/results/SC/sst2"
  "./${MODEL}/results/TC/conll2003_en"
  "./${MODEL}/results/TC/ontonotes"
  "./${MODEL}/results/TC/uner_en"
  "./${MODEL}/results/QA/record"
  "./${MODEL}/results/QA/squad"
  "./${MODEL}/results/QA/squad_v2"
  "./${MODEL}/results/IR/mldr_en/msmarco_pairs"
  "./${MODEL}/results/IR/msmarco/msmarco_pairs"
  "./${MODEL}/results/IR/nq/msmarco_pairs"
  "./${MODEL}/results/QA/niah1"
  "./${MODEL}/results/QA/niah2"
)

# lrs=("2e-05" "6e-05" "1e-04" "5e-04")
lrs=("5e-04")

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
