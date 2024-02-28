#!/bin/bash

EXP_PATH=<path to experiment folder>
DATA_PATH=<path to CrossRE data>

SEEDS=( 4012 5096 8824 8257 9908 )

DOMAIN_TRAIN=( "ai" "literature" "music" "news" "politics" "science" )
DOMAIN_TEST=( "ai" "literature" "music" "news" "politics" "science" )

ATTRIBUTES=( "none" "entity_type" "iv_entities" "entity_length" "entity_distance" "sentence_length" "entity_density" "entity_pair_density" "oov_token_density" "entity_type_frequency" "relation_type_frequency" )

# iterate over attributes
for attribute in "${!ATTRIBUTES[@]}"; do

  # iterate over train sets
  for train in "${!DOMAIN_TRAIN[@]}"; do

    # iterate over seeds
    for rs in "${!SEEDS[@]}"; do
      echo "Experiment on random seed ${SEEDS[$rs]}."

      # iterate over test sets
      for test in "${!DOMAIN_TEST[@]}"; do

        exp_dir=$EXP_PATH/${DOMAIN_TRAIN[$train]}/rs${SEEDS[$rs]}
        echo $exp_dir

        # check if model already exists
        if [ -f "$exp_dir/best.pt" ]; then
          echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
        # if experiment is new, train classifier
        else
          echo "Training model ${TASK} on random seed ${SEEDS[$rs]}."

          # train
          python3 main.py \
                  --train_path "${DATA_PATH}/${DOMAIN_TRAIN[$train]}-train.json" \
                  --dev_path "${DATA_PATH}/${DOMAIN_TRAIN[$train]}-dev.json" \
                  --exp_path ${exp_dir} \
                  --seed ${SEEDS[$rs]}
        fi

        # prediction
        python3 main.py \
                --train_path "${DATA_PATH}/${DOMAIN_TRAIN[$train]}-train.json" \
                --test_path "${DATA_PATH}/${DOMAIN_TEST[$test]}-dev.json" \
                --exp_path ${exp_dir} \
                --seed ${SEEDS[$rs]} \
                --attribute ${ATTRIBUTES[$attribute]} \
                --prediction_only

        # evaluation
        python3 evaluate.py \
                --train_path "${DATA_PATH}/${DOMAIN_TRAIN[$train]}-train.json" \
                --gold_path ${DATA_PATH}/${DOMAIN_TEST[$test]}-dev.json \
                --out_path ${exp_dir} \
                --attribute ${ATTRIBUTES[$attribute]} \
                --summary_exps ${EXP_PATH}/${DOMAIN_TRAIN[$train]}
      done
    done
  done
done