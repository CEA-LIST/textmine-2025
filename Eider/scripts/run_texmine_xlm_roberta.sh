#!/bin/bash

####################################
# DEFINE RUN PARAMETERS
####################################
dataset="TextMine_2025"
num_class=38

model_path="xlm-roberta-large" # set model path
model_name="${HF_model_name#*/}"

# evidence_construction="predictions_and_rule_based"
evidence_construction="rule_based" 


# Recupère les valeurs de batch size et du numéro de run depuis l'environnement
batch_size=${BATCH_SIZE:-4}   # Batch size par défaut 4
run_number=${RUN_NUMBER:-1}   # Run number par défaut 1
learning_rate=${LEARNING_RATE:-3e-5}   # Learning rate par défaut 3e-5
lr_added=${LR_ADDED:-1e-4}   # Learning rate pour couches ajoutées par défaut 1e-4
warmup_ratio=${WARMUP_RATIO:-0.06}   # Warmup ration par défaut 0.06
adam_epsilon=${ADAM_EPSILON:-1e-6} 
###################################

# DEFINE USEFUL PATHS
datetime=$(date +"%d-%m-%Y_%H-%M")
current_run="./results/Eider/${dataset}/${model_name}_finetuning_ml_rb_evidences_bs=${batch_size}_run=${run_number}__${datetime}"


data_dir="./data/formatted_as_docred/${dataset}"
save_finetuned_model_path="${current_run}/finetuned_models/model_finetuned.pt"
pred_path="${current_run}/pred.json"
results_path="${current_run}/results.txt"
#######################

# training
python -u Eider/train.py \
    --data_dir $data_dir \
    --transformer_type xlm-roberta \
    --model_name_or_path $model_path \
    --data_file_with_silver_evidence data_with_silver_evidence.json \
    --rel2id_file rel2id.json \
    --save_finetuned_model_path $save_finetuned_model_path \
    --pred_path $pred_path \
    --results_path $results_path \
    --evidence_construction $evidence_construction \
    --train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --num_labels $num_class \
    --learning_rate ${learning_rate} \
    --lr_added ${lr_added} \
    --adam_epsilon ${adam_epsilon} \
    --max_grad_norm 1.0 \
    --warmup_ratio ${warmup_ratio} \
    --num_train_epochs 30 \
    --seed 66 \
    --num_class $num_class \
    --ablation eider \
    --coref_method none \
    --multi_label

