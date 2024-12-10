#!/bin/bash

####################################
# DEFINE RUN PARAMETERS
####################################
dataset="TextMine_2025"
num_class=38

model_path="xlm-roberta-large" # set model path
model_name="${HF_model_name#*/}"

# Recupère les valeurs de batch size et du numéro de run depuis l'environnement
batch_size=${BATCH_SIZE:-4}   # Batch size par défaut 4
run_number=${RUN_NUMBER:-1}   # Run number par défaut 1
###################################

# DEFINE USEFUL PATHS
datetime=$(date +"%d-%m-%Y_%H-%M")
current_run="./results/LACE/${dataset}/${model_name}_finetuning_full_train_bs=${batch_size}_run=${run_number}__${datetime}"

data_dir="./data/formatted_as_docred/${dataset}"
save_finetuned_model_path="${current_run}/finetuned_models/model_finetuned.pt"
pred_path="${current_run}/pred.json"
results_path="${current_run}/results.txt"

adjacency_matrix_save_path="./LACE/meta/${dataset}_adj.pkl"
label_embeddings_save_path="./LACE/meta/${dataset}_${model_name}_label_embeddings.pkl"
#######################

# training
python -u LACE/train_on_full_training_set.py \
    --data_dir $data_dir \
    --transformer_type roberta \
    --model_name_or_path $model_path \
    --data_file data.json \
    --test_file test_01-07-2024.json \
    --rel2id_file rel2id.json \
    --save_finetuned_model_path $save_finetuned_model_path \
    --pred_path $pred_path \
    --results_path $results_path \
    --adjacency_matrix_save_path $adjacency_matrix_save_path \
    --label_embeddings_save_path $label_embeddings_save_path \
    --train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --num_labels $num_class \
    --learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 30 \
    --seed 66 \
    --num_class $num_class \
    --multi_label
