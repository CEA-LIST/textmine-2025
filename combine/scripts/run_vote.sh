#!/bin/bash

# Déclaration des variables
save_folder="" # set path 
training_method="" # LACE, EIDER...
vote_method="absolute_majority"  # choices=['absolute_majority', 'relative_majority']
train_data_path="./data/formatted_as_docred/${dataset}/data.json"
files_path="${save_folder}/${training_method}_files.txt"
pred_path="${save_folder}/${training_method}_${vote_method}_combined_preds.csv"
results_path="${save_folder}/results_${training_method}_${vote_method}.txt"
unfiltered_results_path="${save_folder}/unfiltered_results_${training_method}_${vote_method}.txt"

# Exécution du script Python
python /path/to/combine/vote.py \
    --files "file1.csv" "file2.csv" "file3.csv" \
    --vote_method ${vote_method} \
    --train_data_path $train_data_path \
    --files_path ${files_path} \
    --results_path ${results_path} \
    --unfiltered_results_path ${unfiltered_results_path} \
    --final_pred_path ${pred_path}