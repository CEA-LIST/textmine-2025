#!/bin/bash

#SBATCH --job-name=lace_xlm_roberta
#SBATCH --output=./results/logs_lace/output_%j.txt
#SBATCH --error=./results/logs_lace/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu80G
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00   # Maximum execution time (hh:mm:ss)
#SBATCH --mem=32G         # Memory required (per node)

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=ALL
### start,end,fail
#SBATCH --mail-user=arthur.peuvot@cea.fr

# Affiche la machine(s)
echo "Begin on machine :"
echo $HOSTNAME

# Affiche le (ou les gpus) alloués par Slurm pour ce job
echo $CUDA_VISIBLE_DEVICES

# Chargement de l'environnement
. ~/.bashrc
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate textmine

# Initialisation des chemins pour CUDA
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so.11.0:$LD_LIBRARY_PATH


####################################
# DEFINE RUN PARAMETERS
####################################
# project_path="/linkhome/rech/genqsn01/uqa69bx/TextMine_2025"
project_path="/home/users/apeuvot/TextMine_2025_github"
dataset="TextMine_2025"
num_class=38

HF_model_name="xlm-roberta-large"
model_path="/home/data/dataset/huggingface/LLMs/${HF_model_name}"
# model_path="/gpfsdswork/dataset/HuggingFace_Models/${HF_model_name}"
model_name="${HF_model_name#*/}"

# Recupère les valeurs de batch size et du numéro de run depuis l'environnement
batch_size=${BATCH_SIZE:-4}   # Batch size par défaut 4
run_number=${RUN_NUMBER:-1}   # Run number par défaut 1
###################################

# DEFINE USEFUL PATHS
datetime=$(date +"%d-%m-%Y_%H-%M")
current_run="${project_path}/results/LACE/${dataset}/${model_name}_finetuning_full_train_bs=${batch_size}_run=${run_number}__${datetime}"

data_dir="${project_path}/data/formatted_as_docred/${dataset}"
save_finetuned_model_path="${current_run}/finetuned_models/model_finetuned.pt"
pred_path="${current_run}/pred.json"
results_path="${current_run}/results.txt"

adjacency_matrix_save_path="${project_path}/LACE/meta/${dataset}_adj.pkl"
label_embeddings_save_path="${project_path}/LACE/meta/${dataset}_${model_name}_label_embeddings.pkl"
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
    --num_train_epochs 2 \
    --seed 66 \
    --num_class $num_class \
    # --multi_label
