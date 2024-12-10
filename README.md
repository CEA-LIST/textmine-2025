# TextMine 2025 

This repository contains the experiments conducted for the **[TextMine 2025 challenge](https://www.kaggle.com/competitions/defi-text-mine-2025/)**. 


The majority of our work focused on **[ATLOP-based models](https://github.com/wzhouad/ATLOP/tree/main)**, namely **[LACE](https://github.com/LUMIA-Group/LACE/tree/main)** and **[EIDER](https://github.com/yiqingxyq/Eider/tree/main)**. 


## Install Environment

The experiments were conducted using `Python 3.9.19`.

#### Installing Dependencies
First, install the required dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

#### Installing DGL
If running on GPUs, install the appropriate version of **DGL** based on your CUDA version. Replace `XXX` with the CUDA version you are using:
```bash
pip install dgl==1.0.0+cuXXX -f https://data.dgl.ai/wheels/cuXXX/repo.html
```

If not using GPUs, install the CPU version:
```bash
pip install dgl==1.0.0
```

#### Installing Apex
Install **Apex** by following the instructions available at the [NVIDIA Apex GitHub repository](https://github.com/NVIDIA/apex). Our experiments have been run under `apex==0.1`


## Running the Models

There are three main approaches for training and evaluating the models:

- **Train on 80% of the dataset, evaluate on 10%, and test on the remaining 10%:**   
   In this setup, the proportions of training, evaluation, and testing datasets are maintained for each relation type.  
   Example command:
   ```bash
   sh ./LACE/scripts/run_texmine_xlm_roberta.sh
   ```

- **Cross-validation:**   
   This approach uses 5-fold cross-validation to ensure robust evaluation.   
   Example command:
   ```bash
   sh ./LACE/scripts/run_texmine_xlm_roberta_cross_validation.sh
   ```

- **Train on the entire training set and predict on the test set:**   
   This method was used for to get predictions for the submissions.   
   Example command:
   ```bash
   sh ./LACE/scripts/run_texmine_xlm_roberta_full_train.sh
   ```


## Model Combination

To improve the robustness and accuracy of predictions, we implemented a **voting-based model combination strategy**. This approach aggregates the predictions from multiple models using a voting mechanism.

Two voting methods are available:

- **Absolute Majority**:   
A prediction is selected only if more than 50% of the models made this prediction.

- **Relative Majority**:  
For a given pair of entities, the prediction with the highest number of votes is selected, even if it does not exceed the 50% threshold.


Command: 
   ```bash
   sh ./combine/scripts/run_vote.sh
   ```