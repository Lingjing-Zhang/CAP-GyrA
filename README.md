# Introduction
This study proposes a GNN model (chemical activity prediction of GyrA, CAP-GyrA) based on a two-stream neural network architecture, which integrates chemical and BRICS molecular fragment structural representations. The model is designed to predict the inhibitory activity of chemicals on GyrA of Escherichia coli.
![image](https://github.com/user-attachments/assets/2badc132-feef-4a77-9e8a-1159fa2f4ac0)

# Usage
## Data processing
Running molpro.py for chemical structure cleaning and data_split.py for dataset splitting.
## Model training
Users can customize the relevant hyperparameters for each experiment with yaml file in configs folder. The experiment can be run via the command:python ./hyperparameter-tuning/train.py --cfg ./configs/a_input_pro/a_input_pro.yaml --opts 'SEED' 3407 'MODEL.BRICS' True 'MODEL.F_ATT' True --tag seed_3407
## Hyperparameter tuning
## Interpretation of result
