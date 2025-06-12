# Introduction
This study proposes a GNN model (chemical activity prediction of GyrA, CAP-GyrA) based on a two-stream neural network architecture, which integrates chemical and BRICS molecular fragment structural representations. The model is designed to predict the inhibitory activity of chemicals on GyrA of Escherichia coli.

# Usage
## Data processing
Running molpro.py for chemical structure cleaning and data_split.py for dataset splitting.
## Model training
Users can customize the relevant hyperparameters for each experiment with yaml file in configs folder. The experiment can be run via the command: python ./hyperparameter-tuning/train.py --cfg ./configs/a_input_pro/a_input_pro.yaml --opts 'SEED' 3407 'MODEL.BRICS' True 'MODEL.F_ATT' True --tag seed_3407
## Hyperparameter tuning
The experiment can be run via the following command for cross-validation and Bayesian hyperparameter tuning：python ./hyperparameter-tuning/cross_validate.py --cfg ./configs/a_input_pro/a_input_pro.yaml --opts 'SEED' 3407 'MODEL.BRICS' True 'MODEL.F_ATT' True --tag seed_3407
## Interpretation of result
To get the interpretation of predictive result, the heatmap of BRICS contribution can be obtained by running the aw.py in interpretation folder.
