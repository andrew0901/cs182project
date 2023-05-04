# GNN_enantioselectivity
This is a project of using graph neural network to predict enantioselectivity

Dependencies:
rdkit, torch, torch_geometric, sklearn, numpy, matplotlib



Results:
The notebooks for investigation of data representation (graph & one-hot encoding) is in the folder data_representation_exps
The ones for investigation of convolution layers are in folder conv_layers_exps
Hyperparameter tuning is in hypertuning_onehot


Data Structures:
dataset.py defines a Dataset class for loading data from .csv
reaction.py defines a Reaction class. This is just a helper data structure that contains information of a single datum, including the reactants and enantioselectivity
graph_fn.py defines functions that change a Reaction object to a torch_geometric.data.Data on which the models will actually use. The Reaction to Data abstraction is needed because we want different encoding of data. So we could define different functions in graph_fn.py.
ASOhelpers.py gives helper functions for dataset.py
models.py defines the models used to train.
trainer.py defines the Trainer class to train the model.

Original Data:
The enantioselectivity of the reactions are in exps.csv
The ligand structures (SMILES strings) are in ligands.csv
The naming and numbering of each structure is manually copied from paper: 
Zahrt, A. F., Henle, J. J., Rose, B. T., Wang, Y., Darrow, W. T., & Denmark, S. E. (2019). Prediction of higher-selectivity catalysts by computer-driven workflow and machine learning. Science (New York, N.Y.), 363(6424), eaau5631.
