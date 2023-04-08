# GNN_enantioselectivity
This is a project of using graph neural network to predict enantioselectivity

dataset.py defines a Dataset class for loading data from .csv

reaction.py defines a Reaction class. This is just a helper data structure that contains information of a single datum, including the reactants and enantioselectivity

graph_fn.py defines functions that change a Reaction object to a torch_geometric.data.Data on which the models will actually use. The Reaction to Data abstraction is needed because we want different encoding of data. So we could define different functions in graph_fn.py.

ASOhelpers.py gives helper functions for dataset.py

models.py defines the models used to train.

trainer.py defines the Trainer class to train the model.

In the folder runs, there are the results of different models or encoding (the .ipynb files)


To run the jupyter notebook, please make sure that the csv files are under the same folder as the notebook is.
