import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error as MAE

class Trainer():
    def __init__(self, model, device, optimizer, scheduler, glob_v = False, no_graph = False):
        #glob_v means if there's any vector that is global to the graph, i.e. the one-hot encoded data that needs to be concatenated after graph-pooling
        #no_graph means if the model contains no graph convolution layers
        self.model = model
        self.deivce = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.glob_v = glob_v
        self.no_graph = no_graph
        self.log = {"train_loss":[], "val_loss":[]}
    
    def train_one_epoch(self, train_loader, loss_fn):
        #should only be called by train
        self.model.train()
        for data in train_loader:
            if not self.glob_v:
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch, data.reactant_vec)
            out = out.reshape(-1)
            y = data.y
            if not torch.is_tensor(data.y):
                y = torch.tensor(data.y)
            loss = loss_fn()(out.reshape(-1), y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def eval(self, test_loader, plot_flag, loss_fn = None, print_flag = False):
        self.model.eval()
        outputs = None
        targets = None
        if not loss_fn:
            loss_fn = self.loss_fn
        with torch.no_grad():
            for data in test_loader:
                if not self.glob_v:
                    out = self.model(data.x, data.edge_index, data.edge_attr, data.batch).cpu()
                else:
                    out = self.model(data.x, data.edge_index, data.edge_attr, data.batch, data.reactant_vec).cpu()
                if outputs == None:
                    y = data.y
                    if not torch.is_tensor(data.y):
                        y = torch.tensor(data.y)
                    targets = y
                    outputs = out
                else:
                    outputs = torch.cat((outputs, out),0)
                    y = data.y
                    if not torch.is_tensor(data.y):
                        y = torch.tensor(data.y)
                    targets = torch.cat((targets, y),0)
            #print(np.array(outputs)-np.array(targets))
        outputs = outputs.reshape(-1)
        r2 = r2_score(targets.numpy(), outputs.numpy())
        if plot_flag:
            plt.figure()
            plt.scatter(targets.numpy(), outputs.numpy())
            x = y = np.linspace(0,1,100)
            plt.plot(x, y, '-r', label='y=x')
        if print_flag:
            test_loss, test_MAE, test_r2 = loss_fn()(outputs, targets), MAE(outputs, targets), r2
            print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_MAE:.4f},Test R2: {test_r2:.4f}')
        return loss_fn()(outputs, targets), MAE(outputs, targets), r2
    
    def train(self, train_loader, test_loader, loss_fn, epochs, silent = False):
        #the function called to trained a model with multiple epochs
        #train_loader and test_loader should be DataLoader object
        self.loss_fn = loss_fn

        best_train_r2 = float("-inf")
        best_test_r2 = float("-inf")
        best_train_MAE = float("inf")
        best_test_MAE = float("inf")

        for epoch in range(epochs):
            self.train_one_epoch(train_loader, loss_fn)
            train_loss, train_MAE, train_r2 = self.eval(train_loader, epoch == epochs - 1, loss_fn=loss_fn)
            test_loss, test_MAE, test_r2 = self.eval(test_loader, epoch == epochs - 1, loss_fn=loss_fn)
            
            best_train_r2 = max(best_train_r2, train_r2)
            best_test_r2 = max(best_test_r2, test_r2)
            best_train_MAE = min(best_train_MAE, train_MAE)
            best_test_MAE = min(best_test_MAE, test_MAE)

            self.log["train_loss"].append(train_loss)
            self.log["val_loss"].append(test_loss)
            if (not silent) or epoch == epochs - 1:
                print(f'Epoch: {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train MAE: {train_MAE:.4f}, Test MAE: {test_MAE:.4f}, , Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
            if epoch == epochs - 1:
                return np.array([train_loss, best_train_MAE, best_train_r2, test_loss, best_test_MAE, best_test_r2])

    def train_one_epoch_no_graph(self, train_X, train_y, loss_fn):
        #used only by train_no_graph
        self.model.train()
        out = self.model(train_X)
        out = out.reshape(-1)
        loss = loss_fn()(out.reshape(-1), train_y.reshape(-1))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def eval_no_graph(self, test_X, test_y, plot_flag, print_flag = False):
        #used only by train_no_graph
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_X)
        outputs = outputs.reshape(-1)
        r2 = r2_score(test_y.numpy(), outputs.numpy())
        if plot_flag:
            plt.figure()
            plt.scatter(test_y.numpy(), outputs.numpy())
            x = y = np.linspace(0,1,100)
            plt.plot(x, y, '-r', label='y=x')
        test_loss, test_MAE, test_r2 = nn.BCELoss()(outputs, test_y), MAE(outputs, test_y), r2
        if print_flag:
            print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_MAE:.4f},Test R2: {test_r2:.4f}')
        return test_loss, test_MAE, test_r2
    
    def train_no_graph(self, train_X, test_X, train_y, test_y, loss_fn, epochs, silent = False):
        #train the model when it is not a graph neural network, just for one hot encoding everything
        best_train_r2 = float("-inf")
        best_test_r2 = float("-inf")
        best_train_MAE = float("inf")
        best_test_MAE = float("inf")

        for epoch in range(epochs):
            self.train_one_epoch_no_graph(train_X, train_y, loss_fn)
            train_loss, train_MAE, train_r2 = self.eval_no_graph(train_X, train_y, epoch == epochs - 1)
            test_loss, test_MAE, test_r2 = self.eval_no_graph(test_X, test_y, epoch == epochs - 1)

            best_train_r2 = max(best_train_r2, train_r2)
            best_test_r2 = max(best_test_r2, test_r2)
            best_train_MAE = min(best_train_MAE, train_MAE)
            best_test_MAE = min(best_test_MAE, test_MAE)

            self.log["train_loss"].append(train_loss)
            self.log["val_loss"].append(test_loss)
            if (not silent) or epoch == epochs - 1:
                print(f'Epoch: {epoch + 1:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train MAE: {train_MAE:.4f}, Test MAE: {test_MAE:.4f}, , Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}')
            if epoch == epochs - 1:
                return np.array([train_loss, best_train_MAE, best_train_r2, test_loss, best_test_MAE, best_test_r2])

    def kfold(self, data, fold, loss_fn, epochs, batch_size, silent = True, shuffle = True):
        #for k-fold cross validation
        #data should be array like
        kf = KFold(n_splits=fold, shuffle=shuffle)
        total_results = np.array([0,0,0,0,0,0])
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            print("Fold " + str(i))
            train_data = [data[j] for j in train_index]
            test_data = [data[k] for k in test_index]
            train_loader = DataLoader(train_data, batch_size=batch_size)
            test_loader = DataLoader(test_data, batch_size=batch_size)

            results = self.train(train_loader, test_loader, loss_fn, epochs, silent = silent)
            total_results += results
        
        average_results = total_results / fold
        print(f'Average Train Loss: {average_results[0]:.4f}, Test Loss: {average_results[3]:.4f}, Train MSE: {average_results[1]:.4f}, Test MSE: {average_results[4]:.4f}, , Train R2: {average_results[2]:.4f}, Test R2: {average_results[5]:.4f}')
    
    def plot_log(self):
        train_log = self.log["train_loss"]
        val_log = self.log["val_loss"]
        x = list(range(len(train_log)))
        plt.figure()
        plt.title("Loss vs Epochs")
        plt.plot(x, train_log, '-b', label="train loss")
        plt.plot(x, val_log, '-r', label = "validation loss")