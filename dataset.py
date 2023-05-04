import csv, random
from ASOhelpers import parse_reaction
from reaction import CScoupling
from sklearn.model_selection import train_test_split

class Dataset():
    
    def __init__(self) -> None:
        pass
    
    def load_data(self):
        #return a tuple of Dataloader of training set and validation set
        pass

    def load_cross_validation(self):
        #return a list of tuple of Dataloaders for cross validation 
        pass

class ASODataset(Dataset):
    initiated = False
    reaction_list = None
    ligand_dict = None

    def __init__(self, graph_fn, learning_from = "ee", no_graph = False):
        self.reaction_list = self.read_from_csv()
        #self.reaction_list should be a list of Reaction object
        self.no_graph = no_graph
        self.data = graph_fn(self.reaction_list, learning_from)
        #self.data should be a list of torch_geometric.data.Data
        
    def split_data(self, test_size, random_state, shuffle = True):
        return train_test_split(self.data, test_size = test_size, random_state = random_state, shuffle = shuffle)
    
    def sample_test_data(self, acid_num, thiol_num, imine_num):
        #return a training set and at test set
        #randomly select acid_num numbers of acids to be in the test set, same for thiol and imine
        assert acid_num < 43, "Please sample less than 43 phosphorus acid"
        assert thiol_num < 5, "Please sample less than 5 thiols"
        assert imine_num < 5, "Please sample less than 5 imines"

        acids = list(self.ligand_dict.values())
        thiols = ['A','B','C','D','E']
        imines = ['1','2','3','4','5']

        test_acids = set(random.sample(acids, k=acid_num))
        test_thiols = set(random.sample(thiols, k=thiol_num))
        test_imines = set(random.sample(imines, k=imine_num))

        
        if not self.no_graph:
            training_set = []
            test_set = []
            for d in self.data:
                if d.acid in test_acids and d.thiol in test_thiols and d.imine in test_imines:
                    test_set.append(d)
                elif (d.acid not in test_acids) and (d.thiol not in test_thiols) and (d.imine not in test_imines):
                    training_set.append(d)
            return training_set, test_set
        else:
            train_X = []
            test_X = []
            train_y = []
            test_y = []
            for j in range(len(self.data[0])):
                X = self.data[0][j]
                y = self.data[1][j]
                a_sampled = False
                t_sampled = False
                i_sampled = False
                for i in range(43):
                    if X[i] == 1:
                        if acids[i] in test_acids:
                            a_sampled = True
                        break
                for i in range(5):
                    if X[i+43] == 1:
                        if thiols[i] in test_thiols:
                            t_sampled = True
                        break
                for i in range(5):
                     if X[i+48] == 1:
                        if imines[i] in test_imines:
                            i_sampled = True
                        break
                if a_sampled and t_sampled and i_sampled:
                    test_X.append(X)
                    test_y.append(y)
                elif (not a_sampled) and (not t_sampled) and (not i_sampled):
                    train_X.append(X)
                    train_y.append(y)
            return train_X, test_X, train_y, test_y

    def sample_test_data_catalyst_only(self, acid_num, thiol_num, imine_num):
        #return a training set and at test set
        #randomly select acid_num numbers of acids to be in the test set, same for thiol and imine
        assert acid_num < 43, "Please sample less than 43 phosphorus acid"
        assert thiol_num < 5, "Please sample less than 5 thiols"
        assert imine_num < 5, "Please sample less than 5 imines"

        acids = list(self.ligand_dict.values())

        test_acids = set(random.sample(acids, k=acid_num))

        
        if not self.no_graph:
            training_set = []
            test_set = []
            for d in self.data:
                if d.acid in test_acids:
                    test_set.append(d)
                elif d.acid not in test_acids:
                    training_set.append(d)
            return training_set, test_set
        else:
            train_X = []
            test_X = []
            train_y = []
            test_y = []
            for j in range(len(self.data[0])):
                X = self.data[0][j]
                y = self.data[1][j]
                a_sampled = False
                for i in range(43):
                    if X[i] == 1:
                        if acids[i] in test_acids:
                            a_sampled = True
                        break
                if a_sampled:
                    test_X.append(X)
                    test_y.append(y)
                else:
                    train_X.append(X)
                    train_y.append(y)
            return train_X, test_X, train_y, test_y
        
    def sample_test_data_reactant_only(self, acid_num, thiol_num, imine_num):
        #return a training set and at test set
        #randomly select acid_num numbers of acids to be in the test set, same for thiol and imine
        assert acid_num < 43, "Please sample less than 43 phosphorus acid"
        assert thiol_num < 5, "Please sample less than 5 thiols"
        assert imine_num < 5, "Please sample less than 5 imines"

        thiols = ['A','B','C','D','E']
        imines = ['1','2','3','4','5']

        test_thiols = set(random.sample(thiols, k=thiol_num))
        test_imines = set(random.sample(imines, k=imine_num))

        
        if not self.no_graph:
            training_set = []
            test_set = []
            for d in self.data:
                if d.thiol in test_thiols and d.imine in test_imines:
                    test_set.append(d)
                elif (d.thiol not in test_thiols) and (d.imine not in test_imines):
                    training_set.append(d)
            return training_set, test_set
        else:
            train_X = []
            test_X = []
            train_y = []
            test_y = []
            for j in range(len(self.data[0])):
                X = self.data[0][j]
                y = self.data[1][j]
                t_sampled = False
                i_sampled = False
                for i in range(5):
                    if X[i+43] == 1:
                        if thiols[i] in test_thiols:
                            t_sampled = True
                        break
                for i in range(5):
                     if X[i+48] == 1:
                        if imines[i] in test_imines:
                            i_sampled = True
                        break
                if t_sampled and i_sampled:
                    test_X.append(X)
                    test_y.append(y)
                elif (not t_sampled) and (not i_sampled):
                    train_X.append(X)
                    train_y.append(y)
            return train_X, test_X, train_y, test_y

    def read_from_csv(self):
        #generate a dictionary of ligand
        if ASODataset.initiated:
            return self.reaction_list

        with open('ligands.csv') as f:
            reader = csv.reader(f)
            ligands = list(reader)
        f.close()
        ligand_dict = {}
        for p in ligands:
            ligand_dict[p[0]] = p[1]
        
        #read the reactions from csv
        with open('exps.csv') as f:
            reader = csv.reader(f)
            reactions = list(reader)
        f.close()
        
        reaction_list = []
        for r in reactions:
            data = r[0].split()
            y = (float(data[-1]) + 100)/ 200
            reaction_string = data[0]
            reactants, reactants_label, catalyst, catalyst_label = parse_reaction(reaction_string, ligand_dict)
            reaction_list.append(CScoupling(reactants, reactants_label, catalyst, catalyst_label, y))

        #don't need to read again if there's another ASODataset object
        ASODataset.initiated = True
        ASODataset.reaction_list = reaction_list
        ASODataset.ligand_dict = ligand_dict
        return reaction_list