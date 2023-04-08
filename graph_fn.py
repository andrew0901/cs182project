from ASOhelpers import *
from reaction import CScoupling
from torch_geometric.data import Data

def onehot_reactant_batch(list_of_reaction, learning_from):
    #list_of_reaction should be a list of CScoupling object as defined in reaction.py
    #return a list of torch_geometric.data.Data
    results = []
    for r in list_of_reaction:
        results.append(onehot_reactant(r, learning_from))
    return results

def onehot_reactant(CScoupling : CScoupling, learning_from):
    #transform a Reaction object to the torch graph object (torch_geometric.data.Data)
    #the resulting graph object have thiol and imine one-hot encoded, but phosphoric acids are one hot-encoded
    assert learning_from == "ee" or learning_from == "ddG", "Please choose from ee or ddG"

    acid, thiol, imine, t_key, i_key = CScoupling.phos_acid, CScoupling.thiol, CScoupling.imine, CScoupling.thiol_label, CScoupling.imine_label
    
    x_a, edge_index_a, edge_attr_a = generate_attrs(acid)
    x_t, edge_index_t, edge_attr_t = generate_attrs(thiol)
    x_i, edge_index_i, edge_attr_i = generate_attrs(imine)

    thiols = ['A','B','C','D','E']
    imines = ['1','2','3','4','5']

    #just a dummy one hot encoding for thiols and imines
    thiol_vec = []
    imine_vec = []
    for i in range(5):
        if t_key == thiols[i]:
            thiol_vec.append(1)
        else:
            thiol_vec.append(0)
        if i_key == imines[i]:
            imine_vec.append(1)
        else:
            imine_vec.append(0)

    '''
    r = ReactionData(edge_index_a=edge_index_a, edge_attr_a=edge_attr_a, x_a=x_a,
                       edge_index_t=edge_index_t, edge_attr_t=edge_attr_t, x_t=x_t,
                       edge_index_i=edge_index_i, edge_attr_i=edge_attr_i, x_i=x_i, y=y)
                       '''
    if learning_from == "ddG":
        y = np.float32(np.log((1 + CScoupling.y) / (1 - CScoupling.y))*8.314*298/4200)
    else:
        y = CScoupling.y

    r = Data(edge_index=edge_index_a, edge_attr=edge_attr_a, x=x_a, y = y)
    r['reactant_vec'] = torch.tensor(thiol_vec + imine_vec)  
    r.acid = acid
    r.thiol = t_key
    r.imine = i_key
    return r

def single_graph_batch(list_of_reaction, learning_from):
    #put phosphoric acid, thiol and imine in the same graph as unconnected components
    assert learning_from == "ee" or learning_from == "ddG", "Please choose from ee or ddG"
    results = []
    for r in list_of_reaction:
        results.append(single_graph(r, learning_from))
    return results

def single_graph(cscoupling: CScoupling, learning_from):
    acid, thiol, imine = cscoupling.phos_acid, cscoupling.thiol, cscoupling.imine
    x, edge_index, edge_attr = generate_attrs_all(acid, thiol, imine)
    
    if learning_from == "ddG":
        y = np.float32(np.log((1 + cscoupling.y) / (1 - cscoupling.y))*8.314*298/4200)
    else:
        y = cscoupling.y

    r = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y = y)
    r.acid = acid
    r.thiol = cscoupling.thiol_label
    r.imine = cscoupling.imine_label

    
    return r

def onehot_everything(list_of_reaction, learning_from):
    #one hot encode everything, still returning a list of torch_geometric.data.Data for simplicity in Trainer class
    assert learning_from == "ee" or learning_from == "ddG", "Please choose from ee or ddG"
    acid_count = 0
    thiol_count = 0
    imine_count = 0
    acid_dict, thiol_dict, imine_dict = {}, {}, {}
    for r in list_of_reaction:
        if r.phos_acid not in acid_dict:
            acid_dict[r.phos_acid] = acid_count
            acid_count += 1
        if r.thiol not in thiol_dict:
            thiol_dict[r.thiol] = thiol_count
            thiol_count += 1
        if r.imine not in imine_dict:
            imine_dict[r.imine] = imine_count
            imine_count += 1
    results = [[], []]
    for r in list_of_reaction:
        one_hot_acid = np.zeros(acid_count, dtype = int)
        one_hot_acid[acid_dict[r.phos_acid]] = 1

        one_hot_thiol = np.zeros(thiol_count, dtype = int)
        one_hot_thiol[thiol_dict[r.thiol]] = 1

        one_hot_imine = np.zeros(imine_count, dtype = int)
        one_hot_imine[imine_dict[r.imine]] = 1

        if learning_from == "ddG":
            y = np.float32(np.log((1 + r.y) / (1 - r.y))*8.314*298/4200)
        else:
            y = r.y

        results[0].append(np.concatenate((one_hot_acid, one_hot_thiol, one_hot_imine)))
        results[1].append(y)
    results[0] = np.array(results[0])
    results[1] = np.array(results[1])
    return results