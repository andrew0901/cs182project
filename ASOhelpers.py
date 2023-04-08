from reaction import Reaction, CScoupling
import rdkit.Chem as Chem
import numpy as np
import torch

reactant_dict = {}
reactant_dict['1'] = 'O=C(/N=C/C1=CC=CC=C1)C2=CC=CC=C2'
reactant_dict['2'] = 'O=C(/N=C/C1=CC=C(C(F)(F)F)C=C1)C2=CC=CC=C2'
reactant_dict['3'] = 'O=C(/N=C/C1=CC=C(OC)C=C1)C2=CC=CC=C2'
reactant_dict['4'] = 'O=C(/N=C/C1=C(C=CC=C2)C2=CC=C1)C3=CC=CC=C3'
reactant_dict['5'] = 'O=C(/N=C/C1=C(C)C=CC=C1)C2=CC=CC=C2'

reactant_dict['A'] = 'SC1=CC=CC=C1'
reactant_dict['B'] = 'SCC'
reactant_dict['C'] = 'SC1CCCCC1'
reactant_dict['D'] = 'SC1=CC=C(OC)C=C1'
reactant_dict['E'] = 'SC1=C(C)C=CC=C1'

elem_list = ['C', 'N', 'O', 'F', 'Si', 'P','Cl', 'Br','unknown']
bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
atom_fdim = len(elem_list) + 5
bond_fdim = 5

#parse the naming of reaction into ligand and two reactants
#the original dataset name the reaction as something like "1_i_1_A"
def parse_reaction(reaction_string, ligand_dict, reactant_dict = reactant_dict):
    thiol = reaction_string[-1]
    reaction_string = reaction_string[:-1]
    reaction_string = reaction_string[:-1]
    imine = reaction_string[-1]
    reaction_string = reaction_string[:-1]
    ligand = reaction_string[:-1]
    return [reactant_dict[thiol], reactant_dict[imine]], [thiol, imine], ligand_dict[ligand], ligand

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))    

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + [atom.GetFormalCharge()] 
            + [atom.GetDegree()]
            + [atom.GetExplicitValence()]
            + [atom.GetImplicitValence()]
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()], dtype=np.float32)

#generate attributes given a smile, returns x,edge_index,edge_attr as required to build a torch_geometric.data.Data (see torch official document)
def generate_attrs(smile):
    mol = Chem.MolFromSmiles(smile)
    
    x = []
    for i in range(mol.GetNumAtoms()):
        x.append(atom_features(mol.GetAtomWithIdx(i)))
    x = torch.tensor(np.array(x), dtype=torch.float)
    
    edge_attr = []
    edge_index = [[],[]]
    bonds = mol.GetBonds()
    for b in bonds:
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[0].append(b.GetEndAtomIdx())
        
        edge_index[1].append(b.GetEndAtomIdx())
        edge_index[1].append(b.GetBeginAtomIdx())
        
        edge_attr.append(bond_features(b))
        edge_attr.append(bond_features(b))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return x,edge_index,edge_attr

#generate reactants and ligands in a single graph as not connected components
def generate_attrs_all(ligand_smile,thiol_smile,imine_smile):
    ligand = Chem.MolFromSmiles(ligand_smile)
    thiol = Chem.MolFromSmiles(thiol_smile)
    imine = Chem.MolFromSmiles(imine_smile)
    
    ligand_atom_num = ligand.GetNumAtoms()
    thiol_atom_num = thiol.GetNumAtoms()
    imine_atom_num = imine.GetNumAtoms()
    
    x = []
    for i in range(ligand_atom_num):
        x.append(atom_features(ligand.GetAtomWithIdx(i)))
    for i in range(thiol_atom_num):
        x.append(atom_features(thiol.GetAtomWithIdx(i)))
    for i in range(imine_atom_num):
        x.append(atom_features(imine.GetAtomWithIdx(i)))
    x = torch.tensor(np.array(x), dtype=torch.float)
    
    
    edge_attr = []
    edge_index = [[],[]]
    bonds = ligand.GetBonds()
    for b in bonds:
        edge_index[0].append(b.GetBeginAtomIdx())
        edge_index[0].append(b.GetEndAtomIdx())
        
        edge_index[1].append(b.GetEndAtomIdx())
        edge_index[1].append(b.GetBeginAtomIdx())
        
        edge_attr.append(bond_features(b))
        edge_attr.append(bond_features(b))
        
    bonds = thiol.GetBonds()
    for b in bonds:
        edge_index[0].append(b.GetBeginAtomIdx() + ligand_atom_num)
        edge_index[0].append(b.GetEndAtomIdx()+ ligand_atom_num)
        
        edge_index[1].append(b.GetEndAtomIdx() + ligand_atom_num)
        edge_index[1].append(b.GetBeginAtomIdx() + ligand_atom_num)
        
        edge_attr.append(bond_features(b))
        edge_attr.append(bond_features(b))
    
    bonds = imine.GetBonds()
    for b in bonds:
        edge_index[0].append(b.GetBeginAtomIdx() + ligand_atom_num + thiol_atom_num)
        edge_index[0].append(b.GetEndAtomIdx()+ ligand_atom_num + thiol_atom_num)
        
        edge_index[1].append(b.GetEndAtomIdx() + ligand_atom_num + thiol_atom_num)
        edge_index[1].append(b.GetBeginAtomIdx() + ligand_atom_num + thiol_atom_num)
        
        edge_attr.append(bond_features(b))
        edge_attr.append(bond_features(b))
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return x,edge_index,edge_attr