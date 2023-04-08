class Reaction():
    #the data structure Reaction is just a class that helps with abstraction
    #the graph functions defined in graph_fn.py will create pytorch graph objects from Reaction object

    def __init__(self, reactants, reactants_label, catalyst, catalyst_label, y) -> None:
       self.reactants = reactants
       self.catalyst = catalyst
       self.reactants_label = reactants_label
       self.catalyst_label = catalyst_label
       self.y = y
    
    def reactants(self):
        return self.reactants
    
    def reactants_label(self):
        return self.reactants_label
    
    def catalyst(self):
        return self.catalyst
    
    def catalyst_label(self):
        return self.catalyst_label
    
    def y(self):
        return self.y

class CScoupling(Reaction):
    def __init__(self, reactants, reactants_label, catalyst, catalyst_label, y) -> None:
       self.reactants = reactants
       self.catalyst = catalyst
       self.reactants_label = reactants_label
       self.catalyst_label = catalyst_label
       self.y = y
       self.thiol = reactants[0]
       self.thiol_label = reactants_label[0]
       self.imine = reactants[1]
       self.imine_label = self.reactants_label[1]
       self.phos_acid = self.catalyst
       self.phos_label = self.catalyst_label