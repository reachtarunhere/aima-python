from collections import defaultdict, Counter
import itertools
import math
import random

from utils import weighted_sample_with_replacement

def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


class BayesNet(object):
    "Bayesian network: a graph of variables connected by parent links."
     
    def __init__(self): 
        self.variables = [] # List of variables, in parent-first topological sort order
        self.lookup = {}    # Mapping of {variable_name: variable} pairs
            
    def add(self, name, parentnames, cpt):
        "Add a new Variable to the BayesNet. Parentnames must have been added previously."
        parents = [self.lookup[name] for name in parentnames]
        var = Variable(name, cpt, parents)
        self.variables.append(var)
        self.lookup[name] = var
        return self
    
class Variable(object):
    "A discrete random variable; conditional on zero or more parent Variables."
    
    def __init__(self, name, cpt, parents=()):
        "A variable has a name, list of parent variables, and a Conditional Probability Table."
        self.__name__ = name
        self.parents  = parents
        self.cpt      = CPTable(cpt, parents)
        self.domain   = set(itertools.chain(*self.cpt.values())) # All the outcomes in the CPT
                
    def __repr__(self): return self.__name__
    
class ProbDist(dict):
    """A Probability Distribution is an {outcome: probability} mapping.
    The values are normalized to sum to 1.
    ProbDist(0.75) is an abbreviation for ProbDist({T: 0.75, F: 0.25})."""
    def __init__(self, mapping=(), **kwargs):
        if isinstance(mapping, float):
            mapping = {T: mapping, F: 1 - mapping}
        self.update(mapping, **kwargs)
        normalize(self)

    def sample(self, n):
        return weighted_sample_with_replacement(self.keys(), self.values(), n)
        
class Evidence(dict): 
    "A {variable: value} mapping, describing what we know for sure."
        
class CPTable(dict):
    "A mapping of {row: ProbDist, ...} where each row is a tuple of values of the parent variables."
    
    def __init__(self, mapping, parents=()):
        """Provides two shortcuts for writing a Conditional Probability Table. 
        With no parents, CPTable(dist) means CPTable({(): dist}).
        With one parent, CPTable({val: dist,...}) means CPTable({(val,): dist,...})."""
        if len(parents) == 0 and not (isinstance(mapping, dict) and set(mapping.keys()) == {()}):
            mapping = {(): mapping}
        for (row, dist) in mapping.items():
            if len(parents) == 1 and not isinstance(row, tuple): 
                row = (row,)
            self[row] = ProbDist(dist)

class Bool(int):
    "Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"
    __str__ = __repr__ = lambda self: 'T' if self else 'F'
        
T = Bool(True)
F = Bool(False)

class Factor(dict):

    def __init__(self, variables, prob_mapping):
        self.variables = variables
        self.update(prob_mapping)

    def __getitem__(self, mapping):
        "Return probability of row as key or filter mapping to obtain row and return its probability."
        if isinstance(mapping, dict):
            mapping = filter_event_values(self.variables, mapping)  # filter if type is dict
        prob = dict.__getitem__(self, mapping)
        return prob

    def pointwise_product(self, other, evidence={}):
        "Multiply two factors, combining their variables."
        variables = list(set(self.variables) | set(other.variables))
        new_mapping = defaultdict(float)  # new_mapping contains union of vars in both factors
        for new_row in all_consistent_events(variables, evidence):  # new_row is a row in new_mapping
            p_self = self[Evidence(zip(variables, new_row))]
            p_other = other[Evidence(zip(variables, new_row))]
            # prob of new_row is product of compaitible rows in both factors.
            new_mapping[new_row] = p_self * p_other
        return Factor(variables, new_mapping)

    def sum_out(self, var, evidence={}):
        "Make a factor eliminating var by summing over its values."
        remaining_vars = [X for X in self.variables if X != var]
        new_mapping = defaultdict(float)
        for new_row in all_consistent_events(remaining_vars, evidence):  # new_row is a row in new_mapping
            # prob of new_row is given by sum of prob of compaitable rows in current dist 
            new_mapping[new_row] = sum(self[row] for row in self if matches_evidence(row,
                                       Evidence(zip(remaining_vars, new_row)), self.variables))
        return Factor(remaining_vars, new_mapping)


def matches_evidence(row, evidence, var_ordering):
    "Does the tuple of values for this row agree with the evidence given variable ordering of the tuple?"
    return all(evidence[v] == row[var_ordering.index(v)]
               for v in evidence)


def all_consistent_events(variables, evidence={}):
    "All possible events in joint dist of variables"
    all_events =  itertools.product(*[var.domain for var in variables])
    return [row for row in all_events if matches_evidence(row, evidence, variables)]


def filter_event_values(variable_list, event):
    "Filter and reorder event values to only include vars in variable_list in correct order."
    return tuple(event[var] for var in variable_list)


def variable_to_factor(var, evidence={}):
    # As a convention the last variable will be the variable itself
    variables = var.parents
    variables.append(var)
    factor_mapping = {}
    for row in var.cpt:
        for value, prob in var.cpt[row].items():
            new_row = row + (value,)
            if matches_evidence(new_row, evidence, variables):
                factor_mapping[new_row] = prob

    return Factor(variables, factor_mapping)