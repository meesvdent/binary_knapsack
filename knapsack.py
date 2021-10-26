"""
Contains knapsack class.

Takes two vectors: payload weights and payload values.
Creates these payload instances and stores them in a list.
Will get functions to perform the actual evolutionary algorithm on them.
"""

import numpy as np
from individual import Individual


class Knapsack:

    def __init__(self, n_objects=100, pop_size=100):
        """

        :param n_objects: Integer, number of objects to initialize
        :param constraint: Double, maximum weight of selected objects

        creates weights and values with gaussian distribution
        """
        init_pop = np.random.rand(n_objects, 2)
        self.pop_size = pop_size
        self.n_objects = n_objects
        self.weights = np.array(init_pop[:, 0], dtype=np.double)
        self.values = np.array(init_pop[:, 1], dtype=np.double)
        self.constraint = 0.25 * sum(self.weights)
        self.population = np.empty(pop_size, dtype=Individual)

    def __repr__(self):
        rep_string = ""
        for i in range(len(self.weights)):
            rep_string += str(self.weights[i])
            rep_string += ", "
            rep_string += str(self.values[i])
            rep_string += "\n"
        return rep_string

    def set_population(self, i, array):
        self.population[i] = array

    def get_pop(self):
        return self.population

    def get_constraint(self):
        return self.constraint

    def get_weight(self, i):
        return self.weights[i]

    def get_value(self, i):
        return self.values[i]

    def init_pop(self):
        for i in range(self.pop_size):
            self.set_population(i, Individual(n_objects=self.n_objects))

    def objective_func(self, n):
        individual = self.population[n]
        value = individual.objective_function(self)
        return value









