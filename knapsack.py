"""
Contains knapsack class.

Takes two vectors: payload weights and payload values.
Creates these payload instances and stores them in a list.
Will get functions to perform the actual evolutionary algorithm on them.
"""

import numpy as np
import random
from individual import Individual
import matplotlib.pyplot as plt



class Knapsack:

    def __init__(self):
        """

        :param n_objects: Integer, number of objects to initialize
        :param constraint: Double, maximum weight of selected objects

        creates weights and values with gaussian distribution
        """

        # parameters
        self.alpha = 0.05
        self.k = 3
        self.n_offspring = 200
        self.n_pop_size = 100
        self.init_pop_size = self.n_pop_size
        self.iter = 100
        self.n_payload = 100

        # generate payload=weights and values
        init_payload = np.random.rand(self.n_payload, 2)
        self.weights = np.array(init_payload[:, 0], dtype=np.double)
        self.values = np.array(init_payload[:, 1], dtype=np.double)
        self.constraint = 0.25 * sum(self.weights)

        # declare population array
        self.population = np.empty(self.init_pop_size, dtype=Individual)
        self.offspring = []

    def __repr__(self):
        rep_string = ""
        for i in range(len(self.weights)):
            rep_string += str(self.weights[i])
            rep_string += ", "
            rep_string += str(self.values[i])
            rep_string += "\n"
        return rep_string

    def evolution(self):
        mean = []
        best = []
        iteration = []
        for i in range(self.iter):
            self.offspring = []
            for j in range(self.n_offspring):
                parent_one = self.population[self.k_tournament(self.k)]
                parent_two = self.population[self.k_tournament(self.k)]
                child = self.recombinate(parent_one, parent_two)
                self.offspring.append(child)
                mutated_offspring = self.mutate(self.offspring, alpha=self.alpha)
            self.population = np.concatenate((self.population, mutated_offspring))
            self.eliminate(self.n_pop_size)

            print("Iteration: ", i)
            objective_values = [x.get_value(self) for x in self.population]
            print(np.mean(objective_values))
            mean.append(np.mean(objective_values))
            print(np.max(objective_values))
            best.append(np.max(objective_values))
            iteration.append(i)
            print("Population size: ", len(self.population))
        plt.plot(iteration, best, label="Best value")
        plt.plot(iteration, mean, label="Mean value")
        plt.show()

        print("Heuristic solution: ", self.calculate_heuristic())

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
        for i in range(self.init_pop_size):
            order = np.random.permutation(range(self.n_payload))
            self.set_population(i, Individual(order))

    def objective_func(self, n):
        individual = self.population[n]
        value = individual.objective_function(self)
        return value

    def k_tourn_elim(self, k):
        selected = random.sample(range(len(list(self.population))), k)
        individuals = self.population[selected]
        values = [individual.get_value(self) for individual in individuals]
        return selected[np.argmin(values)]

    def eliminate(self, new_pop_size):
        n_elim = len(self.population) - new_pop_size
        for i in range(n_elim):
            to_delete = self.k_tourn_elim(5)
            self.population = np.delete(self.population, to_delete)

    def mutate(self, offspring, alpha):
        for individual in offspring:
            individual.mutate(alpha)
        return offspring

    def recombinate(self, parent_one, parent_two):
        offspring = np.intersect1d(parent_one.items, parent_two.items)
        uncommon = np.setxor1d(parent_one.items, parent_two.items)
        for item in uncommon:
            if np.random.rand() < 0.5:
                offspring = np.append(offspring, np.array(item))
        offspring = np.random.permutation(offspring)
        left = set(range(self.n_payload)) - set(offspring)
        left_order = np.random.permutation(list(left))
        offspring = np.concatenate([offspring, left_order], axis=0)
        offspring_ind = Individual(offspring)
        return offspring_ind

    def k_tournament(self, k):
        selected = random.sample(range(len(list(self.population))), k)
        individuals = self.population[selected]
        values = [individual.get_value(self) for individual in individuals]
        return selected[np.argmax(values)]

    def calculate_heuristic(self):
        values = list(self.values)
        weights = list(self.weights)
        devided = [values[i]/weights[i] for i in range(self.n_payload)]
        payload = [[weights[i], values[i], devided[i]] for i in range(self.n_payload)]
        payload_ordered = sorted(range(self.n_payload), key=lambda k: payload[k][1]/payload[k][0], reverse=True)
        print(payload_ordered)
        heuristic_ind = Individual(payload_ordered)
        return heuristic_ind.get_value(self)




