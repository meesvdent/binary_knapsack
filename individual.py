"""
Individual class

Individuals, which have properties:
* order: a permutation of length nvalues

And method:
* get_value
* get_weight
"""

import numpy as np

class Individual:

    def __init__(self, order):
        self.order = order
        self.value = 0
        self.items = []

    def __repr__(self):
        return str(self.get_order())

    def get_order(self):
        return self.order

    def in_knapsack(self, knapsack):
        remaining_capacity = knapsack.get_constraint()
        for payload in self.get_order():
            if knapsack.get_weight(payload) <= remaining_capacity:
                remaining_capacity -= knapsack.get_weight(payload)
                self.items.append(payload)

        return self.items

    def objective_function(self, knapsack):
        items = self.in_knapsack(knapsack)
        self.value = 0
        for payload in items:
            self.value += knapsack.get_value(payload)

    def get_value(self, knapsack):
        if self.value == 0:
            self.objective_function(knapsack)
        return self.value

    def mutate(self, alpha):
        if np.random.rand() < alpha:
            ind = np.random.randint(len(self.order), size=2)
            first = int(self.order[ind[0]])
            second = int(self.order[ind[1]])
            self.order[ind[0]] = second
            self.order[ind[1]] = first
        return self








