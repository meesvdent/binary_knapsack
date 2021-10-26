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

    def __init__(self, n_objects):
        self.order = np.random.permutation(n_objects)

    def __repr__(self):
        return str(self.get_order())

    def get_order(self):
        return self.order

    def objective_function(self, knapsack):
        remaining_capacity = knapsack.get_constraint()
        total_value = 0
        for payload in self.get_order():
            if knapsack.get_weight(payload) <= knapsack.get_constraint():
                remaining_capacity -= knapsack.get_weight(payload)
                total_value += knapsack.get_value(payload)

        return total_value








