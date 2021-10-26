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









