"""
Test knapsack class
"""

from knapsack import Knapsack


def test_knapsack():
    test_object = Knapsack(n_objects=100)
    test_object.init_pop()
    print(test_object.get_pop())








if __name__ == "__main__":
    test_knapsack()
