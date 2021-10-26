"""
Test knapsack class
"""

from knapsack import Knapsack


def test_knapsack():
    test_object = Knapsack(n_objects=10)
    test_object.init_pop()
    print(test_object.get_pop())
    print("obj func")
    print(test_object.objective_func(74))








if __name__ == "__main__":
    test_knapsack()
