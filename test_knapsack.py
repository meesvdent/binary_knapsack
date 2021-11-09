"""
Test knapsack class
"""

from knapsack import Knapsack


def test_knapsack():
    test_object = Knapsack(n_objects=100)
    test_object.init_pop()
    for object in test_object.population:
        print(object.order)
        print(object.objective_function(test_object))
    print(test_object.mutate(list(range(10)), 0.9))

def test_evolution():
    pass






if __name__ == "__main__":
    test_object = Knapsack()
    test_object.init_pop()
    test_object.evolution()
