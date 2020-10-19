"""Seriation - NP-hard ordering of elements in a set given the distance matrix."""
from typing import List

import numpy
import ortools # Using ortools version 7
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

class IncompleteSolutionError(Exception):
    """Indicate that a solution for the TSP problem was not found."""
    pass


class InvalidDistanceValues(ValueError):
    """Indicate that the distance array contains invalid values."""
    pass


def _validate_data(dists: numpy.ndarray):
    """Check dists contains valid values."""
    try:
        isinf = numpy.isinf(dists).any()
        isnan = numpy.isnan(dists).any()
    except Exception as e:
        raise InvalidDistanceValues() from e
    if isinf:
        raise InvalidDistanceValues("Data contains inf values.")
    if isnan:
        raise InvalidDistanceValues("Data contains NaN values.")


def seriate(dists: numpy.ndarray, approximation_multiplier: int = 1000,
            timeout: float = 2.0) -> List[int]:

    # Validate dataset
    _validate_data(dists)
    if timeout > 0:
        return _seriate(dists=dists, approximation_multiplier=approximation_multiplier,
                        timeout=timeout)
    elif timeout < 0:
        raise ValueError("timeout cannot be negative.")
    timeout = 1.
    route = None
    while route is None:
        try:
            route = _seriate(dists=dists, approximation_multiplier=approximation_multiplier,
                             timeout=timeout)
        except IncompleteSolutionError:
            timeout *= 2
    return route


def _seriate(dists: numpy.ndarray, approximation_multiplier=1000, timeout=2.0) -> List[int]:
    assert dists[dists < 0].size == 0, "distances must be non-negative"
    assert timeout > 0
    squareform = len(dists.shape) == 2
    if squareform:
        assert dists.shape[0] == dists.shape[1]
        size = dists.shape[0]
    else:
        raise InvalidDistanceValues("Data is not squareform.")

    manager = pywrapcp.RoutingIndexManager(size + 1, 1, size)
    routing = pywrapcp.RoutingModel(manager)

    def dist_callback(x, y):
        x = manager.IndexToNode(x)
        y = manager.IndexToNode(y)
        if x == size or y == size or x == y:
            return 0
        if squareform:
            dist = dists[x, y]
        else:
            # convert to the condensed index
            if x < y:
                x, y = y, x
            dist = dists[size * y - y * (y + 1) // 2 + x - y - 1]
        # ortools wants integers, so we approximate here
        return int(dist * approximation_multiplier)

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_callback))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.FromMilliseconds(int(timeout * 1000))
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment is None:
        raise IncompleteSolutionError("No solution was found. Please increase the timeout value or set it to 0.")
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node < size:
            route.append(node)
        index = assignment.Value(routing.NextVar(index))
    return route

if __name__ == "__main__":
    # Create simulated data as in the paper
    X = numpy.zeros((100, 100))
    for n in [0,10,20,30,40,50,60]:
        X[int(10.*n/7):int(10.*(n+10)/7):,n:n+40] = 1

    seaborn.heatmap(X)
    plt.figure()

    numpy.random.shuffle(X)
    X = X.T
    numpy.random.shuffle(X)
    X = X.T

    Y = numpy.zeros((100, 100))

    dist1 = squareform(pdist(X, metric="euclidean"))
    dist2 = squareform(pdist(X.T, metric="euclidean"))

    seaborn.heatmap(X)

    tsp_data1 = seriate(dist2)
    tsp_data2 = seriate(dist1)
    plt.figure()
    print(tsp_data1)
    print(tsp_data2)

    X = pd.DataFrame(X)
    Y = X.iloc[tsp_data2, tsp_data1]

    seaborn.heatmap(Y)
    plt.show()
