import numpy
import ortools # Using ortools version 7
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

class TravelingSalesPerson:
    def __init__(self, data, metric='euclidean', approximation_multiplier=1000, timeout=2.0):
        self.data = data
        self.metric = metric
        self.approximation_multiplier = approximation_multiplier
        self.timeout = timeout

    # def get_ordered_data(self):
    #     # Get distances along rows 
    #     dist1 = squareform(pdist(self.data, metric=self.metric))
    #     # Get distances along columns
    #     dist2 = squareform(pdist(self.data.T, metric=self.metric))

    #     row_order = self.seriate(dist1)
    #     column_order = self.seriate(dist2)

    #     ordered_data = pd.DataFrame(self.data)
    #     ordered_data = ordered_data.iloc[row_order, column_order]
    #     return ordered_data

    def get_ordered_data(self):
        # Get distances along rows 
        dist1 = squareform(pdist(self.data, metric=self.metric))
        row_order = self.seriate(dist1)
        data = pd.DataFrame(self.data)
        data = data.iloc[row_order,:]

        # Get distances along columns
        dist2 = squareform(pdist(data.values.T, metric=self.metric))
        column_order = self.seriate(dist2)
        ordered_data = data.iloc[:, column_order]
        return ordered_data

    def _validate_data(self, dists):
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

    def seriate(self, dists):
        # Validate distances
        self._validate_data(dists)
        if self.timeout > 0:
            return self._seriate(dists=dists)
        elif self.timeout < 0:
            raise ValueError("timeout cannot be negative.")
        self.timeout = 1.
        route = None
        while route is None:
            try:
                route = self._seriate(dists=dists)
            except IncompleteSolutionError:
                self.timeout *= 2
        return route


    def _seriate(self, dists):
        assert dists[dists < 0].size == 0, "distances must be non-negative"
        assert self.timeout > 0
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
            return int(dist * self.approximation_multiplier)

        routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_callback))
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.FromMilliseconds(int(self.timeout * 1000))
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

class IncompleteSolutionError(Exception):
    """Indicate that a solution for the TSP problem was not found."""
    pass


class InvalidDistanceValues(ValueError):
    """Indicate that the distance array contains invalid values."""
    pass

if __name__ == "__main__":
    # Create simulated data as in the paper
    X = numpy.zeros((100, 100))
    for n in [0,10,20,30,40,50,60]:
        X[int(10.*n/7):int(10.*(n+10)/7):,n:n+40] = 1

    X = squareform(pdist(X, metric="euclidean"))
    # X = squareform(pdist(X, metric="hamming"))

    seaborn.heatmap(X)
    plt.figure()

    numpy.random.shuffle(X)
    X = X.T
    numpy.random.shuffle(X)
    X = X.T

    seaborn.heatmap(X)
    tsp = TravelingSalesPerson(X)
    plt.figure()

    # Visualize the output data
    Y = tsp.get_ordered_data()
    seaborn.heatmap(Y)
    plt.show()
