import itertools
import numpy as np
def tsp_opt(points):
    """
    Dynamic programing solution for TSP - O(2^n*n^2)
    https://gist.github.com/mlalevic/6222750
    :param points: List of (x, y) points
    :return: Optimal solution
    """
    if len(points)>19:
        return []

    def length(x_coord, y_coord):
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])
