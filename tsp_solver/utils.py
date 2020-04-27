import numpy as np
# from scipy.spatial import distance_matrix, distance
from .googleORsolver import *
from .tsp_optimal import *
from itertools import combinations
import mlrose
import torch
import matplotlib.pyplot as plt
from .dist_mat import distance_matrix


def coord_to_dist_mat(coord):
    # return distance_matrix(coord, coord)
    return distance_matrix(coord)



def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = []
    y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, "co")

    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x)) / float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []
            yi = []
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(
                xi[-1],
                yi[-1],
                (xi[0] - xi[-1]),
                (yi[0] - yi[-1]),
                head_width=a_scale,
                color="r",
                length_includes_head=True,
                ls="dashed",
                width=0.001 / float(num_iters),
            )
            for i in range(0, len(x) - 1):
                plt.arrow(
                    xi[i],
                    yi[i],
                    (xi[i + 1] - xi[i]),
                    (yi[i + 1] - yi[i]),
                    head_width=a_scale,
                    color="r",
                    length_includes_head=True,
                    ls="dashed",
                    width=0.001 / float(num_iters),
                )

    # Draw the primary path for the TSP problem
    plt.arrow(
        x[-1],
        y[-1],
        (x[0] - x[-1]),
        (y[0] - y[-1]),
        head_width=a_scale,
        color="g",
        length_includes_head=True,
    )
    for i in range(0, len(x) - 1):
        plt.arrow(
            x[i],
            y[i],
            (x[i + 1] - x[i]),
            (y[i + 1] - y[i]),
            head_width=a_scale,
            color="g",
            length_includes_head=True,
        )

    # Set axis too slitghtly larger than the set of x and y
    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.show()



class BaselineSolver(object):
    """[summary]
    Baseline Solver To solve TSP using conventional Algorithmsself.
    Call `.create` to initialize a new TSP problem
    """

    def __init__(self):
        self.model = {}
        self.solvers = {
            "Nearest Neighbour": self.solve_nearest,
            "Farthest Neighbour": self.solve_farthest,
            "Optimal": self.optimal,
            "Google OR": self.google_or_solver,
            "random": self.solve_random,
            "Nearest insertion": self.nearest_insertion,
            "Genetic": self.genetic,
        }

    def create(self, coord):
        """[summary]
        Creates and loads a new TSP problemset

        Arguments:
            coord {[type]} -- List of Coordinates in the TSP
        """
        self.coord = coord
        self.coordDict = {i: self.coord[i] for i in range(0, len(self.coord))}
        self.dist_mat = coord_to_dist_mat(self.coord)

    def nearest_neighbour(self, arr):
        mask = np.isin(np.arange(0, len(self.coord)), self.untraversed)
        subset_idx = np.argmin(arr[mask])
        parent_idx = np.arange(arr.shape[0])[mask][subset_idx]
        return parent_idx

    def farthest_neighbour(self, arr):
        mask = np.isin(np.arange(0, len(self.coord)), self.untraversed)
        subset_idx = np.argmax(arr[mask])
        parent_idx = np.arange(arr.shape[0])[mask][subset_idx]
        return parent_idx

    def random(self, arr):
        mask = np.isin(np.arange(0, len(self.coord)), self.untraversed)
        return np.random.choice(np.arange(0, len(self.coord))[mask])

    def solve_nearest(self):
        return self.solve(criterion="nearest")

    def solve_farthest(self):
        return self.solve(criterion="farthest")

    def solve_random(self):
        return self.solve(criterion="random")

    def add_model(self, model,device):
        self.model = model
        self.solvers["Model"] = self.solve_model
        self.model_device=device

    def solve_model(self):
        with torch.no_grad():
            o, p = self.model(torch.tensor([self.coord]).float().to(self.model_device))
        return p.tolist()[0]

    def __create_mask(self, tour):
        ur = np.triu_indices(self.dist_mat.shape[0])
        mask = np.zeros(self.dist_mat.shape).astype(bool)
        for i in tour:
            mask[:, i] = True
            mask[i, :] = True
        for i, j in combinations(tour, 2):
            i, j = sorted([i, j], reverse=True)
            mask[i, j] = False
        mask[ur] = False
        return mask

    def nearest_insertion(self):
        # Select smallest subtour
        self.untraversed = list(self.coordDict.keys())
        mask = np.ones(self.dist_mat.shape)
        ur = np.triu_indices(self.dist_mat.shape[0])
        mask[ur] = 0
        mask = mask.astype(np.bool)
        masked_dist_mat = np.ma.masked_array(self.dist_mat, ~mask)
        pos = np.where(masked_dist_mat == np.amin(self.dist_mat[mask]))
        tour = []
        tour.append(pos[0][0])
        tour.append(pos[1][0])

        # select nearest insertion
        def find_nearest(mask):
            masked_dist_mat = np.ma.masked_array(self.dist_mat, ~mask)
            pos = np.where(masked_dist_mat == np.amin(self.dist_mat[mask]))
            return pos[0][0], pos[1][0]

        def insert_nearest(tour, pos):
            if pos[0] in tour:
                or_pt, new_pt = pos
            else:
                new_pt, or_pt = pos
            d1 = self.dist_mat[new_pt, or_pt]
            pr_pt = (
                tour[len(tour) - 1]
                if tour.index(or_pt) - 1 < 0
                else tour[tour.index(or_pt) - 1]
            )
            d2 = self.dist_mat[new_pt, pr_pt]
            nx_pt = (
                tour[0]
                if tour.index(or_pt) + 1 >= len(tour)
                else tour[tour.index(or_pt) + 1]
            )
            d3 = self.dist_mat[new_pt, nx_pt]
            if d2 > d3:
                tour = tour[: tour.index(or_pt)] + [new_pt] + tour[tour.index(or_pt) :]
            else:
                tour = (
                    tour[: tour.index(or_pt) - 1]
                    + [new_pt]
                    + tour[tour.index(or_pt) - 1 :]
                )
            return tour

        while len(tour) < len(self.coord):
            mask = self.__create_mask(tour)
            pos = find_nearest(mask)
            tour = insert_nearest(tour, pos)
        return tour

    def solve(self, criterion="nearest"):
        start = 0
        self.untraversed = list(self.coordDict.keys())
        if criterion == "nearest":
            self.criterion = self.nearest_neighbour
        elif criterion == "farthest":
            self.criterion = self.farthest_neighbour
        elif criterion == "random":
            self.criterion = self.random
        self.tour = []
        cur = start
        while len(self.untraversed) != 0:
            cur = self._solve(cur)
        return np.array(self.tour)

    def _solve(self, cur):
        self.tour.append(cur)
        self.untraversed.remove(cur)
        if len(self.untraversed) == 0:
            return False
        cur = self.criterion(self.dist_mat[cur])
        return cur

    def tour_length(self, tour):
        """[summary]
        Returns the tour length of a given tour for the loaded problemset
        Arguments:
            tour {List[int]} -- Tour whose length is to be calculated

        Returns:
            Int -- Tour Length
        """
        assert len(tour) > 2
        tour_len = 0
        for i in range(len(tour) - 1):
            tour_len += self.dist_mat[tour[i], tour[i + 1]]
        tour_len += self.dist_mat[tour[0], tour[-1]]
        return tour_len

    def google_or_solver(self):
        return np.array(GoogleORsolver(self.dist_mat))

    def genetic(self):
        problem_no_fit = mlrose.TSPOpt(
            length=len(self.coord), coords=self.coord, maximize=False
        )
        best_state, best_fitness = mlrose.genetic_alg(
            problem_no_fit, mutation_prob=0.2, max_attempts=100, random_state=2
        )

        return best_state

    def optimal(self):
        return tsp_opt(self.coord)

    def solve_all(
        self, coord, except_alg=[], returnTours=False,
    ):
        """[summary]

        Arguments:
            coord {List[float]} -- List of coordinates

        Keyword Arguments:
            except_alg {list} -- Algorithms NOT to run (default: {[]})
            returnTours {bool} -- Return Tours along with length (default: {False})

        Returns:
            [dict] -- dictionary with tour length of all algorithms
        """

        self.create(coord)

        metrics = {}
        for name, solver in self.solvers.items():
            if name not in except_alg:
                tour = solver()
                if returnTours:
                    metrics[name] = (self.tour_length(tour), tour)
                else:
                    metrics[name] = self.tour_length(tour)
        return metrics


def create_random_points(num_points, val_range=(0, 100)):
    return np.random.randint(
        val_range[0], val_range[0] + val_range[1], size=(num_points, 2)
    )
