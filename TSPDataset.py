# @title TSP Dataset

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tsp_solver import tsp_optimal
from parser import FileParser


class TSPDataset(Dataset):
    """[summary]

    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(
        self,
        data_size,
        seq_len,
        solver=tsp_optimal.tsp_opt,
        solve=True,
        onlyInt=False,
        file=None,
    ):
        """[summary]

        Arguments:
            data_size {[type]} -- Size of dataset (TSP problems)
            seq_len {[type]} -- Length of one TSP problem

        Keyword Arguments:
            solver {[type]} --  (default: {tsp_optimal.tsp_opt})
            solve {bool} --  (default: {True})
            onlyInt {bool} --  (default: {False})s
            file {[type]} -- File to be parsed (default: {None})
        """
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.onlyInt = onlyInt
        if file:
            self.data = self._read_file(file)
        else:
            self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data["Points_List"][idx]).float()
        solution = (
            torch.from_numpy(self.data["Solutions"][idx]).long() if self.solve else None
        )
        sample = {"Points": tensor, "Solution": solution}
        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit="data")
        for i, _ in enumerate(data_iter):
            if self.onlyInt:
                points_list.append(np.random.randint(1, 101, size=(self.seq_len, 2)))
            else:
                points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit="solve")
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions.append(self.solver(points))
        else:
            solutions = None
        return {"Points_List": points_list, "Solutions": solutions}

    def _read_file(self, file):
        parser = FileParser(self.seq_len, file)
        points_list = np.array(parser.coords)
        solutions = np.array(parser.tours)
        return {"Points_List": points_list, "Solutions": solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec

    def pairwise_distances(self, x, y=None):
        """
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

