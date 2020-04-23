import numpy as np
from scipy.spatial import distance_matrix,distance
from .googleORsolver import *
from .tsp_optimal import *


def coordToDistMat(coord):
    return distance_matrix(coord,coord)

class BaselineSolver(object):
    def __init__(self,coord):
        self.coord=coord
        self.coordDict={i : self.coord[i] for i in range(0, len(self.coord))}
        self.dist_mat=coordToDistMat(self.coord)
        self.solvers={"Nearest Neighbour":self.solve_nearest,"Farthest Neighbour":self.solve_farthest,"Optimal":self.optimal,"Google OR":self.GoogleORSolver}

    def nearest(self,arr):
        mask = np.isin(np.arange(0,len(self.coord)),self.untraversed)
        subset_idx = np.argmin(arr[mask])
        parent_idx = np.arange(arr.shape[0])[mask][subset_idx]
        return parent_idx

    def farthest(self,arr):
        mask = np.isin(np.arange(0,len(self.coord)),self.untraversed)
        subset_idx = np.argmax(arr[mask])
        parent_idx = np.arange(arr.shape[0])[mask][subset_idx]
        return parent_idx

    def solve_nearest(self):
        return self.solve(criterion="nearest")

    def solve_farthest(self):
        return self.solve(criterion="farthest")

        
    def solve(self,criterion="nearest"):
        start=0
        self.untraversed=list(self.coordDict.keys())
        if criterion=="nearest":
            self.criterion=self.nearest
        elif criterion=="farthest":
            self.criterion=self.farthest
        self.tour=[]
        cur=start
        while len(self.untraversed)!=0:
            cur=self._solve(cur)
        return np.array(self.tour)

    def _solve(self,cur):
        self.tour.append(cur)
        self.untraversed.remove(cur)
        if len(self.untraversed)==0:
            return False
        cur=self.criterion(self.dist_mat[cur])
        return cur

    def tourLength(self,tour):
        assert len(tour)>2
        tour_len=0
        for i in range(len(tour)-1):
            tour_len+=self.dist_mat[tour[i],tour[i+1]]
        tour_len+=self.dist_mat[tour[0],tour[-1]]
        return tour_len

    def GoogleORSolver(self):
        return np.array(GoogleORsolver(self.dist_mat))

    def optimal(self):
        return tsp_opt(self.coord)

    def solve_all(self,returnTours=False):
        metrics={}
        for name,solver in self.solvers.items():
            tour=solver()
            if returnTours:
                metrics[name]=(self.tourLength(tour),tour)
            else:
                metrics[name]=self.tourLength(tour)
        return metrics


    
def create_random_points(num_points,val_range=(0,100)):
    return np.random.randint(val_range[0],val_range[0]+val_range[1],size=(num_points, 2))



if __name__=="__main__":
    coord=create_random_points(15)
    solver=BaselineSolver(coord)
    print(solver.solve_all())



        

