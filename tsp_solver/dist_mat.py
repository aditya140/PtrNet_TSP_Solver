import numpy as np

def euc_dist(a,b):
    return np.sqrt(np.square(a[0]-b[0])+np.square(a[1]-b[1]))

def distance_matrix(coord):
    distance_matrix=np.zeros((len(coord),len(coord)))
    for i in range(len(coord)):
        for j in range(len(coord)):
            distance_matrix[i,j]=euc_dist(coord[i],coord[j])
    return distance_matrix