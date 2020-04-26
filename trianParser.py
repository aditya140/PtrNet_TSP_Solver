#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:20:01 2020

@author: sameeradhikari
"""
from sklearn.utils import shuffle
from utils import *

class TrainParser(object):
    def __init__(self, numOfNodes, filePath):
        self.numOfNodes = numOfNodes
        self.filePath = filePath
        self.filedata = shuffle(open(filePath, "r").readlines())
        self.process_files(self.filedata)
        solver=utils.BaselineSolver()
 
        

    def process_files(self, lines):
        for line_num, line in enumerate(lines):
            line = line.split(" ")
            nodes_coord = []
            for idx in range(0, 2 * self.numOfNodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            tour = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            print(solver.solve_all(nodes_coord,returnTours=True))
            #print(nodes_coord)
            print(solver.tour_length(tour))
            print(tour)
        

if __name__ == "__main__":
    parser = TrainParser(10, "/Users/sameeradhikari/Desktop/VRP/git/PtrNet_TSP_Solver/test.txt")
