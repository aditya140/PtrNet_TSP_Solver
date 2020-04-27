import numpy as np

class FileParser(object):
    def __init__(self, numOfNodes, filePath):
        self.numOfNodes = numOfNodes
        self.filePath = filePath
        self.filedata = open(filePath, "r").readlines()
        self.process_files(self.filedata)
 
        

    def process_files(self, lines):
        self.coords=[]
        self.tours=[]
        for line_num, line in enumerate(lines):
            line = line.split(" ")
            nodes_coord = []
            for idx in range(0, 2 * self.numOfNodes, 2):
                nodes_coord.append([np.float64(line[idx]), np.float64(line[idx + 1])])
            tour = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
            self.coords.append(nodes_coord)
            self.tours.append(tour)
        