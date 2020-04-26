from tsp_solver import utils
from tsp_solver import tsp_optimal
from TSPDataset import *
from PointerNet import *
# from PtrNet import *

data=TSPDataset(100,10,onlyInt=False)
train_dataloader=DataLoader(data,batch_size=16,num_workers=10)
solver=utils.BaselineSolver()

Ptr=PointerNet(100,50,1,False)
for i in train_dataloader:
    o,p=Ptr(i["Points"])
    print(solver.solve_all(i["Points"][0].tolist()))

