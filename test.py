from tsp_solver import utils
from tsp_solver import tsp_optimal
from TSPDataset import TSPDataset
from PointerNet import PointerNet
from torch.utils.data import Dataset,DataLoader

data=TSPDataset(10,10,onlyInt=False)
train_dataloader=DataLoader(data,batch_size=16,num_workers=10)
solver=utils.BaselineSolver()
remove_alg=["Genetic","Optimal"]
Ptr=PointerNet(100,50,1,False)
solver.add_model(Ptr)
for i in train_dataloader:
    for x in i["Points"]:
        print(solver.solve_all(x.tolist(),remove_alg,returnTours=False))
    

