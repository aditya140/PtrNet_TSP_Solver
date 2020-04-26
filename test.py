from tsp_solver import utils
from tsp_solver import tsp_optimal
from TSPDataset import TSPDataset
from PointerNet import PointerNet
from torch.utils.data import Dataset,DataLoader
import pandas as pd




data=TSPDataset(4,10,file="./data/test.txt")
test_dataloader=DataLoader(data,batch_size=16,num_workers=10)
solver=utils.BaselineSolver()
remove_alg=["Genetic","Optimal"]
Ptr=PointerNet(100,50,1,False)
solver.add_model(Ptr)
data=[]
for i in test_dataloader:
    for x,y in zip(i["Points"],i["Solution"]):
        res=(solver.solve_all(x.tolist(),remove_alg,returnTours=False))
        optimal=solver.tour_length(y.tolist())
        for k in res.keys():
            res[k]=res[k]/optimal
        data.append(res)
df=pd.DataFrame(data)
print(df.mean())


