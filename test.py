from tsp_solver import utils
from tsp_solver import tsp_optimal
from TSPDataset import *
from pointerNet import *
# from PtrNet import *

data=TSPDataset(10,5,onlyInt=True)
train_dataloader=DataLoader(data,batch_size=16,num_workers=10)


Ptr=PointerNet(100,50,1,False)
for i in train_dataloader:
    o,p=Ptr(i["Points"])
    print(p)
    print(i["Solution"])
    break

