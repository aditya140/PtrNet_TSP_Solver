from hparams import model_params,test_params
from TSPDataset import TSPDataset
from torch.utils.data import Dataset,DataLoader
from PointerNet import PointerNet
from tqdm import tqdm
from tsp_solver.utils import BaselineSolver
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import time
import pandas as pd




def main():
    test_dataset=TSPDataset(test_params.test_size,test_params.nof_points,file=test_params.file)
    test_dataloader=DataLoader(test_dataset,batch_size=test_params.batch_size,num_workers=10)
    model = PointerNet(model_params.embedding_size,
                    model_params.hiddens,
                    model_params.nof_lstms,
                    model_params.bidir,
                    model_params.dropout)

    if model_params.gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    model.to(device)
    model.load_state_dict(torch.load(test_params.model,map_location=device))
    solver = BaselineSolver()
    remove_alg = ["Genetic", "Optimal"]
    solver.add_model(model,device)
    model.eval()
    data = []
    cnt=0
    for i in tqdm(test_dataloader):
        for x, y in zip(i["Points"], i["Solution"]):
            res = solver.solve_all(x.numpy(), remove_alg, returnTours=False)
            if "Optimal" in remove_alg:
                res["Optimal"] = solver.tour_length(y.tolist())
            res["Optimal"]=min(res["Google OR"],res["Optimal"])
            for k in res.keys():
                if k!="Optimal":
                    res[k] = res[k] / res["Optimal"]
            data.append(res)
            cnt+=1
        if cnt>=test_params.test_size:
            break
    df = pd.DataFrame(data)
    result_file=test_params.model.split("/")[-1].split(".")[0]+test_params.file.split("/")[-1].split(".")[0]+".csv"
    df.to_csv(result_file)
    print(df.mean())
if __name__ == '__main__':
    main()