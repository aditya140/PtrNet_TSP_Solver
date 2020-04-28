from PointerNet import PointerNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ast
def read_model_txt(path):
    params={}
    with open(path,"r") as f:
        for line in f.readlines()[1:9]:
            key,value=line.split(":")
            params[key]=(value.replace("\n","").replace(" ",""))
    return params

def load_model(name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path=f"./model/{name}.txt"
    params =read_model_txt(model_path)
    model=PointerNet(int(params["embedding_size"]),int(params["hiddens"]),int(params["nof_lstms"]),False,float(params["dropout"]))
    model.load_state_dict(torch.load("./model/"+params["name"],map_location=device))
    return model

