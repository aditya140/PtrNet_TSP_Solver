from hparams import params
from TSPDataset import TSPDataset
from torch.utils.data import Dataset,DataLoader
from PointerNet import PointerNet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model,iter,loss_func,optimizer,device):
    model.train()
    losses=[]
    for idx,batch in enumerate(iter):
        train_batch = Variable(batch['Points']).to(device)
        target_batch = Variable(batch['Solution']).to(device)
        o, p = model(train_batch)
        o = o.contiguous().view(-1, o.size()[-1])
        target_batch = target_batch.view(-1)
        loss = CCE(o, target_batch)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(losses)

def eval(model,iter,loss_func,device):
    model.eval()
    val_losses=[]
    for idx,batch in enumerate(iter):
        train_batch = Variable(batch['Points']).to(device)
        target_batch = Variable(batch['Solution']).to(device)
        with torch.no_grad():
            o, p = model(train_batch)
        o = o.contiguous().view(-1, o.size()[-1])
        target_batch = target_batch.view(-1)
        loss = CCE(o, target_batch)
        val_losses.append(loss.item())
    return np.mean(val_losses)


if __name__=="__main__":
    ## Create Dataset for training
    train_dataset=TSPDataset(params.train_size,params.nof_points)
    val_dataset=TSPDataset(params.val_size,params.nof_points)
    test_dataset=TSPDataset(params.test_size,params.nof_points)

    train_dataloader=DataLoader(train_dataset,batch_size=params.batch_size,num_workers=10)
    val_dataloader=DataLoader(val_dataset,batch_size=params.batch_size,num_workers=10)
    test_dataloader=DataLoader(test_dataset,batch_size=params.batch_size,num_workers=10)

    model_path="./"

    model = PointerNet(params.embedding_size,
                    params.hiddens,
                    params.nof_lstms,
                    params.bidir)
    if params.gpu:
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    model.to(device)
    CCE = torch.nn.CrossEntropyLoss()
    optimizer=optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=params.lr)
    best_valid_loss=float('inf')

    for epoch in range(params.nof_epoch):
        train_iterator = train_dataloader
        val_iterator = val_dataloader
        st_time=time.time()
        train_loss=train(model,train_iterator,CCE,optimizer,device)
        valid_loss=eval(model,val_iterator,CCE,device)
        e_time=time.time()
        epoch_mins, epoch_secs = epoch_time(st_time, e_time)
        if valid_loss<best_valid_loss:
            best_valid_loss=valid_loss
            torch.save(model.state_dict(),model_path+"model.pt")
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

