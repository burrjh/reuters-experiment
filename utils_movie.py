import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class MovieLens(Dataset):
    " Extends PyTorch's Dataset class so that its Dataloader can operate ontop of it"

    def __init__(self,train):
        if train:
            self.ratings = pd.read_csv("ratings_train.csv", index_col=0).to_numpy()
            self.users   = pd.read_csv("user_dummied.csv", index_col=0).to_numpy()

        else:
            self.ratings = pd.read_csv("ratings_test.csv", index_col=0).to_numpy()
            self.users   = pd.read_csv("user_dummied.csv", index_col=0).to_numpy()

    def __getitem__(self,index):
        user_id = self.ratings[index][0] #the user_id is the first element of the vector
        return self.ratings[index][1:], self.users[user_id-1][1:] #users is a df ordered by user_id but starting to count at 0
        #not returning user_id in both cases, therfore [1:] 

    def __len__(self):
        return len(self.ratings)

def valids_per_task(output,labels):
        #init
        n_one=2
        n_two=7
        n_three=21
        length= n_one+n_two+n_three


        out_1_max = torch.argmax(output[:,0:n_one],dim=1)
        true_1_max = torch.argmax(labels[:,0:n_one],dim=1)
        
        valid_1 = torch.sum(out_1_max == true_1_max)
        
        ################       
        out_2_max = torch.argmax(output[:,n_one:(n_one+n_two)],dim=1)
        true_2_max = torch.argmax(labels[:,n_one:(n_one+n_two)],dim=1)
        
        valid_2 = torch.sum(out_2_max == true_2_max)
        
        ##################
        out_3_max = torch.argmax(output[:,(n_one+n_two):length],dim=1)
        true_3_max = torch.argmax(labels[:,(n_one+n_two):length],dim=1)
        
        valid_3 = torch.sum(out_3_max == true_3_max)
        
        
        return valid_1, valid_2, valid_3



def train(net,dataloader,loss_fct,optimizer,device):
    """ Training Function.
    Trains Neural Net on Sample given by dataloader.
    Loss, Optimizer, Scheduler need to be specified
    (device needs to be passed, because otherwise the variable would 
    not be in the scope of the utils-function)
    Returns
    --------
    Optimized Net, Training Accuracy, training loss
    """
    valid_1 = 0
    valid_2 = 0
    valid_3 = 0
    total = 0
    batches_total = 0
    loss_sum = 0
    
    for i,data in enumerate(dataloader):
        # enumerate over complete Training Data
        inputs,labels = data           
        inputs = inputs.to(device)                  #put data on the gpu
        labels = labels.to(device)

        optimizer.zero_grad()
        
        output = net(inputs.float())              #hot'ish fix here: tensors need to be converted to float
        loss   = loss_fct(output, labels.float()) 
        loss.backward()
        optimizer.step()

        loss_sum    += loss
        batches_total += 1
        total += len(output) 
        #calculate accuracy per task
        valids_1,valids_2,valids_3 = valids_per_task(output,labels)
        valid_1 += valids_1
        valid_2 += valids_2
        valid_3 += valids_3 #augmented assignement is not possible here

    train_acc_one    = valid_1/(total)   
    train_acc_two    = valid_2/(total)
    train_acc_three  = valid_3/(total)
    train_acc_mean = (float(valid_1+valid_2+valid_3))/(3*total)

    loss_total     = loss_sum / batches_total
    
    return net, train_acc_mean, loss_total, [train_acc_one, train_acc_two,train_acc_three]

def test(net, dataloader, loss_fct, device):
    """ Test Function
    Requires (trained) Net and TestLoader
    Returns
    ------
    Test Accuracy, AUC-score, Validation-Loss
    """

    valid_1 = 0
    valid_2 = 0
    valid_3 = 0
    correct=0
    total=0
    loss_sum = 0
    batches_total = 0

    with torch.no_grad():
        for i,data in enumerate(dataloader):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = net.forward(inputs.float())

            loss     = loss_fct(output, labels.float()) 
            loss_sum += loss
            batches_total += 1

            total   += len(output)

            #calculate accuracy per task
            valids_1,valids_2,valids_3 = valids_per_task(output,labels)
            valid_1 += valids_1
            valid_2 += valids_2
            valid_3 += valids_3


    test_acc_one    = float(valid_1)/(total)
    test_acc_two    = float(valid_2)/(total)
    test_acc_three  = float(valid_3)/(total)
    test_acc_mean = (float(valid_1+valid_2+valid_3))/(3*total)


    loss_total   = loss_sum / batches_total


    return test_acc_mean, loss_total, [test_acc_one,test_acc_two,test_acc_three]



