import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np



from utils_movie import train, test, valids_per_task, MovieLens
from networks_movie import LNet

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

if __name__ == "__main__":
         

#load data, transform to DataSet Class & create DataLoader
    print("Loading Data.......")


    data_train = MovieLens(train=True)
    data_test  = MovieLens(train=False)

    
    batchsize = 50

    train_loader = DataLoader(dataset=data_train,
                         batch_size=batchsize,
                         shuffle=True,
                         num_workers=2)


    test_loader = DataLoader(dataset=data_test,
                         batch_size=batchsize,
                         shuffle=False,
                         num_workers=2)

    print("Loaded.")


# Using third GPU
    torch.set_num_threads(42)
    use_cuda = torch.cuda.is_available()
    print("We'll use cuda:", use_cuda)
    #device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")     # use cuda:3 on specific GPU
    device = torch.device("cpu")
    


#define experimental variables
    in_size = 16912
    n_classes = 30
    net_dim=800
    reps=20
    epochs = 30
    loss_fct = torch.nn.L1Loss(reduction='sum')


    print("Start!")

    results=[]

    for rep in range(reps):

        net= LNet(in_size,net_dim,n_classes)    
        net.init_weights()  
        net.to(device)
        learning_rate= 0.01
        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x : 0.999**x)

        for epoch in range(epochs):

            net, train_acc_mean, train_loss, task_train_accs = train(net,train_loader,loss_fct,optimizer,device)
            test_acc_mean, val_loss, task_test_accs  = test(net, test_loader,loss_fct, device)
            acc_1, acc_2, acc_3 = task_test_accs
            epoch_results = [rep+1,epoch+1, train_acc_mean,test_acc_mean, train_loss.item(), val_loss.item(),acc_1,acc_2,acc_3]  # .item() necessary to get pure number instead of tensor object
            scheduler.step()  

            print("__________________________")
            print("After Epoch ", epoch+1)
            print("Training Loss:", train_loss.item())
            print("Evaluation Loss:", val_loss.item())
            print("Train Accuracy:", train_acc_mean)
            print("Test Accuracy:", test_acc_mean)
            print("Test Acc Task1:", acc_1)
            print("Test Acc Task2:", acc_2)
            print("Test Acc Task3:", acc_3)

            results.append(epoch_results)


        print("---------------------------")
        print("Finished Repetition", rep+1, "starting next one.")
        print("---------------------------")




    df_lnet = pd.DataFrame(results)

    df_lnet.to_csv('lnet_movielens_lowerlr.csv')