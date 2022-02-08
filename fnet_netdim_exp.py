import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np



from utils import train, test, load_data, ReutersTrain, ReutersTest, feature_selection
from networks import FLayer,FNet
    



if __name__ == "__main__":


#load data, transform to DataSet Class & create DataLoader
    print("Loading Data.......")
    data = load_data()

    data = feature_selection(data)  #trains a Decision Tree Ensemble and deletes features estimated to have zero importance


    reuters_train = ReutersTrain(data)
    reuters_test  = ReutersTest(data)

    
    batchsize = 50

    train_loader = DataLoader(dataset=reuters_train,
                         batch_size=batchsize,
                         shuffle=True,
                         num_workers=2)


    test_loader = DataLoader(dataset=reuters_test,
                         batch_size=batchsize,
                         shuffle=False,
                         num_workers=2)

    print("Loaded.")


# Using third GPU
    use_cuda = torch.cuda.is_available()
    print("We'll use cuda:", use_cuda)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")     # use cuda:3 on specific GPU



#define experimental variables
    in_size = data['x_train'].shape[1]
    n_classes = 90
    net_range = range(50,850,50)
    fact_rank = 100
    repetitions=20
    epochs = 40
    loss_fct = torch.nn.BCELoss()


    print("Start!")

    results=[]

    for rep in range(repetitions):

        for net_dim in net_range:

            net= FNet(in_size,fact_rank,net_dim,n_classes)    
            net.init_weights()  
            net.to(device)
            learning_rate= 0.001
            optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x : 0.999**x)

            for epoch in range(epochs):

                net,train_acc,train_loss = train(net,train_loader,loss_fct,optimizer, device)
                test_acc, auc, val_loss  = test(net, test_loader, loss_fct, device)
                epoch_results = [rep+1,net_dim,epoch+1,train_acc.item(),test_acc.item(), train_loss.item(), val_loss.item(),auc]  # .item() necessary to get pure number instead of tensor object

                scheduler.step()  

                print("__________________________")
                print("After Epoch ", epoch+1, "with fact_rank", fact_rank)
                print("Training Loss:", train_loss.item())
                print("Evaluation Loss:", val_loss.item())
                print("Train Accuracy:", train_acc.item())
                print("Test Accuracy:", test_acc.item())
                print("Area under Curve:", auc)

                results.append(epoch_results)

            ##initialize new net for next repetition
            print("---------------------------")
            print("Finished Repetition", rep+1, "starting next one.")
            print("---------------------------")




    df_lnet = pd.DataFrame(results)

    df_lnet.to_csv('fnet_netdim_res.csv')