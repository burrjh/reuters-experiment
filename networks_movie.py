import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FLayer(nn.Module):
#fact layer structure

	__constants__ = ["m","r","n"]
	
	def __init__(self,m,r,n):
		super(FLayer, self).__init__()
		self.n = n
		self.m = m
		self.r = r
		self.U = nn.Parameter(data=torch.zeros(n,r))
		self.V = nn.Parameter(data=torch.zeros(m,r))

		self.bias = nn.Parameter(data=torch.zeros(n,1))

	def forward(self,x):
		t1 = torch.matmul(self.V.T,x.T)		
		t2 = torch.matmul(self.U,t1).T
		t2.add_(self.bias.T)					##in-place addition of bias 
		
		return t2

class FNet(nn.Module):

    def __init__(self,input_n,fact_rank,net_dim,n_classes):

        super(FNet,self).__init__()

        self.input_n    = input_n
        self.fact_rank  = fact_rank
        self.net_dim   = net_dim
        self.n_classes  = n_classes

        #factorized layer
        self.fact = FLayer(self.input_n,self.fact_rank,self.net_dim)
        #fully connected layers:
        self.fc1 = nn.Linear(self.net_dim,self.net_dim,bias=True)
        self.fc2 = nn.Linear(self.net_dim,self.n_classes, bias=True)

    def forward(self,input):
        x = F.relu(self.fact(input))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x), dim=1)

        return x


    def init_weights(self):
        for p in self.parameters():
            sd = 1.0 / np.max(p.size())
            nn.init.normal_(p,0,sd)

    def maxdim(self):
        m = 0
        for p in self.parameters():
            m = np.max([m,np.max(p.size())])
        return m

    def effective_size(self):
        out=0
        for p in self.parameters():
            out += p.size().numel()
        return out



class LNet(nn.Module):
    """ fully connected neural net with one hidden Layer
    """
    def __init__(self,input_n,net_rank,n_classes):

        super(LNet,self).__init__()

        self.input_n   = input_n
        self.net_rank  = net_rank
        self.n_classes = n_classes

        self.fc1  = nn.Linear(self.input_n,self.net_rank)
        self.fc2  = nn.Linear(self.net_rank, self.net_rank)
        self.fc3  = nn.Linear(self.net_rank,self.n_classes)

    def forward(self,input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def init_weights(self):
        for p in self.parameters():
            sd = 1.0 / np.max(p.size())
            nn.init.normal_(p,0,sd)

    def maxdim(self):
        m = 0
        for p in self.parameters():
            m = np.max([m,np.max(p.size())])
        return m

    def effective_size(self):
        out=0
        for p in self.parameters():
            out += p.size().numel()
        return out