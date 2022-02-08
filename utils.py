import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from itertools import compress

n_classes = 90
labels = reuters.categories()

def load_data(config={}):
    """
    Load the Reuters dataset.
    top 8k features, excluding numbers
    Returns 
    -------
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=r"(?ui)\b[a-z]{2}[a-z]+\b")
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith("test/")]
    train = [d for d in documents if d.startswith("training/")]

    docs = {}
    docs["train"] = [reuters.raw(doc_id) for doc_id in train]
    docs["test"] = [reuters.raw(doc_id) for doc_id in test]
    xs = {"train": [], "test": []}
    xs["train"] = vectorizer.fit_transform(docs["train"]).toarray()
    features = vectorizer.get_feature_names()
    xs["test"] = vectorizer.transform(docs["test"]).toarray()
    ys = {"train": [], "test": []}
    ys["train"] = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train])
    ys["test"] = mlb.transform([reuters.categories(doc_id) for doc_id in test])
    
    data = {
        "x_train": xs["train"],
        "y_train": ys["train"],
        "x_test": xs["test"],
        "y_test": ys["test"],
        "labels": globals()["labels"],
        "features": features
    }
    return data




class ReutersTrain(Dataset):
    " Extends PyTorch's Dataset class so that its Dataloader can operate ontop of it"

    def __init__(self,data):
        self.x_train = data["x_train"]
        self.y_train = data["y_train"]

    def __getitem__(self,index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.x_train)

class ReutersTest(Dataset):
    " Extends PyTorch's Dataset class so that its Dataloader can operate ontop of it"

    def __init__(self,data):
        self.x_test = data["x_test"]
        self.y_test = data["y_test"]

    def __getitem__(self,index):
        return self.x_test[index], self.y_test[index]

    def __len__(self):
        return len(self.x_test)

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
    correct = 0
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
        #calculate accuracy
        out      = (output>0.5)
        correct += (out == labels).float().sum()
        total   += len(out)

    train_acc  = correct/total
    loss_total = loss_sum / batches_total
    
    return net, train_acc, loss_total

def test(net, dataloader, loss_fct, device):
    """ Test Function
    Requires (trained) Net and TestLoader
    Returns
    ------
    Test Accuracy, AUC-score, Validation-Loss
    """

    y_pred = []
    y_true = []

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

            out      = (output>0.5)
            correct += (out == labels).float().sum()
            total   += len(out)


            #auc: collect all datapoints to calculate auc in the end
            y_pred.extend( out.data.float().cpu().numpy() )         # bring output tensor to cpu & convert to list
            y_true.extend( labels.data.cpu().numpy() )              # same, safe complete batchresults in 1 list


    loss_total   = loss_sum / batches_total
    auc          = roc_auc_score(np.array(y_true).flatten().reshape(total,n_classes), np.array(y_pred).flatten().reshape(total,n_classes), average="weighted", multi_class = 'ovr')
    acc          = correct/total

    return acc, auc, loss_total




def feature_selection(data):

    model = ExtraTreesClassifier(n_estimators=50, random_state=42)
    model = model.fit(data["x_train"], data["y_train"])
    
    selector = SelectFromModel(model, prefit=True, threshold=1e-99) # deletes features with zero importance
    X_new = selector.transform(data["x_train"]) 
    
    X_test = selector.transform(data["x_test"])
    
    maske = selector.get_support()                                 # returns list of True,False
    features_remaining = list(compress(data["features"], maske))   # use it to select True-Features
    
    data_new = {
        
        "x_train": X_new,
        "y_train": data["y_train"],
        "x_test": X_test,
        "y_test": data["y_test"],
        "labels": data["labels"],
        "features": features_remaining
    }
    
    
    
    return data_new