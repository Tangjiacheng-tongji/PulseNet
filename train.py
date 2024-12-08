import pandas as pd
import torch
from network import RNN
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_data(path):
    data = pd.read_csv(path, sep='\t',header=None)
    train_data = data.iloc[:,1:]
    train_label = data[0]
    train_label = (train_label + 1) // 2
    return torch.from_numpy(train_data.values).unsqueeze(axis=-1).float(), torch.from_numpy(train_label.values)

train_data,train_label = get_data('../data/FordB/FordB_TRAIN.tsv')
test_data,test_label = get_data('../data/FordB/FordB_TEST.tsv')

train_label = train_label.unsqueeze(-1)
#train_label = torch.zeros(len(train_label), max(train_label)+1).scatter_(1, train_label, 1)

def train(model, num_epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    for epoch in range(num_epochs):
        pred = model(train_data)
        cost = criterion(pred, train_label.float())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print("Epoch [%d/%d] Loss %.4f"%(epoch+1, num_epochs, cost.item()))
    print("Training Finished!")
def test(model):
    res = torch.argmax(model(test_data),axis=1)
    return int(sum(res == test_label))/len(res)

rnn = RNN(1,64,1)
train(rnn, 100)

acc = test(rnn)
print(acc)
torch.save(rnn.state_dict(), "rnn_{}.pkl".format(round(acc,2)))
