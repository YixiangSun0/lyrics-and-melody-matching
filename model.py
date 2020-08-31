#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load the model 
# model = RNN(emb_dim, hid_dim)
# model.load_state_dict(torch.load(SAVE_PATH))


# In[69]:


import torch
import torch.nn as nn
import torch.optim as optim
import time
import real_data_processing
from real_data_processing import combined_data
from torch.utils.data import Dataset, DataLoader

skip_header=True
class RNN(nn.Module):
    def __init__(self, emb_dim_L, hid_dim_L, emb_dim_M, hid_dim_M):
        super(RNN, self).__init__()
        self.rnn_L = nn.LSTM(input_size=emb_dim_L,
                           hidden_size=hid_dim_L,
                           num_layers=2,
                           bidirectional=False)
        
        self.rnn_M = nn.LSTM(input_size=emb_dim_M,
                           hidden_size=hid_dim_M,
                           num_layers=2,
                           bidirectional=False)
        
        self.fc_L = nn.Linear(hid_dim_L, 128)
        self.fc_M = nn.Linear(hid_dim_M, 128)


    def forward(self, lyrics, notes):
        lyrics = lyrics.permute(1,0,2)
        notes = notes.permute(1,0,2)
        output_L, (hidden_L, cell) = self.rnn_L(lyrics)
#         print(notes, lyrics.shape)
        output_M, (hidden_M, cell) = self.rnn_M(notes)
        
        final_output_L = self.fc_L(hidden_L[-1])
        final_output_M = self.fc_M(hidden_M[-1])
        
        return final_output_L, final_output_M
        



model_LM = RNN(50, 256, 1, 256)


def binary_accuracy(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = (preds==y).float()
    acc = correct.sum() / len(correct) 
    return acc
print (model_LM)

LR = 1e-4

#create random dataset
# train_data_combined = [[torch.randn(300, 1, 50),torch.randn(300, 1, 1),1],[torch.randn(300, 1, 50),torch.randn(300, 1, 1),0],[torch.randn(300, 1, 50),torch.randn(300, 1, 1),1]]
# dev_data_combined = [[torch.randn(300, 1, 50),torch.randn(300, 1, 1),1],[torch.randn(300, 1, 50),torch.randn(300, 1, 1),0],[torch.randn(300, 1, 50),torch.randn(300, 1, 1),1]]

def loss_fn(feature1, feature2, label):
    similarity = cos(feature1, feature2)
    loss = nn.MSELoss() #loss 可改
    return loss(similarity, label)

def train(model_LM, train_data_combined, epoch):

    epoch_loss = 0
    model_LM.train()
    
    for batch_idx, batch in enumerate(train_data_combined, 0):
        
        
        lyrics = batch[0]
        notes = batch[1].float()
        label = batch[2]
        
        label = torch.autograd.variable(label).float()
        feature_L, feature_M = model_LM(lyrics, notes)
        optimizer.zero_grad()
        
        loss = loss_fn(feature_L, feature_M, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        
        if batch_idx % 10 == 0:
            print ("Epoch: {}, Batch Idx: {}, Loss: {}".format(epoch, batch_idx, loss.item()))
        
    return epoch_loss/len(train_data_combined)
    
    

def evaluate(model_LM, dev_data_combined):

    epoch_acc = 0
    
    #model_LM.test()
    for batch_idx, batch in enumerate(dev_data_combined):
        lyrics = batch[0]#.lyrics
        notes = batch[1].float()
        label = batch[2]#.label
        
        feature_L, feature_M = model_LM(lyrics, notes) #text_lengths).squeeze(1)
        
        pred = (cos(feature_L, feature_M) > 0.5)
        
        acc = sum(pred == label)*1.0/len(dev_data_combined)
        
        epoch_acc += acc.item()
        return epoch_acc/len(dev_data_combined)


def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        mins = int(elapsed_time / 60)
        secs = int(elapsed_time - (mins)*60)
        print ("Time: {} mins {} secs".format(mins, secs))

EPOCH = 10


dataset = combined_data()

# get first sample and unpack
# first_data = dataset[0]
# feature1, feature2, labels = first_data
# print(feature1, feature2)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_data_combined = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

dev_data_combined = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

optimizer = optim.Adam(model_LM.parameters(), lr=LR)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
for epoch in range(1, EPOCH+1):
        start_time = time.time() 
    
        train_loss = train(model_LM, train_data_combined, epoch)
        
        dev_acc = evaluate(model_LM, dev_data_combined)
        end_time = time.time()
        epoch_time(start_time, end_time)
        print ("Epoch: {}, Acc: {}, Train loss: {}".format(epoch, dev_acc, train_loss))
    
        SAVE_PATH = '%03d.pth' % epoch
        torch.save(model_LM.state_dict(), SAVE_PATH)
# save model


# In[ ]:


# load the model 
# model = RNN(emb_dim, hid_dim)
# model.load_state_dict(torch.load(SAVE_PATH))

