# data processing
import torch
import numpy as np
import math
import os, sys, glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader



path = r'/Users/yixiangsun/Desktop/torch_file'
file = glob.glob(os.path.join(path, "*.pth"))
data = []
for x in file:
    data.append(torch.load(x))




def alignment(note_list, lyrics_list):  
    note_list_new = []
    lyrics_list_new = []
    i = j = 0
    while i < len(note_list) and j < len(lyrics_list):
        while not (note_list[i][1] == lyrics_list[j][1]):
            #print(i)
            lyrics_list_new.append(lyrics_list[j][2])
            note_list_new.append(note_list[i][2])
            i += 1
            if i >= len(note_list):
                break
         
        if i >= len(note_list) or j>= len(lyrics_list):
            break
            
            
        note_list_new.append(note_list[i][2])
        lyrics_list_new.append(lyrics_list[j][2])
        i += 1 
        j += 1

    if len(lyrics_list_new) or len(note_list_new) < 300:
        note_list_new = note_list_new + note_list_new
        lyrics_list_new = lyrics_list_new + lyrics_list_new
        
    if len(lyrics_list_new) or len(note_list_new) < 300:
        note_list_new = note_list_new + note_list_new
        lyrics_list_new = lyrics_list_new + lyrics_list_new
        
    if len(lyrics_list_new) or len(note_list_new) < 300:
        note_list_new = note_list_new + note_list_new
        lyrics_list_new = lyrics_list_new + lyrics_list_new

    note_list_new = note_list_new[:300]
    lyrics_list_new = lyrics_list_new[:300]
    return note_list_new, lyrics_list_new



original_data_L = []
original_data_M = []
for data1 in data:

    note,lyrics = alignment(data1[0],data1[1])

    # word embedding model
    from gensim.models import Word2Vec
    model = Word2Vec(np.array(data1[1])[:,2].tolist(), min_count=1,size = 50)

    
    for i in lyrics:
        s = str(i)
        n = ord(s[0])
        if n > 96 and n < 123 or n >64 and n < 91:
            j = lyrics[lyrics.index(i)-1]
            lyrics[lyrics.index(i)] = j
            
            
    lyrics_song = [model.wv[i] for i in lyrics]
    note_song =[i for i in note]


    tensor1 = np.array(lyrics_song).reshape(300, -1)
   
    tensor2 = np.array(note).reshape(300,-1)
   

    original_data_L.append(tensor1)
    original_data_M.append(tensor2)


class combined_data(Dataset):

    def __init__(self):
        
        self.n_samples = len(original_data_L)*2


        self.x_data1 = torch.from_numpy(np.concatenate((np.array(original_data_L), np.array(original_data_L))))
        self.x_data2 = torch.from_numpy(np.concatenate((np.array(original_data_M,dtype=float), sklearn.utils.shuffle(np.array(original_data_M,dtype=float)))))
        self.y_data = torch.from_numpy(np.concatenate((np.ones(self.n_samples,dtype=float), np.zeros(self.n_samples,dtype=float))))
        print(self.x_data1.shape, self.x_data2.shape, self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data1[index], self.x_data2[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
                 
dataset = combined_data()
print (len(dataset))
test_dataset, train_dataset = sklearn.model_selection.train_test_split(dataset, test_size=1000, train_size=3450)



# the RNN training network
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn import model_selection

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
        output_M, (hidden_M, cell) = self.rnn_M(notes)
        
        final_output_L = self.fc_L(hidden_L[-1])
        final_output_M = self.fc_M(hidden_M[-1])
        
        return final_output_L, final_output_M
        



model_LM = RNN(50, 256, 1, 256)



print (model_LM)

LR = 1e-4


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
    
    for batch_idx, batch in enumerate(dev_data_combined):
        lyrics = batch[0]
        notes = batch[1].float()
        label = batch[2]
        
        
        feature_L, feature_M = model_LM(lyrics, notes)
        
        output = cos(feature_L, feature_M).float()
        pred = []
        for i in output:
            if i > 0.999:
                pred.append(1)
            if i < -0.999:
                pred.append(0)
            else: 
                pred.append(0.5)
        pred = torch.from_numpy(np.array(pred))
        acc = sum(pred == label)*1.0/len(label)
        
        epoch_acc += acc.item()
    return epoch_acc/(batch_idx+1)
    
    
def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        mins = int(elapsed_time / 60)
        secs = int(elapsed_time - (mins)*60)
        print ("Time: {} mins {} secs".format(mins, secs))

EPOCH = 20



test_dataset, train_dataset = sklearn.model_selection.train_test_split(dataset, test_size=1000, train_size=3450)

train_data_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

dev_data_loader = DataLoader(dataset=test_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

optimizer = optim.Adam(model_LM.parameters(), lr=LR)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
for epoch in range(1, EPOCH+1):
        start_time = time.time()
    
        train_loss = train(model_LM, train_data_loader, epoch)
        
        dev_acc = evaluate(model_LM, dev_data_loader)
        end_time = time.time()
        epoch_time(start_time, end_time)
        print ("Epoch: {}, Acc: {}, Train loss: {}".format(epoch, dev_acc, train_loss))
    
        SAVE_PATH = '%03d.pth' % epoch
        torch.save(model_LM.state_dict(), SAVE_PATH)
