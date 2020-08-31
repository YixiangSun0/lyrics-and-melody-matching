#!/usr/bin/env python
# coding: utf-8

# In[383]:


import torch
import numpy as np
import math
import os, sys, glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# In[384]:


path = r'/Users/yixiangsun/Desktop/torch_file'
file = glob.glob(os.path.join(path, "*.pth"))
#print(file[2225])
data = []
for x in file:
    data.append(torch.load(x))


# In[386]:


def alignment(note_list, lyrics_list):
#     print(note_list[0], len(lyrics_list))    
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
    #print(len(note_list_new))
    return note_list_new, lyrics_list_new


# In[389]:


original_data_L = []
original_data_M = []
#index = 0
for data1 in data:

    note,lyrics = alignment(data1[0],data1[1])

    # word embedding model
    from gensim.models import Word2Vec
    model = Word2Vec(np.array(data1[1])[:,2].tolist(), min_count=1,size = 50)

    #embed to get list
#     print(index)
    #index += 1
    
    for i in lyrics:
        s = str(i)
        n = ord(s[0])
        if n > 96 and n < 123 or n >64 and n < 91:
            #print(i)
            j = lyrics[lyrics.index(i)-1]
            lyrics[lyrics.index(i)] = j
            
            
    lyrics_song = [model.wv[i] for i in lyrics]
    note_song =[i for i in note]

    #turn list into tensor and unsqueeze tensor
    #print(np.array(lyrics_song).shape)
    tensor1 = np.array(lyrics_song).reshape(300, -1)
   
    tensor2 = np.array(note).reshape(300,-1)
   

    original_data_L.append(tensor1)
    original_data_M.append(tensor2)


# In[391]:


# tensor_combined = torch.FloatTensor(np.array(original_data))

class combined_data(Dataset):

    def __init__(self):
        
        self.n_samples = len(original_data_L)

        # here the last column is the class label, the rest are the features
        print(np.array(original_data_L).shape)
        self.x_data1 = torch.from_numpy(np.array(original_data_L)) # size [n_samples, n_features]
        self.x_data2 = torch.from_numpy(np.array(original_data_M,dtype=float))
        self.y_data = torch.from_numpy(np.ones(self.n_samples,dtype=float)) # size [n_samples, 1]
        print(self.x_data1.shape)

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data1[index], self.x_data2[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = combined_data()
# get first sample and unpack
first_data = dataset[0]
feature1, feature2, labels = first_data
print(feature1, feature2)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

# convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# Dummy Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations) #到这一步都没有问题


for epoch in range(num_epochs):
    for i, (feature1, feature2, labels) in enumerate(train_loader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Run your training process
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {feature1.shape} | Labels {labels.shape}')



# # train_dataset = torchvision.datasets.MNIST(root='./data', 
# #                                            train=True, 
# #                                            transform=torchvision.transforms.ToTensor(),  
# #                                            download=True)

# # train_loader = DataLoader(dataset=train_dataset, 
# #                                            batch_size=4, 
# #                                            shuffle=True)

# # look at one random sample
dataiter = iter(train_loader)
data = dataiter.next()
feature1, feature2, targets = data
print(feature1.shape, feature2.shape, targets.shape)


# In[ ]:



