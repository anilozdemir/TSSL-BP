# Preprocessing for CIFAR10DVS, adapted from code for "Convolutional spiking
# neural networks (SNN) for spatio-temporal feature extraction" paper 
# Ali Samadzadeh et al.
# https://github.com/aa-samad/conv_snn

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isfile
from scipy.io import loadmat


class DVSCifar10(Dataset):
    def __init__(self, dataset_path, n_steps, transform=None):
        self.path = dataset_path
        self.samples = []
        self.labels = []
        self.transform = transform
        self.n_steps = n_steps
        
        mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird'  ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'   ,
            6 :'frog'  ,
            7 :'horse' ,
            8 :'ship'  ,
            9 :'truck'     }

        for class0 in mapping.keys():
            sample_dir = dataset_path + mapping[class0] + '/'
            for f in listdir(sample_dir):
                filename = sample_dir + "{}".format(f)
                if isfile(filename):
                    self.samples.append(filename)
                    self.labels.append(class0)

    def __getitem__(self, index):
        filename = self.samples[index]
        label = self.labels[index]

        events = loadmat(filename)['out1']
        data = np.zeros((2, 128, 128, self.n_steps))
        
        # --- building time surfaces
        for i in range(self.n_steps): # frames
            r1 = i * (events.shape[0] // self.n_steps)
            r2 = (i + 1) * (events.shape[0] // self.n_steps) # split into 10 frames
            data[events[r1:r2, 3], events[r1:r2, 1], events[r1:r2, 2], i] += events[r1:r2, 0] # add each frame
        
        for i in range(10): # normalise across each time frame?
            data[:, :, :, i] = data[:, :, :, i] / np.max(data[:, :, :, i])
            
        if self.transform:
            data = self.transform(data)
            data = data.type(torch.float32)
        else:
            data = torch.FloatTensor(data)

        return data, label

    def __len__(self):
        return len(self.samples)


def get_cifar10dvs(data_path, network_config):
    n_steps = network_config['n_steps']
    batch_size = network_config['batch_size']
    print("loading CIFAR10-DVS")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
    train_path = data_path + '/dvs-cifar10/train/'
    test_path = data_path + '/dvs-cifar10/test/'
    
    trainset = DVSCifar10(train_path, n_steps)
    testset = DVSCifar10(test_path, n_steps)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader
