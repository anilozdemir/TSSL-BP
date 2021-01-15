'''
     Preprocessing for CIFAR10DVS, adapted from code for "Convolutional spiking
     neural networks (SNN) for spatio-temporal feature extraction" paper,
     Samadzadeh et al
     https://github.com/aa-samad/conv_snn
'''

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from scipy.io import loadmat

# label number mapping
mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird' ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'    ,
            6 :'frog'   ,
            7 :'horse'       ,
            8 :'ship'      ,
            9 :'truck'     }


def gather_addr(directory, start_id, end_id): # creates list of all filenames
    fns = []
    for i in range(start_id, end_id):
        for class0 in mapping.keys():
            file_name = directory + '/' + mapping[class0] + '/' + "{}".format(i) + '.mat'
            fns.append(file_name)
    return fns


def events_to_frames(filename, dt):
    label_filename = filename[:].split('/')[1]
    label = int(list(mapping.values()).index(label_filename))
    frames = np.zeros((2, 128, 128, 10)) # axis flipped so in line with NMNIST
    events = loadmat(filename)['out1']

    # --- building time surfaces
    for i in range(10): # frames
        r1 = i * (events.shape[0] // 10)
        r2 = (i + 1) * (events.shape[0] // 10) # split into 10 frames
        frames[events[r1:r2, 3], events[r1:r2, 1], events[r1:r2, 2], i] += events[r1:r2, 0] # add each frame

    for i in range(10): # normalise across each time frame?
        frames[:, :, :, i] = frames[:, :, :, i] / np.max(frames[:, :, :, i])

    return frames, label


def create_npy():
    
    # data filename locations
    train_filename = 'dvs-cifar10/train/{}.pt'
    test_filename = 'dvs-cifar10/test/{}.pt'
    
    # crete test and train folders if do not exist
    if not os.path.exists('dvs-cifar10/train'):
        os.mkdir('dvs-cifar10/train')
    if not os.path.exists('dvs-cifar10/test'):
        os.mkdir('dvs-cifar10/test')

    # portion of test: train data
    train_test_portion = 0.7

    # list of filenames
    fns_train = gather_addr('dvs-cifar10', 0, int(train_test_portion * 1000))
    fns_test = gather_addr('dvs-cifar10', int(train_test_portion * 1000), 1000)

    print("processing training data...")
    
    key = -1 # for printing amount completed
    for file_d in fns_train: # process training data
    
        if key % 100 == 0: # print amount completed
            print("\r\tTrain data {:.2f}% complete\t\t".format(key / train_test_portion / 100), end='')
            
        frames, labels = events_to_frames(file_d, dt=5000) # convert to frames
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor([labels,])],  
                   train_filename.format(key)) # save in .pt tensor

    print("\nprocessing testing data...")
    key = -1
    for file_d in fns_test:
        if key % 100 == 0:
            print("\r\tTest data {:.2f}% complete\t\t".format(key / (1 - train_test_portion) / 100), end='')
        frames, labels = events_to_frames(file_d, dt=5000)
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor([labels,])],
                   test_filename.format(key))
    print('\n')

if __name__ == "__main__":
    create_npy()