'''
     Preprocessing for CIFAR10DVS, adapted from code for "Convolutional spiking
     neural networks (SNN) for spatio-temporal feature extraction" paper, 
     Samadzadeh et al.
     https://github.com/aa-samad/conv_snn
'''


import os
import torch
from torch.utils.data import Dataset

class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '{}.pt'.format(index))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


def get_cifar10dvs(data_path, network_config):
    print("loading CIFAR10-DVS")
    batch_size = network_config['batch_size']

    trainset = data_path + '/dvs-cifar10/train/'
    trainloader = torch.utils.data.DataLoader(DVSCifar10(trainset), batch_size=batch_size, shuffle=True, num_workers=8)

    testset = data_path + '/dvs-cifar10/test/'
    testloader = torch.utils.data.DataLoader(DVSCifar10(testset), batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader
