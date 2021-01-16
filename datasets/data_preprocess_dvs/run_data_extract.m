% converts to .mat files
% 
% Preprocessing for CIFAR10DVS, adapted from code for "Convolutional spiking
% neural networks (SNN) for spatio-temporal feature extraction" paper 
% Ali Samadzadeh et al.
% https://github.com/aa-samad/conv_snn

clc
clear

labels = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}; % labels

data_split = 699;
total_data = 999;

for folder0 = 1:length(labels)
    % train data
    for file0 = 0:data_split
        addr = sprintf('E:\\CIFAR10-DVS\\cifar-raw\\raw\\%s\\cifar10_%s_%d.aedat', labels{folder0}, labels{folder0}, file0);
        out1 = dat2mat(addr);
        save(sprintf('E:\\CIFAR10-DVS\\dvs-cifar10\\train\\%s\\%d.mat', labels{folder0}, file0), 'out1')
    end
    
    % test data
    for file0 = data_split+1:total_data
        addr = sprintf('E:\\CIFAR10-DVS\\cifar-raw\\raw\\%s\\cifar10_%s_%d.aedat', labels{folder0}, labels{folder0}, file0);
        out1 = dat2mat(addr);
        save(sprintf('E:\\CIFAR10-DVS\\dvs-cifar10\\test\\%s\\%d.mat', labels{folder0}, file0), 'out1')
    end
end