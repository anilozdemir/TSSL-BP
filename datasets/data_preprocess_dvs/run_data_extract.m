% converts to .mat files
% 
% Preprocessing for CIFAR10DVS, adapted from code for "Convolutional spiking
% neural networks (SNN) for spatio-temporal feature extraction" paper 
% Samadzadeh et al.
% https://github.com/aa-samad/conv_snn

clc
clear

labels = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}; % labels

data_path = 'E:\\CIFAR10-DVS\\cifar-raw\\raw'; % update to own datapath
dest_path = 'dvs-cifar10'; % add save destination path

for folder0 = 1:length(labels)
    for file0 = 0:999
        if mod(file0, 10) == 0
           fprintf('step: %s %d\n', labels{folder0}, file0)
        end
        addr = sprintf('%\\%s\\cifar10_%s_%d.aedat', data_path, labels{folder0}, labels{folder0}, file0);
        out1 = dat2mat(addr);
        save(sprintf('%\\%s\\%d.mat', dest_path, labels{folder0}, file0), 'out1')
    end
end