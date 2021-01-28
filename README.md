# [ML Reproducibility Challenge:] Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks (TSSL-BP)

This repository is for the reproduction of [Temporal Spike Sequence Learning via Backpropagation for Deep Spiking Neural Networks](https://arxiv.org/abs/2002.10085). 

## Requirements (https://github.com/stonezwr/TSSL-BP/)
### Dependencies and Libraries
* python 3.7
* pytorch
* torchvision

### Installation
To install requirements:

```setup
pip install -r requirements.txt
```

## Training
### Before running
Modify the data path and network settings in the [config files](https://github.com/stonezwr/TSSL-BP/tree/master/Networks). 

Select the index of GPU in the [main.py](https://github.com/stonezwr/TSSL-BP/blob/master/main.py#L198) (0 by default)

### Run the code
```sh
$ python main.py -config Networks/config_file.yaml
$ python main.py -config Networks/config_file.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```

## Results
Performance comparison between original paper and this reproduction:

### MNIST

                   | Network Size         | Time Steps | Epochs | Mean | Stddev | Best |
------------------| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
Original paper | 15C5-P2-40C5-P2-300   |     5         |     100      |  99.50% | 0.02% |  99.53% |
Reproduction | 15C5-P2-40C5-P2-300   |     5         |     100      |  99.40% | 0.04% |  99.47% |


### CIFAR 10
                   | Network Size | Time Steps | Epochs | Mean | Stddev | Best |
------------------ | ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
Original paper | 96C3-256C3-P2-384C3-P2-384C3-256C3-1024-1024  |     5        |     150      |  88.98% | 0.27% |  89.22% |
Reproduction | 96C3-256C3-P2-384C3-P2-384C3-256C3-1024-1024  |     5        |     150      |  88.96% | 0.10% |  89.07% |
------------------ | ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- | 
Original paper | 128C3-256C3-P2-512C3-P2-1024C3-512C3-1024-512   |     5         |     150      |  N/A | N/A |  91.41% |
Reproduction | 128C3-256C3-P2-512C3-P2-1024C3-512C3-1024-512   |     5         |     150      |  N/A | N/A |  89.61% |

