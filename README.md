# Phantom

generatePoisonSet.py is used to generate and visualize the poisoned data set using CIFAR-10
model_evaluation.py is used to evaluate a Wide ResNet28-2 model on CIFAR10 test set
model is trained using the help of TorchSSL git hub repository (https://github.com/TorchSSL/TorchSSL.git). Train model with normal partially labeled CIFAR10 with 40 labeled set size, and poisoned partially labeled CIFAR10 with 40 labeled set size. 

to run with free version of google colab the parameter used for fixmatch from fixmatch_cifar10_40_0.yaml is changed:
multiprocessing_distributed is set to False
gpu is set to 0
batch size is set to 32 instead of 64

