# Phantom

- generatePoisonSet.py is used to generate and visualize the poisoned data set using CIFAR-10
- model_evaluation.py is used to evaluate a Wide ResNet28-2 model on CIFAR10 test set
- model is trained using the help of TorchSSL git hub repository (https://github.com/TorchSSL/TorchSSL.git). Train model with normal partially labeled CIFAR10 with 40 labeled set size, and poisoned partially labeled CIFAR10 with 40 labeled set size. 

- to run with free version of google colab the parameter used for fixmatch from fixmatch_cifar10_40_0.yaml is changed:
  - multiprocessing_distributed is set to False
  - gpu is set to 0
  - batch size is set to 32 instead of 64

- dependency used for the setup in google colab:
    - !pip install torch torchvision numpy pyyaml tqdm tensorboard tensorboardX

- clone the SSL repository:
    - !git clone https://github.com/TorchSSL/TorchSSL.git

- start training on clean data set:
1. move to TorchSSL directory: %cd TorchSSL
2. configuration of fix match with 40 label samples located at: TorchSSL/config/fixmatch/fixmatch_cifar10_40_0
  - multiprocessing_distributed is set to False
  - gpu is set to 0
  - batch size is set to 32 instead of 64
3. specify SSL algorithm configuration for fixmatch with 40 labeled sample: !python fixmatch.py --c config/fixmatch/fixmatch_cifar10_40_0.yaml

- resume training on clean data set:
1. move to TorchSSL directory %cd Torch SSL
2. create directory to place saved model:
   - !mkdir saved_models
   - %cd saved_models
   - !mkdir fixmatch_cifar10_40_0
   - %cd ..
3. import the saved model into the fixmatch_cifar10_40_0 directory
4. setup the fixmatch_cifar10_40_0.yaml configuration file for resume training
  - set resume to True
  - set load_path to ./saved_models/fixmatch_cifar10_40_0/latest_model.pth
5. specify SSL algorithm configuration for fixmatch with 40 labeled sample: !python fixmatch.py --c config/fixmatch/fixmatch_cifar10_40_0.yaml

- start phantom training
  - similar to normal training but instead we use a partially poisoned dataset where dataset is created using generatePoisonSet.py
    - we upload the poison data set into colab as zip file then unzip it: !unzip phantom_data.zip -d /content
    - we move data set to TorchSSL/phantom data: !mv phantom_data TorchSSL/phantom_data
    - specify to use poisoned data set for training: change the data_dir to ./phantom_data within the fixmatch of 40 labeled sample configuration file located at ./TorchSSL/config/fixmatch/fixmatch_cifar10_40_0.yaml

- to resume phantom training
  - similar to resume of normal training we create directory to place the saved model
  - set up configuration file: change resume to True and specify the load_path. And specify to use the poisoned dataset within TorchSSL/phantom_data that we added into TorchSSL

- to evaluate model on a clean test set use model_evaluation.y
  - it's specify to load model named latest_model.pth that we upload to google colab
      
      
    


