import sys, os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.model.cv.resnet import resnet56

import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random

# client_id = 0
epochs = 2000 

dataset = 'cifar10'
data_dir = "./../../../data/cifar10"
partition_method = 'hetero'
partition_alpha = 0.5
client_num_in_total = 3
batch_size = 64
save_model_path = 'model/client_{0}_epochs_{1}.pt'

# device = f'cuda:{client_id}'
device = 'cuda:0'
parser = argparse.ArgumentParser()
parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')


train_data_num, test_data_num, train_data_global, test_data_global, \
train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
class_num = load_partition_data_cifar10(dataset, data_dir, partition_method,
                        partition_alpha, client_num_in_total, batch_size)

def train_model(client, epochs):
    lr = 0.001
    wd = 0.001
    
    client.model.to(client.device)
    client.model.train()
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, client.model.parameters()), lr=lr,
                                         weight_decay=wd, amsgrad=True)
    epoch_loss = []
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(client.train_data):
            x, labels = x.to(device), labels.to(device)
            client.model.zero_grad()
            log_probs = client.model(x)
            loss = criterion(log_probs, labels)
            loss.backward()

            # to avoid nan loss
            torch.nn.utils.clip_grad_norm_(client.model.parameters(), 1.0)

            optimizer.step()
            # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
            #            100. * (batch_idx + 1) / len(train_data), loss.item()))
            batch_loss.append(loss.item())
            
        if epoch % 200 == 0:
            torch.save(client.model.state_dict(), str.format(save_model_path, client.id, epoch))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            client.id, epoch, sum(epoch_loss) / len(epoch_loss)))

class Client(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, device, model):
        self.id = client_index
        self.train_data = train_data_local_dict[self.id]
        self.local_sample_number = train_data_local_num_dict[self.id]
        self.test_local = test_data_local_dict[self.id]
        
        self.device = device
        self.model = model
        
clients = []
for i in range(3):
    client = Client(i, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, device, resnet56(class_num=class_num))

    train_model(client, epochs)
# client = Client(client_id, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, device, resnet56(class_num=class_num))
# train_model(client, epochs)


