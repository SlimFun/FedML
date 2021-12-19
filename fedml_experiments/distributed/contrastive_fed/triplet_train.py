import sys, os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.model.contrastive_cv.resnet_with_embedding import Resnet56

import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
import pickle

# from triplet_loss import TripletLoss
from hard_triplet_loss import TripletLoss

dataset = 'cifar10'
data_dir = "./../../../data/cifar10"
# partition_method = 'hetero'
partition_method = 'homo'
partition_alpha = 0.5
client_num_in_total = 3
batch_size = 100
total_epochs = 500
save_model_path = 'model/cs_{0}_{1}_client_{2}_triplet_epochs_{3}.pt'

device = 'cuda:1'
# parser = argparse.ArgumentParser()
# parser.add_argument('--client_optimizer', type=str, default='adam',
#                         help='SGD with momentum; adam')
# parser.add_argument('--epochs', type=int, default=5, metavar='EP',
#                         help='how many epochs will be trained locally')


# train_data_num, test_data_num, train_data_global, test_data_global, \
# train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
# class_num = load_partition_data_cifar10(dataset, data_dir, partition_method,
#                         partition_alpha, client_num_in_total, batch_size)
with open(f'dataset_{partition_method}_{client_num_in_total}.pickle', 'rb') as f:
    dataset = pickle.load(f)

class Client(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, device, model):
        self.id = client_index
        self.train_data = train_data_local_dict[self.id]
        self.local_sample_number = train_data_local_num_dict[self.id]
        self.test_local = test_data_local_dict[self.id]
        
        self.device = device
        self.model = model
        
# print(f'train_data_local_num')
model = Resnet56(class_num=dataset[-1], neck='bnneck')
client_1 = Client(0, dataset[5], dataset[4], dataset[6], device, model)

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(client, epochs):
#     learning_rate = 0.001
#     wd = 0.0001
#     learning_rate = 0.00035
#     wd = 0.0005
    
    margin = 0.3
    
    client.model.to(client.device)
    client.model.train()
    
    triplet = TripletLoss(margin)
#     triplet = HardTripletLoss(margin)
    criterion = nn.CrossEntropyLoss().to(device)
    
#     curr_lr = learning_rate
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, client.model.parameters()), lr=curr_lr,
#                                          weight_decay=wd, amsgrad=True)
    optimizer = torch.optim.SGD(client.model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    epoch_loss = []
    total_step = len(client.train_data)
    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(client.train_data):
            x, labels = x.to(device), labels.to(device)
            client.model.zero_grad()
            score, feat = client.model(x)
            
#             dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(-1), dim=1, keepdim=True)
#             print(dist_mat[is_pos].contiguous())
#             dist_ap, relative_p_inds = torch.max(
#                 dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
            
            tri_loss, tri_prec = triplet(feat, labels)
            loss = criterion(score, labels) + tri_loss
            loss.backward()

            optimizer.step()
            batch_loss.append(loss.item())
            
        if epoch % 100 == 0:
            torch.save(client.model.state_dict(), str.format(save_model_path, client_num_in_total, partition_method, client.id, epoch))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            client.id, epoch, sum(batch_loss) / len(batch_loss)))
        
    torch.save(client.model.state_dict(), str.format(save_model_path, client_num_in_total, partition_method, client.id, epochs))
    
train_model(client_1, 400)