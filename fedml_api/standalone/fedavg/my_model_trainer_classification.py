import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, cl, ml, traindata_cls_count):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        if args.client_optimizer == "sgd":
#             optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.comm_round*args.epochs)
        epoch_loss = []
#         for epoch in range(args.epochs):
        while self.epoch < args.epochs:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs, cova_score, feats = model(x)
#                 self.update_cova_memory(feats, labels, cl, ml)
#                 is_weight = (1.0 / 5000) / (traindata_cls_count[])
                is_weights = []
                for l in labels:
                    is_weights.append((1.0 / 10) / traindata_cls_count[l.cpu()])
                is_weights = torch.FloatTensor(is_weights).to(device)
#                 loss = is_weights * criterion(log_probs, labels)
                loss = (is_weights * criterion(log_probs, labels)).mean()
#                 print(is_weights)
#                 print(loss.shape)
#                 loss = (is_weights * loss).mean()
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, self.epoch, sum(epoch_loss) / len(epoch_loss)))
            scheduler.step()
            self.epoch += 1
            
            if self.epoch % args.freq_of_the_sync_stats == 0:
                return False
        return True
    
    def update_cova_memory(self, feats, labels, cl, ml):
#         covaM_list, mean_list = cal_covariance(feats)
        
        for f, label in zip(feats, labels):
            for i in range(len(cl)):
                if label.data.cpu() == i:
                    ml[i] = f.cpu()
#                     cl[i] = 
#             labels.append(label.cpu())
#             for i in range(len(cl)):
#                 if label.data.cpu() == i:
#                     cl[i] = lbd * cl[i] + (1-lbd) * covaM.cpu()
#                     f = torch.unsqueeze(f, 0)
#                     B, C, h, w = f.size()

#                     f = f.permute(1, 0, 2, 3)
#                     f = f.contiguous().view(C, -1)
#                     ml[i] = lbd * ml[i] + (1-lbd) * f.cpu()

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                log_probs, cova_score, feat = model(x)
                loss = criterion(log_probs, target)

                _, predicted = torch.max(log_probs, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
