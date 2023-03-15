from .utils import transform_tensor_to_list
import torch


class FedPAVTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer, traindata_cls_counts):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None
        self.traindata_cls_counts = traindata_cls_counts
        self.traindata_cls_count = None

        self.device = device
        self.args = args
        self.cl = [torch.zeros((256,256)) for i in range(10)]
        self.ml = [torch.zeros((256,64)) for i in range(10)]

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_classifier(self, classifier, model):
        self.trainer.update_classifier(classifier, model)
        
    def get_epoch(self):
        return self.trainer.get_epoch()
     
    def restart_train(self):
        self.trainer.restart_train()

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        # self.test_local = self.test_data_local_dict[client_index]
        self.test_local = None
        p_local = []
        # for i in range(10):
        #     if i in self.traindata_cls_counts[client_index]:
        #         p_local.append(self.traindata_cls_counts[client_index][i] / float(self.local_sample_number))
        #     else:
        #         p_local.append(0.0)
        self.traindata_cls_count = p_local

    def train(self, round_idx = None):
        self.args.round_idx = round_idx
        done = self.trainer.train(self.client_index, self.train_local, self.device, self.args, self.cl, self.ml, self.traindata_cls_count)

        if done:
            weights = self.trainer.get_model_params()

            # transform Tensor to list
            if self.args.is_mobile == 1:
                weights = transform_tensor_to_list(weights)
            return weights, self.local_sample_number, done
        else:
            return None, None, done

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']

        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample