import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from ...model.reid.ft_net import ft_net
import torch.nn as nn


class FedPAVClientManager(ClientManager):
    def __init__(self, args, trainer, class_num_dict, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

        self.classifier_dict = dict()
        self.mocel_dict = dict()
        for k,v in class_num_dict.items():
            full_model = ft_net(v)
            classifier = full_model.classifier.classifier
            full_model.classifier.classifier = nn.Sequential()
            self.classifier_dict[k] = classifier
            self.mocel_dict[k] = full_model

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_STATISTICS_TO_CLIENT,
                                              self.handle_message_receive_statistics_from_server)
        
    def handle_message_receive_statistics_from_server(self, msg_params):
        logging.info("handle_message_receive_statistics_from_server")
        global_statistics = msg_params.get(MyMessage.MSG_ARG_KEY_STATISTICS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info(f"client {client_index} receive statistics {global_statistics} from server")
        
        self.__train()

    def handle_message_init(self, msg_params):
        print('handle init')
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_classifier(self.classifier_dict[int(client_index)], self.mocel_dict[int(client_index)])
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.__train()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        logging.info("#######training########### round_id = %d" % self.round_idx)
        self.trainer.restart_train()
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            post_complete_message_to_sweep_process(self.args)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
#         logging.info("send_model_to_server")
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_statistics_to_server(self, receive_id, statistics, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_STATISTICS_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_STATISTICS, self.get_sender_id())
        self.send_message(message)
        
    def __train(self):
        weights, local_sample_num, done = self.trainer.train(self.round_idx)
        if done:
            self.send_model_to_server(0, weights, local_sample_num)
        else:
            self.send_statistics_to_server(0, weights, local_sample_num)
        
#     def __train(self):
#         logging.info("#######training########### round_id = %d" % self.round_idx)
#         self.trainer.restart_train()
# #         sync_stats_freq = 2
#         while True:
#             weights, local_sample_num, done = self.trainer.train(self.round_idx)
#             if done:
#                 break
#             self.send_statistics_to_server(0, weights, local_sample_num)
# #         logging.info("train finished at round_id")
#         self.send_model_to_server(0, weights, local_sample_num)
# #         logging.info("send_model_to_server finished")
