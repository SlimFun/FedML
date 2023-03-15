from mpi4py import MPI

# from .FedAVGAggregator import FedAVGAggregator
# from .FedAVGTrainer import FedAVGTrainer
# from .FedAvgClientManager import FedAVGClientManager
# from .FedAvgServerManager import FedAVGServerManager
from .FedPAVAggregator import FedPAVAggregator
from .FedPAVTrainer import FedPAVTrainer
from .FedPavClientManager import FedPAVClientManager
from .FedPavServerManager import FedPAVServerManager

from ...standalone.fedpav.my_model_trainer_reid import MyModelTrainer as MyModelTrainerReID

def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedPav_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, traindata_cls_counts, class_num_dict, model_trainer=None, preprocessed_sampling_lists=None):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, train_data_num, train_data_global,
                    test_data_global, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                    model_trainer, preprocessed_sampling_lists)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
                    train_data_local_dict, test_data_local_dict, traindata_cls_counts, class_num_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global,
                train_data_local_dict, test_data_local_dict, train_data_local_num_dict, model_trainer, preprocessed_sampling_lists=None):
    # if model_trainer is None:
    #     if args.dataset == "stackoverflow_lr":
    #         model_trainer = MyModelTrainerTAG(model)
    #     elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #         model_trainer = MyModelTrainerNWP(model)
    #     else: # default model trainer is for classification problem
    #         model_trainer = MyModelTrainerCLS(model)
    model_trainer = MyModelTrainerReID(model)
    model_trainer.set_id(-1)

    # aggregator
    worker_num = size - 1
    aggregator = FedPAVAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  worker_num, device, args, model_trainer)

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None :
        server_manager = FedPAVServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = FedPAVServerManager(args, aggregator, comm, rank, size, backend,
            is_preprocessed=True, 
            preprocessed_client_lists=preprocessed_sampling_lists)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model, train_data_num, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, traindata_cls_counts, class_num_dict, model_trainer=None):
    client_index = process_id - 1

    # if model_trainer is None:
    #     if args.dataset == "stackoverflow_lr":
    #         model_trainer = MyModelTrainerTAG(model)
    #     elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
    #         model_trainer = MyModelTrainerNWP(model)
    #     else: # default model trainer is for classification problem
    #         model_trainer = MyModelTrainerCLS(model)
    model_trainer = MyModelTrainerReID(model)
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedPAVTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device, args, model_trainer, traindata_cls_counts)
    client_manager = FedPAVClientManager(args, trainer, class_num_dict, comm, process_id, size, backend)
    client_manager.run()
