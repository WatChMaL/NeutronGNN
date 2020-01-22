# for commit 300052df3430228ef5e8bc55e46845d95d5e57f0

from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "encoded_cheby_highway"
config.model_kwargs = {"k":3, "layers":20, "graph_w":256, "concat_layers":[3,5,7,10,13,16,20], "lin_ws":[64, 64]}

config.data_path = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval.h5"
config.indices_file = "/data/WatChMaL/data/IWCDmPMT_4pi_fulltank_test/split_h5/IWCDmPMT_4pi_fulltank_test_graphnet_trainval_idxs.npz"
config.edge_index_pickle = "/data/WatChMaL/graphnets/visualization/mpmt_edges_dict.pkl"

config.dump_path = "/data/WatChMaL/graphnets/GraphNets/dump/" + config.model_name

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [6]

config.optimizer = "SGD"
config.optimizer_kwargs = {"lr":0.01, "weight_decay":1e-3, "momentum":0.9, "nesterov":True}

config.scheduler_kwargs = {"mode":"min", "min_lr":1e-6, "patience":1, "verbose":True}
config.scheduler_step = 190

config.batch_size = 64
config.epochs = 15

config.report_interval = 50
config.num_val_batches  = 16
config.valid_interval   = 200

config.validate_batch_size = 64
config.validate_dump_interval = 256
config.use_encoded_data = True
