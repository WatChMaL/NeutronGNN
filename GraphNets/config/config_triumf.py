# for commit 300052df3430228ef5e8bc55e46845d95d5e57f0

from config.easy_dict import EasyDict

config = EasyDict()

## Model
config.model_name = "gcn_kipf"
config.model_kwargs = {"w1":16, "w2":16, "w3":16}

## Data paths
config.data_path = "/fast_scratch/NeutronGNN/iwcd_mpmt_shorttank_neutrongnn_trainval.h5"
config.indices_file = "/fast_scratch/NeutronGNN/iwcd_mpmt_shorttank_neutrongnn_trainval_idxs.npz"

## Log location
config.dump_path = "/fast_scratch/NeutronGNN/dump/" + config.model_name

## Computer parameters
config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [1]

## Optimizer parameters
config.optimizer = "Adam"
config.optimizer_kwargs = {"lr":0.01, "weight_decay":5e-4}

## Scheduler parameters
config.scheduler_kwargs = {"mode":"min", "min_lr":1e-6, "patience":1, "verbose":True}
config.scheduler_step = 190

## Training parameters
config.batch_size = 64
config.epochs = 5

## Logging parameters for training
config.report_interval = 50
config.num_val_batches  = 16
config.valid_interval   = 200

## Validating parameters
config.validate_batch_size = 64
config.validate_dump_interval = 256
config.use_encoded_data = False
