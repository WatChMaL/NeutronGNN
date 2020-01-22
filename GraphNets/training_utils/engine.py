# Python standard imports
from abc import ABC, abstractmethod
from time import strftime
from os import stat, mkdir

# PyTorch imports
from torch import device, load, save
from torch.nn import DataParallel
from torch.cuda import is_available

from io_util.data_handler import get_loaders, get_loaders_encoded
from training_utils.logger import CSVData

class Engine(ABC):

    def __init__(self, model, config):
        super().__init__()

        # Engine attributes
        self.model=model
        self.config=config

        # Determine the device to be used for model training and inference
        if (config.device == 'gpu') and config.gpu_list:
            print("Requesting GPUs. GPU list : " + str(config.gpu_list))
            self.devids=["cuda:{0}".format(x) for x in config.gpu_list]
            print("Main GPU : " + self.devids[0])

            if is_available():
                self.device=device(self.devids[0])
                if len(self.devids) > 1:
                    print("Using DataParallel on these devices: {}".format(self.devids))
                    self.model=DataParallel(self.model, device_ids=config.gpu_list, dim=0)
                print("CUDA is available")
            else:
                self.device=device("cpu")
                print("CUDA is not available")
        else:
            print("Unable to use GPU")
            self.device=device("cpu")

        # Send the model to the selected device
        self.model = DataParallel(self.model) if len(self.devids) > 1 else self.model # Changed
        self.model.to(self.device)

        # Setup the parameters tp save given the model type
        if type(self.model) == DataParallel:
            self.model_accs=self.model.module
        else:
            self.model_accs=self.model

        # Create the dataset object
        if config.use_encoded_data:
            out = get_loaders_encoded(config.data_path, config.indices_file,
                          config.edge_index_pickle, config.batch_size, config.num_data_workers)
        else:
            out = get_loaders(config.data_path, config.indices_file,
                          config.edge_index_pickle, config.batch_size, config.num_data_workers)

        self.train_loader, self.val_loader, self.dataset = out

        # Define the variant dependent attributes
        self.criterion=None

        # Create the directory for saving the log and dump files
        self.dirpath=config.dump_path + strftime("%Y%m%d_%H%M%S") + "/"
        try:
            stat(self.dirpath)
        except:
            print("Creating a directory for run dump at : {}".format(self.dirpath))
            mkdir(self.dirpath)

        # Logging attributes
        self.train_log=CSVData(self.dirpath + "log_train.csv")
        self.val_log=CSVData(self.dirpath + "log_val.csv")

#         # Save a copy of the config in the dump path
#         save_config(self.config, self.dirpath + "config_file.ini")   # changed

    @abstractmethod
    def forward(self, data, mode):
        """Forward pass using self.data as input."""
        raise NotImplementedError

    def backward(self, predict, expected):
        """Backward pass using the loss computed for a mini-batch."""
        self.optimizer.zero_grad()  # Reset gradient accumulation
        loss = self.criterion(predict, expected)
        loss.backward()# Propagate the loss backwards
        self.optimizer.step()       # Update the optimizer parameters

        return loss

    @abstractmethod
    def train(self):
        """Training loop over the entire dataset for a given number of epochs."""
        raise NotImplementedError

    def save_state(self, mode="latest", name=""):
        """Save the model parameters in a file.

        Args :
        mode -- one of "latest", "best" to differentiate
                the latest model from the model with the
                lowest loss on the validation subset (default "latest")
        """
        if name:
            path=self.dirpath + self.config.model_name + "_" + mode + "_" + name + ".pth"
        else:
            path=self.dirpath + self.config.model_name + "_" + mode + ".pth"

        # Extract modules from the model dict and add to start_dict
        modules=list(self.model_accs._modules.keys())
        state_dict={module: getattr(self.model_accs, module).state_dict() for module in modules}

        # Save the model parameter dict
        save(state_dict, path)

    def load_state(self, path):
        """Load the model parameters from a file.

        Args :
        path -- absolute path to the .pth file containing the dictionary
        with the model parameters to load from
        """
        # Open a file in read-binary mode
        with open(path, 'rb') as f:

            # Interpret the file using torch.load()
            checkpoint=load(f, map_location=self.device)

            print("Loading weights from file : {0}".format(path))

            local_module_keys=list(self.model_accs._modules.keys())
            for module in checkpoint.keys():
                if module in local_module_keys:
                    print("Loading weights for module = ", module)
                    getattr(self.model_accs, module).load_state_dict(checkpoint[module])

        self.model.to(self.device)
