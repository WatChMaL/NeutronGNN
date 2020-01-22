# Python standard imports
from sys import stdout
from math import floor
from time import strftime, localtime
import numpy as np
import os

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyTorch Geometric imports
from torch_geometric.data import DataLoader

# Custom imports
from io_util.sampler import SubsetSequentialSampler
from training_utils.engine import Engine
from training_utils.optimizer import select_optimizer
from training_utils.logger import CSVData

class EngineGraph(Engine):

    def __init__(self, model, config):
        super().__init__(model, config)
        self.criterion=F.nll_loss
        self.optimizer=select_optimizer(config.optimizer, self.model_accs.parameters(),
                        **config.optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **config.scheduler_kwargs)
        self.keys = ['iteration', 'epoch', 'loss', 'acc']

    def forward(self, data, mode="train"):
        """Overrides the forward abstract method in Engine.py.

        Args:
        mode -- One of 'train', 'validation'
        """

        # Set the correct grad_mode given the mode
        if mode == "train":
            self.model.train()
        elif mode in ["validation"]:
            self.model.eval()

        return self.model(data)

    def train(self):
        """Overrides the train method in Engine.py.

        Args: None
        """

        epochs          = self.config.epochs
        report_interval = self.config.report_interval
        valid_interval  = self.config.valid_interval
        num_val_batches = self.config.num_val_batches
        scheduler_step  = self.config.scheduler_step

        # Initialize counters
        epoch=0.
        iteration=0

        # Parameter to upadte when saving the best model
        best_val_loss=1000000.
        avg_val_loss=1000.
        next_scheduler = scheduler_step

        val_iter = iter(self.val_loader)

        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch', np.round(epoch).astype(np.int),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))

            # Local training loop for a single epoch
            for data in self.train_loader:
                data = data.to(self.device)

                # Update the epoch and iteration
                epoch+=1. / len(self.train_loader)
                iteration += 1

                # Do a forward pass
                res=self.forward(data, mode="train")

                # Do a backward pass
                loss = self.backward(res, data.y)

                # Calculate metrics
                acc = res.argmax(1).eq(data.y).sum().item()/data.y.shape[0]

                # Record the metrics for the mini-batch in the log
                self.train_log.record(self.keys, [iteration, epoch, loss, acc])
                self.train_log.write()
                # Print the metrics at report_intervals
                if iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Acc %1.3f"
                          % (iteration, epoch, loss, acc))

                # Run validation on valid_intervals
                if iteration % valid_interval == 0:
                    val_loss=0.
                    val_acc=0.
                    with torch.no_grad():
                        for val_batch in range(num_val_batches):
                            try:
                                data=next(val_iter)
                            except StopIteration:
                                val_iter=iter(self.val_loader)
                                data=next(val_iter)
                            data = data.to(self.device)

                            # Extract the event data from the input data tuple
                            res=self.forward(data, mode="validation")
                            acc = res.argmax(1).eq(data.y).sum().item()/data.y.shape[0]

                            val_loss+=self.criterion(res, data.y)
                            val_acc+=acc

                    val_loss /= num_val_batches
                    val_acc /= num_val_batches
                    if iteration > next_scheduler:
                        self.scheduler.step(val_loss)
                        next_scheduler += scheduler_step


                    # Record the validation stats to the csv
                    self.val_log.record(self.keys, [iteration, epoch, val_loss, val_acc])
                    self.val_log.write()

                    # Save the best model
                    if val_loss < avg_val_loss:
                        self.save_state(mode="best", name="{}_{}".format(iteration, val_loss))
                        best_val_loss = val_loss
                        avg_val_loss = (val_loss * avg_val_loss)**0.5

                    # Save the latest model
                    self.save_state(mode="latest")

            self.save_state(mode="latest", name="epoch_{}".format(np.round(epoch).astype(np.int)))

        self.val_log.close()
        self.train_log.close()

    def validate(self, subset, name="current"):
        """Overrides the validate method in Engine.py.

        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        """
        # Print start message
        if subset == "train":
            message="Validating model on the train set"
        elif subset == "validation":
            message="Validating model on the validation set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None

        print(message)

        # Setup the path to save output
        # Setup indices to use
        if subset == "train":
            self.log=CSVData(os.path.join(self.dirpath,"train_validation_log_{}.csv".format(name)))
            output_path=os.path.join(self.dirpath, "train_validation_{}".format(name))
            validate_indices = self.dataset.train_indices
        else:
            self.log=CSVData(os.path.join(self.dirpath,"valid_validation_log_{}.csv".format(name)))
            output_path=os.path.join(self.dirpath, "valid_validation_{}".format(name))
            validate_indices = self.dataset.val_indices

        os.makedirs(output_path)
        data_iter = DataLoader(self.dataset, batch_size=self.config.validate_batch_size,
                               num_workers=self.config.num_data_workers,
                               pin_memory=True, sampler=SubsetSequentialSampler(validate_indices))

        key_val = {"index":[], "label":[], "pred":[], "pred_val":[]}

        avg_loss = 0
        avg_acc = 0
        indices_iter = iter(validate_indices)

        dump_interval = self.config.validate_dump_interval
        dump_index = 0

        with torch.no_grad():
            for iteration, data in enumerate(data_iter):
                gpu_data = data.to(self.device)

                stdout.write("Iteration : {}, Progress {} \n".format(iteration, iteration/len(data_iter)))
                res=self.forward(gpu_data, mode="validation")

                acc = res.argmax(1).eq(gpu_data.y).sum().item()
                loss = self.criterion(res, gpu_data.y) * data.y.shape[0]
                avg_acc += acc
                avg_loss += loss

                # Log/Report
                self.log.record(["Iteration", "loss", "acc"], [iteration, loss, acc/data.y.shape[0]])
                self.log.write()

                # Log/Report
                for label, pred, preds in zip(data.y.tolist(), res.argmax(1).tolist(), res.exp().tolist()):
                    key_val["index"].append(next(indices_iter))
                    key_val["label"].append(label)
                    key_val["pred"].append(pred)
                    key_val["pred_val"].append(preds)

                # Check if iteration is valid_dump_interval
                if len(key_val["index"]) >= dump_interval:
                    print("dumping")
                    name = os.path.join(output_path, "{}.npz".format(dump_index))
                    np.savez(name, **key_val)

                    dump_index += 1
                    key_val = {key:[] for key in key_val}

        self.log.close()

        name = os.path.join(output_path, "{}.npz".format(dump_index))
        np.savez(name, **key_val)

        avg_acc/=len(validate_indices)
        avg_loss/=len(validate_indices)

        stdout.write("Overall acc : {}, Overall loss : {}\n".format(avg_acc, avg_loss))

