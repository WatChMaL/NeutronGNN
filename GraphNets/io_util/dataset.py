import torch

from torch_geometric.data import Data, Dataset

import numpy as np
import h5py
import pickle


class WCH5Dataset(Dataset):
    """
    Dataset storing data from Water Cherenkov detector
    Data a fixed length array with elements [q, t, pos(x,y,z), dir(x,y,z)]
    Select only hits pmts using nhits. Rest are zeros.
    memory-maps the detector data from hdf5 file
    The detector data must be uncompresses and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    # Override the default implementation
    def _download(self):
        pass

    def _process(self):
        pass


    def __init__(self, path,
                 transform=None, pre_transform=None, pre_filter=None,
                 use_node_attr=False, use_edge_attr=False, cleaned=False):

        super(WCH5Dataset, self).__init__("", transform, pre_transform,
                                        pre_filter)

        f=h5py.File(path,'r')
        hdf5_event_data = f["event_data"]
        hdf5_labels=f["labels"]
        hdf5_nhits=f["nhits"]

        assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

        event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype

        #this creates a memory map - i.e. events are not loaded in memory here
        #only on get_item
        self.event_data = np.memmap(path, mode='r', shape=event_data_shape,
                                    offset=event_data_offset, dtype=event_data_dtype)

        #this will fit easily in memory even for huge datasets
        self.labels = np.array(hdf5_labels)
        self.nhits = np.array(hdf5_nhits)

        self.transform=transform

    def load_edges(self, nhits):
        edge_index = torch.ones([nhits, nhits], dtype=torch.int64)
        self.edge_index=edge_index.to_sparse()._indices()

    def get(self, idx):
        nhits = self.nhits[idx]
        #nhits = 300
        x = torch.from_numpy(self.event_data[idx, :nhits, :])
        y = torch.tensor([self.labels[idx]], dtype=torch.int64)
        self.load_edges(nhits)

        return Data(x=x, y=y, edge_index=self.edge_index)

    def __len__(self):
        return self.labels.shape[0]
