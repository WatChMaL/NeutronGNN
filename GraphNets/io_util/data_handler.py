from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import DataLoader

import numpy as np

from io_util.dataset import WCH5Dataset
from io_util.transform import transform_reshape

class WCH5Dataset_trainval(WCH5Dataset):
    """
    Dataset storing image-like data from Water Cherenkov detector
    memory-maps the detector data from hdf5 file
    The detector data must be uncompressed and unchunked
    labels are loaded into memory outright
    No other data is currently loaded
    """

    def __init__(self, path, indices_file,
                 edge_index_pickle, nodes=15808,
                 transform=None, pre_transform=None, pre_filter=None,
                 use_node_attr=False, use_edge_attr=False, cleaned=False):

        super(WCH5Dataset_trainval, self).__init__( path,
                 edge_index_pickle, nodes=nodes,
                 transform=transform, pre_transform=pre_transform, pre_filter=pre_filter,
                 use_node_attr=use_node_attr, use_edge_attr=use_edge_attr, cleaned=cleaned)

        all_indices = np.load(indices_file)

        self.train_indices = all_indices["train_idxs"]
        self.val_indices = all_indices["val_idxs"]

def get_loaders(path, indices_file, edges_dict_pickle, batch_size, workers):

    dataset = WCH5Dataset_trainval(path, indices_file, edges_dict_pickle)

    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

    return train_loader, val_loader, dataset


def get_loaders_encoded(path, indices_file, edges_dict_pickle, batch_size, workers):
    
    dataset = WCH5Dataset_trainval(path, indices_file, edges_dict_pickle,
                                nodes=832, transform=transform_reshape((832,38)))
    train_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

    val_loader=DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                            pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))
    return train_loader, val_loader, dataset
