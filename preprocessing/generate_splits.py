#!/usr/bin/evn/ python

import h5py
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate split indices")
    parser.add_argument("h5_file",
                        type=str,
                        help="Path to h5_file,\
                        must contain 'event_data'")
    parser.add_argument('output_file', type=str,
                        help="Path to output file")
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--test_split', type=float, default=0.1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = parse_args()

    lines = 0

    with h5py.File(config.h5_file, 'r') as f:
        lines = f['event_data'].shape[0]

    if lines < 3:
        raise ValueError("No enough data")

    indices = np.random.permutation(lines)
    val_end = np.ceil(lines * config.val_split).astype(np.int)
    test_end = val_end + np.ceil(lines * config.test_split).astype(np.int)

    np.savez(config.output_file, train_idxs=indices[test_end:],
            val_idxs=indices[:val_end], test_idxs=indices[val_end:test_end])

