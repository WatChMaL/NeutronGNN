import torch
import numpy as np

from models.selector import Model
from training_utils.engine_graph import EngineGraph

from config.config_cedar import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Model(name=config.model_name, **config.model_kwargs)
    engine = EngineGraph(model, config)
   
    # Load Model
    # engine.load_state(osp.join(engine.dirpath, config.model_name + "_best.pth"))
    # engine.load_state("/home/jpeng/GraphNets/dump/gcn20191115_071919/gcn_kipf_best.pth")
    engine.load_state("/project/rpp-tanaka-ab/wollip/GraphNets/dump/gcn20191109_093832/gcn_kipf_best.pth")

    # Validation
    engine.validate("validation")
