import torch
import numpy as np

from models.selector import Model
from training_utils.engine_graph import EngineGraph
from training_utils.find_top_models import find_top_models

from config.config_cedar import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Model(name=config.model_name, **config.model_kwargs)
    engine = EngineGraph(model, config)

    # Training
    engine.train()

    # Save network
    engine.save_state()

    #Validation
    models = find_top_models(engine.dirpath, 5)
    for model in models:
        engine.load_state(osp.join(engine.dirpath, model))
        engine.validate("validation", name=osp.splitext(model)[0])
