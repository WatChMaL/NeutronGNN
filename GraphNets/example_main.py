import torch
import numpy as np

from models.selector import Model
from training_utils.engine_graph import EngineGraph

from config.config_example import config

import os.path as osp

if __name__ == '__main__':
    # Initialization
    model = Model(name=config.model_name, **config.model_kwargs)
    engine = EngineGraph(model, config)

    # Training
    engine.train()

    # Validation
    engine.validate("validation", name="before_load")

    # Save network
    engine.save_state()
    print(model.conv1.weight)

    # Reset network
    model.conv1.weight = torch.nn.Parameter(torch.Tensor(np.random.random(model.conv1.weight.shape)))
    print(model.conv1.weight)

    # Load network
    engine.load_state(osp.join(engine.dirpath, config.model_name + "_latest.pth"))
    print(model.conv1.weight)

    # Check load successful
    engine.validate("validation", name="after_load")
