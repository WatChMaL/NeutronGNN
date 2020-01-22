import torch
import numpy as np

from models.selector import Model
from training_utils.engine_graph import EngineGraph
from training_utils.find_top_models import find_top_models 

from config.config_cedar import config

import os.path as osp

if __name__ == '__main__':
    ## Cheby_batch_topk
    #print("cheby_batch_topk")
    #config.model_name = "cheby_batch_topk"
    #config.model_kwargs = {"w1":128, "w2":128, "w3":128, "k":2}
    #config.dump_path = "/project_dir/dump/" + config.model_name

    #model = Model(name=config.model_name, **config.model_kwargs)
    #engine = EngineGraph(model, config)
    #
    #engine.dirpath = "/project_dir/dump/cheby_batch_topk20191202_194531"

    #models = find_top_models(engine.dirpath, 5)
    #for model in models:
    #    print(model)
    #    engine.load_state(osp.join(engine.dirpath, model))
    #    engine.validate("validation", name=osp.splitext(model)[0])

    ## gcn_batch_topk
    #print("gcn_batch_topk")
    #config.model_name = "gcn_batch_topk"
    #config.model_kwargs = {"w1":128, "w2":128, "w3":128}
    #config.batch_size = 128
    #config.dump_path = "/project_dir/dump/" + config.model_name

    #model = Model(name=config.model_name, **config.model_kwargs)
    #engine = EngineGraph(model, config)
    #
    #engine.dirpath = "/project_dir/dump/gcn_batch_topk20191202_045850"
    #
    #models = find_top_models(engine.dirpath, 5)
    #for model in models:
    #    print(model)
    #    engine.load_state(osp.join(engine.dirpath, model))
    #    engine.validate("validation", name=osp.splitext(model)[0])

    ## gcn_batch_reg_
    #print("gcn_batch_reg")
    #config.model_name = "gcn_batch_reg"
    #config.model_kwargs = {"w1":64, "w2":64, "w3":64}
    #config.batch_size = 64
    #config.dump_path = "/project_dir/dump/" + config.model_name

    #model = Model(name=config.model_name, **config.model_kwargs)
    #engine = EngineGraph(model, config)
    #
    #engine.dirpath = "/project_dir/dump/gcn_batch_reg20191126_035057"
    #
    #models = find_top_models(engine.dirpath, 5)
    #for model in models:
    #    print(model)
    #    engine.load_state(osp.join(engine.dirpath, model))
    #    engine.validate("validation", name=osp.splitext(model)[0])

    # gcn_relu
    print("gcn_relu")
    config.model_name = "gcn_relu"
    config.model_kwargs = {"w1":128, "w2":128, "w3":128}
    config.batch_size = 64 
    config.dump_path = "/project_dir/dump/" + config.model_name
    config.validate_batch_size = 64
    config.validate_dump_interval = 2048

    model = Model(name=config.model_name, **config.model_kwargs)
    engine = EngineGraph(model, config)
    
    engine.dirpath = "/project_dir/dump/gcn_relu20191125_090046"
    
    models = find_top_models(engine.dirpath, 5)
    for model in models:
        print(model)
        engine.load_state(osp.join(engine.dirpath, model))
        engine.validate("validation", name=osp.splitext(model)[0])

