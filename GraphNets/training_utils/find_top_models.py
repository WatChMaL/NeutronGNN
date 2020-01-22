import os
import os.path as osp

def find_top_models(dump_path, n):
    assert n > 0
    models = {}
    files = os.listdir(dump_path)
    for f in files:
        if osp.isfile(osp.join(dump_path, f)) and f.endswith('.pth') and 'best' in f:
            score_pth = f.split('_')[-1].split('.')
            if len(score_pth) == 3:
                score = float(score_pth[0] + '.' + score_pth[1])
            elif len(score_pth) == 2:
                score = float(score_pth[0])
            else:
                print("ERRORRRR")
            models[f] = score
    while len(models) > n:
        max_score = 0
        max_f = ""
        for f in models:
            if models[f] > max_score:
                max_score = models[f]
                max_f = f
        del models[max_f]
    return list(sorted(models, key=models.get))

if __name__ == '__main__':
    #model = Model(name=config.model_name, **config.model_kwargs)
    #engine = EngineGraph(model, config)
    models = find_top_models("/project/rpp-tanaka-ab/wollip/GraphNets/dump/cheby_batch_topk20191202_194531", 5)
    print(models)
