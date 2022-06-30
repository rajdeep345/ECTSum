import os
from os.path import exists, join
import json

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = 'processed_data/ect_data_train.jsonl'
        paths['val']   = 'processed_data/ect_data_val.jsonl'
    else:
        paths['test']  = 'processed_data/ect_data_test.jsonl'
    return paths

def get_result_path(save_path, cur_model):
    model_name = save_path.split('/')[-1]
    result_path = join(f'results/{model_name}')
    
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'pred')
    ref_path = join(model_path, 'gold')
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path
