from models.rnn import *
from models.pvcnn_dct import *


named_models = {
                'PVCNN2_DCT_CONT': PVCNN2_DCT_CONT,
                'GRU_POSE':GRU_POSE,
                }

def get_model(cfg):
    model_name = cfg.model_name
    return named_models[model_name](cfg.model_specs)