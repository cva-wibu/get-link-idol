import os
import yaml

import torch

RAW_DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(os.getcwd(), 'data', 'processed')
TRAIN_CONFIG_PATH = os.path.join(os.getcwd(), 'config', 'config.yml')

SEARCH_URL = "https://javmodel.com/jav/order_homepages.php?model_cat=6%20Stars%20JAV"


class Config:
    def __init__(self):
        with open(TRAIN_CONFIG_PATH, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.epoch = cfg['epoch']  # example
        self.train_patience = cfg['train_patience']

        self.batch_size = cfg['batch_size']
        self.valid_ratio = cfg['valid_ratio']
        self.test_ratio = cfg['test_ratio']

        self.lr = cfg['lr']

        self.seed = cfg['seed']
        if cfg['device'] == 'gpu':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        if self.device == 'cuda:0':
            self.num_workers = 1
            self.pin_memory = True
        else:
            self.num_workers = 4
            self.pin_memory = False

        # config for mtcnn
        self.image_size = cfg['image_size']
        self.margin = cfg['margin']
        self.min_face_size = cfg['min_face_size']
        self.threshold = cfg['threshold']
        self.factor = cfg['factor']
        self.prewhiten = cfg['prewhiten']
