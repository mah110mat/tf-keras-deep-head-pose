import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import argparse

import datasets
import utils
from utils import Config
import models

parser = argparse.ArgumentParser(description='train deep head pose')
parser.add_argument('yaml', help='yaml config file(ex. yaml/Mobilenetv2.yaml')
args = parser.parse_args() 

cfg = Config(args.yaml)

dataset = datasets.AFLW2000(cfg.DATA_DIR, 'filename_list.txt', batch_size=cfg.BATCH_SIZE, input_size=cfg.INPUT_SIZE, preprocess = cfg.BACKBONE)
valid_dataset = datasets.AFLW2000(cfg.DATA_DIR, 'filename_list.txt', batch_size=cfg.BATCH_SIZE, input_size=cfg.INPUT_SIZE, preprocess = cfg.BACKBONE)

net = models.Mobilenetv2(dataset, cfg.BIN_NUM, batch_size=cfg.BATCH_SIZE, input_size=cfg.INPUT_SIZE, save_dir= cfg.TEST_SAVE_DIR, valid_dataset=valid_dataset)

net.train(cfg.MODEL_FILE, max_epoches=cfg.EPOCHS, load_weight=False)
#net.train(MODEL_FILE.replace('.h5','_best.h5'), max_epoches=EPOCHS, load_weight=True)

net.test()
