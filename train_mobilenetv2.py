import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio

import datasets
import utils
import models

PROJECT_DIR = "./"

AFLW2000_DATA_DIR = '../data/AFLW2000/'
AFLW2000_MODEL_FILE = PROJECT_DIR + 'models/aflw2000_model_mobilenetv2.h5'
AFLW2000_TEST_SAVE_DIR = '../log/aflw2000_test.2.mobilenetv2/'

BIN_NUM    = 66
INPUT_SIZE = 96
BATCH_SIZE = 16
EPOCHS     = 200

if not os.path.isdir(AFLW2000_TEST_SAVE_DIR):
    os.makedirs(AFLW2000_TEST_SAVE_DIR, exist_ok=True)

dataset = datasets.AFLW2000(AFLW2000_DATA_DIR, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, preprocess = 'Mobilenetv2')
valid_dataset = datasets.AFLW2000(AFLW2000_DATA_DIR, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, preprocess = 'Mobilenetv2')

net = models.Mobilenetv2(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, save_dir= AFLW2000_TEST_SAVE_DIR, valid_dataset=valid_dataset)

net.train(AFLW2000_MODEL_FILE, max_epoches=EPOCHS, load_weight=False)
#net.train(AFLW2000_MODEL_FILE.replace('.h5','_best.h5'), max_epoches=EPOCHS, load_weight=True)

net.test()
