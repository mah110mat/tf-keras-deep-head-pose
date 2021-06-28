import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
#import dlib
#from imutils import face_utils

import datasets
import utils
import models

PROJECT_DIR = './'

AFLW2000_DATA_DIR = '../data/AFLW2000/'
AFLW2000_MODEL_FILE = PROJECT_DIR + 'models/aflw2000_model.h5'
#AFLW2000_MODEL_FILE = '../tf-keras-deep-head-pose.1/models/aflw2000_model.h5'
AFLW2000_TEST_SAVE_DIR = '../log/aflw2000_test.2/'

BIWI_DATA_DIR = 'E:/ml/data/Biwi/kinect_head_pose_db/hpdb/'
BIWI_MODEL_FILE = PROJECT_DIR + 'model/biwi_model.h5'
BIWI_TEST_SAVE_DIR = 'E:/ml/data/biwi_test/'

BIN_NUM    = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS     = 50

dataset = datasets.AFLW2000(AFLW2000_DATA_DIR, 
                    'filename_list.txt', 
                    batch_size=BATCH_SIZE, 
                    input_size=INPUT_SIZE,
                    preprocess = 'Alexnet'
                    )
net = models.AlexNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net.train(AFLW2000_MODEL_FILE, max_epoches=EPOCHS, load_weight=False)
#net.train(AFLW2000_MODEL_FILE, max_epoches=EPOCHS, load_weight=AFLW2000_MODEL_FILE)

if not os.path.isdir(AFLW2000_TEST_SAVE_DIR):
    os.makedirs(AFLW2000_TEST_SAVE_DIR, exist_ok=True)
net.test(AFLW2000_TEST_SAVE_DIR)
#net.test_save_predicted(AFLW2000_TEST_SAVE_DIR, num_test=100)
