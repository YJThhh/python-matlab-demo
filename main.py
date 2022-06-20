import os
import logging


import pandas as pd
from PIL import Image

import matlab.engine
import torch
import argparse
from logging import handlers
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def CalMATLAB(XTrainPath, XTestPath, YTrainPath, YTestPath):
    MATLAB_eng = matlab.engine.start_matlab()  #启动Matlab
    MATLAB_eng.addpath(MATLAB_eng.genpath(MATLAB_eng.fullfile(MatlabProjectRoot)))  #把Matlab的工作目录设置一下
    MATLAB_eng.NetworkTrainTest(XTrainPath, XTestPath, YTrainPath, YTestPath)  #调用工作目录下面的某个function
    print("Matlab code excute finished")
    import time

    while(True):
        time.sleep(5)
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="TrainTest")
    parser.add_argument('YAML', type=str, help='configuration file')
    args = parser.parse_args()

    conf = dict()
    with open(args.YAML, 'r', encoding='UTF-8') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    MatlabProjectRoot=conf['Path']['MatlabProjectRoot']
    XTrainPath = conf['Path']['XTrainPath']
    XTestPath = conf['Path']['XTestPath']
    YTrainPath = conf['Path']['YTrainPath']
    YTestPath = conf['Path']['YTestPath']

    # Step 1: init config and Logger and start matlab engine

    CalMATLAB(XTrainPath, XTestPath, YTrainPath, YTestPath)


