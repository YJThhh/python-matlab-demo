import matlab.engine
import matlab
import os
import logging
import argparse
from logging import handlers
import numpy as np



def CalMATLAB(eng,a,b):
    res = eng.add(a,b)

    return res
if __name__ == '__main__':
    # Step 1: init config and Logger and start matlab engine
    MATLAB_eng = matlab.engine.start_matlab()
    MATLAB_eng.addpath(MATLAB_eng.genpath(MATLAB_eng.fullfile(os.getcwd(),  'matlab')))
    a=2
    b=3
    print(CalMATLAB(MATLAB_eng,a,b))
