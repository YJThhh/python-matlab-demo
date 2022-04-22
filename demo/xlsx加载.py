import matlab.engine
import matlab
import os
import logging
import argparse
from logging import handlers
import numpy as np
import pandas as pd


def CalMATLAB(eng,data):
    res = eng.pd_process(data)

    return res


def  XlsxLoad(path):
    pass
if __name__ == '__main__':
    # Step 1: init config and Logger and start matlab engine
    # MATLAB_eng = matlab.engine.start_matlab()
    # MATLAB_eng.addpath(MATLAB_eng.genpath(MATLAB_eng.fullfile(os.getcwd(),  'matlab')))
    # Step 2: load xlsx file
    xlsx_detail_path= '../data/interpolate_test.xlsx'
    if os.path.exists(xlsx_detail_path):
        data = pd.read_excel(xlsx_detail_path)

    assert data is not None
    #由于这个xlsx没有列名称，索引不是很方便，先更改列名称
    data.columns = ['datetime', 'value']

    #第一列转datetime
    # data.iloc[:,0]=pd.to_datetime(data.iloc[:,0]) #按序号索引列，有名称以后就可以不需要这样做了
    data['datetime']= pd.to_datetime(data['datetime'])
    #新建一个helper datafream来辅助插值
    helper = pd.DataFrame({'datetime': pd.date_range(data['datetime'].min(), data['datetime'].max(),freq='1H')})
    print("helper")
    print(helper)
    data = pd.merge(data, helper, on='datetime', how='outer').sort_values('datetime')
    #线性插值
    data['value'] = data['value'].interpolate(method='linear')
    #打印插值结果
    print("data")
    print(data)
    # cloumn_0_list=data.iloc[:,0].tolist()
    # cloumn_0_list=data.iloc[:,1].tolist()

    # X1 = res_detail.values.tolist()

    # print(CalMATLAB(MATLAB_eng,X1))
    pass