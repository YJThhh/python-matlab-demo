from typing import Type

import matlab.engine
import matlab
import os
import logging
import argparse
from logging import handlers
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
def CalMATLAB(eng,data):
    res = eng.pd_process(data)

    return res



if __name__ == '__main__':
    # Step 1: init config and Logger and start matlab engine
    # MATLAB_eng = matlab.engine.start_matlab()
    # MATLAB_eng.addpath(MATLAB_eng.genpath(MATLAB_eng.fullfile(os.getcwd(),  'matlab')))
    # Step 2: load xlsx file
    xlsx_detail_path= '../data/raw/20-21带时间热量.xlsx'
    if os.path.exists(xlsx_detail_path):
        data = pd.read_excel(xlsx_detail_path)

    assert data is not None


    #由于这个xlsx没有列名称，索引不是很方便，先更改列名称
    data.columns = ['datetime', 'value']

    #第一列转datetime
    # data.iloc[:,0]=pd.to_datetime(data.iloc[:,0]) #按序号索引列，有名称以后就可以不需要这样做了
    data['datetime']= pd.to_datetime(data['datetime'])

    # #去除异常值（太大or太小，时间重复）
    # #data['value'] =data['value'].map(lambda x: x*1000)
    index = []
    for i in range(len(data)):
        if data.iloc[i, 1]<=0.1 or data.iloc[i, 1]>1:
            index.append(i)
    data=data.drop(index=index)


    #新建一个helper datafream来辅助插值
    helper = pd.DataFrame({'datetime': pd.date_range(data['datetime'].min(), data['datetime'].max(),freq='1H')})

    data = pd.merge(data, helper, on='datetime', how='outer').sort_values('datetime')



    data.set_index('datetime', inplace = True)
    # print(data.loc[index_date])
    #线性插值
    data['value'].interpolate(method='time', order=5,inplace=True)

    index_date=pd.Timestamp(year=2020, month=12, day=9, hour=9)
    print(data.loc[index_date])
#####################

    def ToNumeric(inputString):
        inputString=str(inputString)
        inputString=inputString.replace('(','').replace(')','')

        try:

            return float(inputString)
        except ValueError:
            pass

        try:
            import unicodedata
            return unicodedata.numeric(inputString)
        except (TypeError, ValueError):
            pass



    pass

    Y_path = '../data/raw/20-21天气情况_输入.xlsx'
    if os.path.exists(Y_path):
        data_Y = pd.read_excel(Y_path,header=None)
        data_Y=data_Y.applymap(ToNumeric)

    X_path = '../data/raw/无时间全部热量_输入.xlsx'
    if os.path.exists(X_path):
        data_X = pd.read_excel(X_path,header=None)
        data_X=data_X.applymap(ToNumeric)
    test_ratio=0.2
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    X_train.to_excel('X_train'+ '.xlsx', index=False)
    X_test.to_excel('X_test' + '.xlsx', index=False)
    y_train.to_excel('y_train' + '.xlsx', index=False)
    y_test.to_excel('y_test' + '.xlsx', index=False)
    # X = df.iloc[:, :-4]
    # X_train, X_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=test_ratio,
    #                                                     random_state=42)
    # cloumn_0_list=data.iloc[:,0].tolist()
    # cloumn_0_list=data.iloc[:,1].tolist()

    # X1 = res_detail.values.tolist()

    # print(CalMATLAB(MATLAB_eng,X1))
    pass