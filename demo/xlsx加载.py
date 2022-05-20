from typing import Type

import matlab.engine
import matlab
import os
import logging
import argparse
from logging import handlers
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
from datetime import datetime
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
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

######################################################################################
    #行列名和datetime的处理
    #由于这个xlsx没有列名称，索引不是很方便，先更改列名称
    # data.columns = ['datetime', 'value']
    #第一列转datetime
    # data.iloc[:,0]=pd.to_datetime(data.iloc[:,0]) #按序号索引列，有名称以后就可以不需要这样做了
    data['datetime']= pd.to_datetime(data['datetime'])
######################################################################################
    #去除异常值（滤波）
    # #去除异常值（太大or太小，时间重复）
    #data['value'] =data['value'].map(lambda x: x*1000)
    index = []
    for i in range(len(data)):
        if data.iloc[i, 1]<=0.1 or data.iloc[i, 1]>1:
            index.append(i)
    data=data.drop(index=index)
    ###############################################################################
    #限幅滤波
    def LimitFilter(Data, Amplitude):
        Data = Data['value']
        ReturnData = [Data[0]]
        for Value in Data[1:]:
            # print(abs(Value - ReturnData[-1]))
            if abs(Value - ReturnData[-1]) < Amplitude:  # 限幅
                ReturnData.append(Value)
            else:
                ReturnData.append(ReturnData[-1])

        return ReturnData
    data['value'] = LimitFilter(data, 0.05)
    #
    # plt.title("limit")
    #
    plt.plot([i for i in range(549)], data['value'][2000:2886], color="r", label="source data")
    # NextData = LimitFilter(data, 0.05)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="b", label="A=0.05")
    # NextData = LimitFilter(data, 0.1)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="g", label="A=0.1")
    # NextData = LimitFilter(data, 0.2)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="y", label="A=0.2")
    # plt.legend()
    # plt.show()
#######################################################################################
    # # #算数平均值滤波
    # def ArithmeticAverageFilter(Data, N):
    #     ReturnData = []
    #     Accumulate = 0  # 和值
    #     for index, Value in enumerate(Data['value']):
    #         Accumulate += Value
    #         if (index + 1) % N == 0:
    #             Median = Accumulate / N
    #             ReturnData += [Median] * N
    #             Accumulate = 0
    #     # 处理剩余的数据
    #     if len(Data) % N != 0:
    #         Median = Accumulate / (len(Data) % N)
    #         ReturnData += [Median] * (len(Data) % N)
    #     return ReturnData
    #
    #
    # plt.title("ArithmeticAverageFilter")
    # plt.plot([i for i in range(886)], data['value'][2000:2886], color="r",label="source data")
    # NextData = ArithmeticAverageFilter(data, 3)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="b",label="N=3")
    # NextData = ArithmeticAverageFilter(data, 12)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="g",label="N=12")
    # NextData = ArithmeticAverageFilter(data, 24)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="y",label="N=24")
    # plt.legend()
    # plt.show()
#################################################################

    #中位值滤波
    # def MedianFilter(Data, N):
    #     ReturnData = []
    #     StageList = []
    #     for index, Value in enumerate(Data['value']):
    #         StageList.append(Value)
    #         if (index + 1) % N == 0:
    #             StageList.sort()  # 排序
    #
    #             # 取中值
    #             if N % 2 != 0:
    #                 ReturnData += [StageList[int((N + 1) / 2) - 1]] * N
    #                 StageList.clear()
    #             else:
    #                 ReturnData += [(StageList[int(N / 2) - 1] + StageList[int(N / 2)]) / 2] * N
    #                 StageList.clear()
    #
    #     # 处理剩余的数据
    #     Residue = len(StageList)
    #
    #     if Residue != 0:
    #         StageList.sort()
    #         if Residue % 2 != 0:
    #             ReturnData += [StageList[(Residue + 1) / 2 - 1]] * Residue
    #         else:
    #             ReturnData += [(StageList[int(Residue / 2) - 1] + StageList[int(Residue / 2)]) / 2] * Residue
    #
    #     return ReturnData
    #
    # plt.title("Median")
    #
    # plt.plot([i for i in range(886)], data['value'][2000:2886], color="r",label="source data")
    # NextData = MedianFilter(data,3)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="b",label="N=3")
    # NextData = MedianFilter(data,6)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="g",label="N=6")
    # NextData = MedianFilter(data,12)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="y",label="N=12")
    # plt.legend()
    # plt.show()
###############################################################################
    #限幅平均滤波
    # def LimitingAverageFilter(Data, Amplitude, N):
    #     Data=Data['value']
    #     ReturnData = [Data[0]]
    #     StageList = [Data[0]]
    #     for Value in Data[1:]:
    #
    #         # 限幅处理
    #         if abs(Value - StageList[-1]) < Amplitude:
    #             StageList.append(Value)
    #         else:
    #             StageList.append(StageList[-1])
    #
    #         # 保持队列数量不超过N
    #         if len(StageList) > N:
    #             StageList.pop(0)
    #
    #         Number = len(StageList)
    #
    #         ReturnData.append(sum(StageList) / Number)
    #
    #     return ReturnData
    # data['value'] = LimitingAverageFilter(data, 0.05,3)
    # plt.title("limitaverage")
    #
    # plt.plot([i for i in range(886)], data['value'][2000:2886], color="r",label="source data")
    # NextData = LimitingAverageFilter(data, 0.05, 3)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="b",label="A=0.05 N=3")
    # NextData = LimitingAverageFilter(data, 0.05, 12)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="g",label="A=0.05 N=12")
    # NextData = LimitingAverageFilter(data, 0.05, 24)
    #
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="y",label="A=0.05 N=24")
    # plt.legend()
    # plt.show()
################################################################################
    #加权递推平均滤波法
    # def WeightedRecursiveAveragingFalter(Data, N, Weight=None):
    #     if Weight == None:
    #         Weight = [i for i in range(1, N + 1)]  # [1, 2, 3, 4, 5······]
    #
    #     WeightSum = sum(Weight)
    #     Weight = [i / WeightSum for i in Weight]  # 归一化
    #
    #     ReturnData = []
    #     StageList = []
    #     for Value in Data['value']:
    #
    #         # 入队与出队
    #         StageList.append(Value)
    #         if len(StageList) > N:
    #             StageList.pop(0)
    #
    #         if len(StageList) < N:
    #             WNum = 0
    #             VNum = 0
    #             for W, V in zip(Weight[-len(StageList):], StageList):
    #                 WNum += W
    #                 VNum += V
    #             ReturnData.append(VNum / WNum)
    #
    #         else:
    #             SRList = [W * V for W, V in zip(Weight, StageList)]
    #             ReturnData.append(sum(SRList))
    #
    #     return ReturnData
    #
    #
    # plt.title("WeightedRecursiveAveragingFalter")
    # plt.plot([i for i in range(886)], data['value'][2000:2886], color="r",label="source data")
    #
    # NextData = WeightedRecursiveAveragingFalter(data, 3)
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="b",label="N=3")
    #
    # NextData = WeightedRecursiveAveragingFalter(data, 6)
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="y",label="N=6")
    #
    # NextData = WeightedRecursiveAveragingFalter(data, 12)
    # plt.plot([i for i in range(886)], NextData[2000:2886], color="g", label="N=12")
    # plt.legend()
    # plt.show()
    #########################################################################################
    # 补上缺失的时间点
    # 新建一个helper datafream来辅助插值
    helper = pd.DataFrame({'datetime': pd.date_range(data['datetime'].min(), data['datetime'].max(), freq='1H')})
    data = pd.merge(data, helper, on='datetime', how='outer').sort_values('datetime')
    # data.set_index('datetime', inplace=True)
    # print(data.loc[index_date])
    ########################################################################################
    #时间只精确到时，再删除重复行
    data['datetime']=data['datetime'].apply(lambda x:datetime.strftime(x,'%Y-%m-%d-%H'))#时间保留到时
    data.drop_duplicates(subset='datetime',keep='last',inplace=True)#删除重复行
    # data['datetime'] = pd.to_datetime(data['datetime'])
    #data['datetime'] = data['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d-%H'))
    # data.set_index('datetime', inplace=True)
    ##########################################################################################
    # 随机森林插值
    # data['value'].interpolate(method='time', order=5, inplace=True)
    # data.to_excel('../data./processed/rawtime_Y' + '.xlsx', index=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year']=data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour#以年月日时作为数据的新列，以作为随机森林的标签值
    # data=data.drop(columns='datetime')
    data.set_index('datetime', inplace=True)

    known_value = data[data['value'].notnull()]
    unknown_value = data[data['value'].isnull()]
    y = known_value.iloc[:, 0]
    y = np.array(y)

    X = known_value.iloc[:, 1:5]
    X = np.array(X)

    from sklearn.ensemble import RandomForestRegressor

    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    data.loc[(data['value'].isnull()), 'value'] = rfr.predict(unknown_value.iloc[:, 1:5])
    #data['value'].to_excel('../data./processed/sjsl_Y' + '.xlsx', index=False)
    # miss_col=list(data.index[np.where(np.isnan(data))[0]])
    # estimate_list = list(data.index[np.isnan(data['value']) == False])
    # data = data.T
    # def set_missing(df, estimate_list, miss_col):
    #     """df要处理的数据帧，estimate_list用来估计缺失值的字段列表,miss_col缺失字段名称;会直接在原来的数据帧上修改"""
    #     col_list = estimate_list
    #     #col_list.append( miss_col)
    #     col_list.extend(miss_col)
    #     process_df = df.loc[:, col_list]
    #     class_le = LabelEncoder()
    #     for i in col_list[:-1]:
    #         process_df.loc[:, i] = class_le.fit_transform(process_df.loc[:, i].values)
    #     # 分成已知该特征和未知该特征两部分
    #     known = process_df[process_df[miss_col].notnull()].values
    #     known[:, -1] = class_le.fit_transform(known[:, -1])
    #     unknown = process_df[process_df[miss_col].isnull()].values
    #     # X为特征属性值
    #     X = known[:, :-1]
    #     # y为结果标签值
    #     y = known[:, -1]
    #     # fit到RandomForestRegressor之中
    #     rfr = ensemble.RandomForestRegressor(random_state=1, n_estimators=200, max_depth=4, n_jobs=-1)
    #     rfr.fit(X, y)
    #     # 用得到的模型进行未知特征值预测
    #     predicted = rfr.predict(unknown[:, :-1]).round(0).astype(int)
    #     predicted = class_le.inverse_transform(predicted)
    #     #     print(predicted)
    #     # 用得到的预测结果填补原缺失数据
    #     df.loc[(df[miss_col].isnull()), miss_col] = predicted
    #     return df
    # data=set_missing(data, estimate_list, miss_col)

    # data['value'].interpolate(method='time', order=5,inplace=True)
    # index_date=pd.Timestamp(year=2020, month=12, day=9, hour=9)
    # print(data.loc[index_date])
############################################################################################
    #基于Numpy.convolve实现滑动平均滤波
    #data=data.drop(columns=['year','month','day','hour'])
    data = data.reset_index()
    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'same')
        return re
    data_moving_average= moving_average(data['value'], 5)
    data_moving_average=data_moving_average.reshape(1,-1).T
    df = pd.DataFrame(data=data_moving_average[0:, 0:],columns=['value'])#df为滑动平均和随机森林后得到的只有value的dataframe
    # df['datetime']=data['datetime']
    # df['value'].to_excel('../data./processed/sjsl_MA10_Y' + '.xlsx', index=False)
    # plt.plot(data['datetime'][500:600], data['value'][500:600], 'k')
    # plt.plot(df['datetime'][500:600], df['value'][500:600], 'b')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.show()
############################################################################################
#3-sigma
    # data=data.drop(columns=['year','month','day','hour'])
    # data =data.reset_index()
    # n = 3  # n*sigma
    # data_y = data['value']
    # data_x = data['datetime']
    # ymean = np.mean(data_y)
    # ystd = np.std(data_y)
    # threshold1 = ymean - n * ystd
    # threshold2 = ymean + n * ystd
    # outlier = []  # 将异常值保存
    # outlier_x = []
    # for i in range(0, len(data_y)):
    #     if (data_y[i] < threshold1) | (data_y[i] > threshold2):
    #         outlier.append(data_y[i])
    #         outlier_x.append(data_x[i])
    #     else:
    #         continue
    #
    # print('\n异常数据如下：\n')
    # print(outlier)
    # print(outlier_x)
    #
    # plt.plot(data_x, data_y)
    # plt.plot(outlier_x, outlier, 'ro')
    # for j in range(len(outlier)):
    #     plt.annotate(outlier[j], xy=(outlier_x[j], outlier[j]), xytext=(outlier_x[j], outlier[j]))
    # plt.show()
#####################################################################################
#天气情况的数据格式处理,文本转数值
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

####################################################################################
    X_path = '../data/raw/20-21天气情况_输入.xlsx'
    if os.path.exists(X_path):
        data_X = pd.read_excel(X_path,header=None)
        data_X=data_X.applymap(ToNumeric)
        data_X=data_X.drop([0])
        data_X.index = range(len(data_X))
        Previous_moment_load=data['value'].drop([len(data)-1])
        data_X['4']=Previous_moment_load
        time_variable = data[['month', 'day', 'hour']][1:]#把月日时也作为特征
        time_variable = time_variable.reset_index()
        data_X['5'],data_X['6'],data_X['7']=time_variable['month'],time_variable['day'],time_variable['hour']
    #data_X.to_excel('../data./processed/all_X' + '.xlsx', index=False)
    # Y_path = '../data/processed/sjsl_MA10_Y.xlsx'
    # if os.path.exists(Y_path):
    #     data_Y = pd.read_excel(Y_path,header=None)
    #     data_Y=data_Y.applymap(ToNumeric)
    data_Y=df['value'].drop([0])
    data_Y=data_Y.reset_index(drop=True)
    test_ratio=0.1
    #data_Y.to_excel('../data./processed/all_Y' + '.xlsx', index=False)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    X_train.to_excel('../data./processed/sjsl_time_X_train'+ '.xlsx', index=False)
    X_test.to_excel('../data./processed/sjsl_time_X_test' + '.xlsx', index=False)
    y_train.to_excel('../data./processed/sjsl_time_y_train' + '.xlsx', index=False)
    y_test.to_excel('../data./processed/sjsl_time_y_test' + '.xlsx', index=False)
####################################################################################
    # cloumn_0_list=data.iloc[:,0].tolist()
    # cloumn_0_list=data.iloc[:,1].tolist()

    # X1 = res_detail.values.tolist()

    # print(CalMATLAB(MATLAB_eng,X1))
    pass