import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch.nn.functional as Fun
from sklearn.neural_network import MLPRegressor
from torch import nn
from sklearn.preprocessing import MinMaxScaler
#2.定义BP神经网络


def process(xlsx_detail_path,X_path):
    #xlsx_detail_path= '../data/raw/21_22Accumulated_heat.xlsx'
    #xlsx_detail_path = '../data/raw/20_21Accumulated_heat.xlsx'
    if os.path.exists(xlsx_detail_path):
        data = pd.read_excel(xlsx_detail_path,header=None)
    assert data is not None
    data.columns = ['datetime', 'heat']
    data['datetime'] = pd.to_datetime(data['datetime'])
    #补上缺失时间行
    helper = pd.DataFrame({'datetime': pd.date_range(data['datetime'].min(), data['datetime'].max(), freq='1H')})
    data = pd.merge(data, helper, on='datetime', how='outer').sort_values('datetime')
    #去掉分秒，时间只留年月日时
    data = data.reset_index(drop=True)
    data['datetime']=data['datetime'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d-%H'))
    #删除重复时的行，保留最后一个
    data.drop_duplicates(subset='datetime', keep='last', inplace=True)
    #每个小时热量为该时刻热量减去上一时刻热量
    data = data.reset_index(drop=True)
    data['value']=data['heat']-(data['heat'].shift(1))
    data.drop(columns='heat',inplace=True)
    # plt.plot([i for i in range(2880)], data['value'], color="r")
    # plt.show()
    #滤去值非常夸张那种
    for i in range(len(data)):
        if data.iloc[i, 1] <= 0.2 or data.iloc[i, 1] > 1:
            data.loc[i,'value']=np.nan
    data.drop(index=0,inplace=True)#删去11.15日零点数据
    # plt.plot([i for i in range(2879)], data['value'], color="r")
    # plt.show()
    data=data.reset_index(drop=True)
    # #邻近时间点相差超过0.05用nan代替
    for i, Value in enumerate(data['value']):
        if i == 0:
            pass
        else:
            if abs(Value - data['value'][i - 1]) > 0.05:  # Amplitude:  # 限幅
                data.loc[i,'value'] = np.nan
    df_mean = data['value'].mean()
    a = df_mean
    #以年月日时作为特征去进行KNN插值
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['hour'] = data['datetime'].dt.hour  # 以年月日时作为数据的新列，以作为随机森林的标签值
    data.set_index('datetime', inplace=True)
    # impute = KNNImputer(n_neighbors=2)
    # data_filled = impute.fit_transform(data)
    # data_filled=pd.DataFrame(data_filled)
    # data_filled.columns = [ 'value','year','month','day','hour']
    # data.reset_index(inplace=True)
    # data_filled['datatime']=data['datetime']
    #随机森林插值
    known_value = data[data['value'].notnull()]
    unknown_value = data[data['value'].isnull()]
    y = known_value.iloc[:, 0]
    y = np.array(y)
    X = known_value.iloc[:, 1:5]
    X = np.array(X)
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    data.loc[(data['value'].isnull()), 'value'] = rfr.predict(unknown_value.iloc[:, 1:5])
    # plt.plot([i for i in range(2879)],data['value'], color="r")
    # plt.show()
    data = data.reset_index()#此时data为插值之后的dataframe

    #def LimitFilter(Data, Amplitude):
#######################################################

    #限速滤波

    for i, Value in enumerate(data['value']):
        if i == 0 or i==len(data['value'])-1 or i==len(data['value'])-2:
            pass
        else:
            if abs(Value - data['value'][i - 1]) <=0.05:
                pass
            else:

                if abs(data['value'][i + 1] - Value) <= 0.05:
                    data.loc[i, 'value'] = data['value'][i + 1]
                else:
                    data.loc[i, 'value'] = (data['value'][i]+data['value'][i + 1])/2.0
    a = []
    for i, Value in enumerate(data['value']):
        if i == 0 or i==len(data['value'])-1:
            pass
        else:
            if abs(Value - data['value'][i - 1]) >= 0.05:  # Amplitude:  # 限幅
                a .append(i)
                data.loc[i,'value'] = (data['value'][i]+data['value'][i - 1])/2.0

    # plt.plot([i for i in range(19)], data['value'][2860:2880], color="r")
    # plt.show()
    #滑动平均滤波
    # def moving_average(interval, windowsize):
    #     window = np.ones(int(windowsize)) / float(windowsize)
    #     re = np.convolve(interval, window, 'same')
    #     return re
    # data_moving_average = moving_average(data['value'], 10)
    # data_moving_average = data_moving_average.reshape(1, -1).T
    # df = pd.DataFrame(data=data_moving_average[0:, 0:], columns=['value'])# df为滑动平均和随机森林后得到的只有value的dataframe
    # plt.plot([i for i in range(10)], df['value'][0:10], color="r")
    # plt.show()
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
    #X_path = '../data/raw/21-22天气情况_输入.xlsx'#
    #X_path = '../data/raw/20-21天气情况_输入.xlsx'
    if os.path.exists(X_path):
        data_X = pd.read_excel(X_path, header=None)
        data_X = data_X.applymap(ToNumeric)
        data_X = data_X.drop([0,len(data_X)-1])#删去11.15一点钟和3.15日凌晨零点
        data_X.index = range(len(data_X))
        # plt.plot([i for i in range(2878)], data_X.loc[:,0], color="r")
        # fig = plt.figure()
        # ax1 = fig.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot([i for i in range(2880)], data['value'], 'r',label="heat")
        # ax2.plot([i for i in range(2878)], data_X.loc[:,0], 'b',label="temperature")
        # plt.legend(["heat","temperature"])
        # ax1.set_ylabel('heat')
        # ax2.set_ylabel('temperature')
        # plt.show()
        Previous_moment_load = data['value'].drop([len(data) - 1])#删去3.15日23点钟
        data_X['4'] = Previous_moment_load
        # load=data['value'].drop([0])
        # load= load.reset_index(drop=True)
        # Previous_24_load=load.drop(index=data_X[len(load) - 24:len(load)].index)
        # data_X=data_X.drop(index=data_X[0:24].index)
        # data_X = data_X.reset_index(drop=True)
        # data_X['5'] = Previous_24_load
        data_X = data_X.reset_index(drop=True)
    data_Y = pd.DataFrame(data['value'].drop([0]))#删去11.14日一点钟
    #data_Y = pd.DataFrame(data['value'].drop(index=data['value'][0:25].index))
    data_Y = data_Y.reset_index(drop=True)
    # X_train=data_X
    # y_train=data_Y
    # X_train.to_excel('../data./processed/20X_train' + '.xlsx', index=False)
    # y_train.to_excel('../data./processed/20y_train' + '.xlsx', index=False)
    ####删去天气缺项的行
    Data =data_X
    Data['value']=data_Y['value']
    Data = Data.dropna(axis=0, how='any')
    Data=Data.reset_index(drop=True)
    Data_X=Data.iloc[:,:-1]
    Data_Y=Data.loc[:,'value']
    return Data_X, Data_Y
def data_split(data_X, data_Y):
    test_ratio = 0.2
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    X_test.to_excel('../data./processed/X_test' + '.xlsx', index=False,header=None)
    y_test.to_excel('../data./processed/y_test' + '.xlsx', index=False,header=None)
    X_train.to_excel('../data./processed/X_train' + '.xlsx', index=False,header=None)
    y_train.to_excel('../data./processed/y_train' + '.xlsx', index=False,header=None)
    return X_train, X_test, y_train, y_test
if __name__ == '__main__':
    data_Xtrain, data_Ytrain=process('../data/raw/20_21Accumulated_heat.xlsx','../data/raw/20-21天气情况_输入.xlsx')
    data_Xtest,data_Ytest=process('../data/raw/21_22Accumulated_heat.xlsx', '../data/raw/21-22天气情况_输入.xlsx')
    X_train, X_test, y_train, y_test = data_split(data_Xtrain, data_Ytrain)
    model = MLPRegressor(hidden_layer_sizes=(10,),  activation='logistic', solver='lbfgs',  batch_size=5000,
    learning_rate='constant', learning_rate_init=0.001, max_iter=5000, shuffle=True,#lbfgs,激活函数tanh，logistic
    random_state=200,  warm_start=True,
    early_stopping=True)  # BP神经网络回归模型
    model.fit(X_train,y_train)  # 训练模型
    pre = model.predict(X_test) # 模型预测
    pre=pd.Series(pre).T
    y_test=y_test.reset_index(drop=True)
    MRE=[]
    Error=[]
    for i in range(len(pre)):
        error=(pre[i]-y_test[i])
        E=error/y_test[i]
        Error.append(error)
        MRE.append(E)

    plt.plot([i for i in range(len(pre))], pre, color="r")
    plt.plot([i for i in range(len(y_test))], y_test, color="b")
    plt.show()
    plt.plot([i for i in range(len(y_test))], MRE, color="b")
    plt.show()
    plt.plot([i for i in range(len(y_test))], Error, color="g")
    plt.show()
    for i,value in enumerate(MRE):
        if value>0.08:
            print(i)

    # # 模型评价

