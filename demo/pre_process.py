import os
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    xlsx_detail_path= '../data/raw/21_22Accumulated_heat.xlsx'
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
    #滤去值非常夸张那种
    for i in range(len(data)):
        if data.iloc[i, 1] <= 0.1 or data.iloc[i, 1] > 1:
            data.loc[i,'value']=np.nan
    data.drop(index=0,inplace=True)
    data=data.reset_index(drop=True)
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
    # known_value = data[data['value'].notnull()]
    # unknown_value = data[data['value'].isnull()]
    # y = known_value.iloc[:, 0]
    # y = np.array(y)
    # X = known_value.iloc[:, 1:5]
    # X = np.array(X)
    # rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    # rfr.fit(X, y)
    # data.loc[(data['value'].isnull()), 'value'] = rfr.predict(unknown_value.iloc[:, 1:5])
    # plt.plot([i for i in range(81)],data['value'][2800:2881], color="r")
    # plt.show()
    data = data.reset_index()#此时data为插值之后的dataframe
    # def moving_average(interval, windowsize):
    #     window = np.ones(int(windowsize)) / float(windowsize)
    #     re = np.convolve(interval, window, 'same')
    #     return re
    # data_moving_average = moving_average(data['value'], 5)
    # data_moving_average = data_moving_average.reshape(1, -1).T
    # df = pd.DataFrame(data=data_moving_average[0:, 0:], columns=['value'])  # df为滑动平均和随机森林后得到的只有value的dataframe
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
    X_path = '../data/raw/20-21天气情况_输入.xlsx'
    if os.path.exists(X_path):
        data_X = pd.read_excel(X_path, header=None)
        data_X = data_X.applymap(ToNumeric)
        data_X = data_X.drop([0])
        data_X.index = range(len(data_X))
        Previous_moment_load = data['value'].drop([len(data) - 1])
        data_X['4'] = Previous_moment_load
        time_variable = data[['year','month', 'day', 'hour']][1:]  # 把月日时也作为特征
        time_variable = time_variable.reset_index()
        data_X['5'], data_X['6'], data_X['7'] ,data_X['8'] =time_variable['year'], time_variable['month'], time_variable['day'], time_variable['hour']
        data_value=data['value'].drop([0])
        data_value=data_value.reset_index(drop=True)
        data_X['9']=data_value

    # data_Y = pd.DataFrame(data_value)#(data['value'].drop([0]))
    # data_Y = data_Y.reset_index(drop=True)
    # data_Y['Previous_moment_load']=Previous_moment_load
    data_X = data_X.dropna(axis=0,how='any')

    data_Y = pd.DataFrame(data_X['9'])
    data_X=data_X.drop(columns='9')

    test_ratio = 0.2
    # data_Y.to_excel('../data./processed/all_Y' + '.xlsx', index=False)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    #X_train.to_excel('../data./processed/20X_train' + '.xlsx', index=False)
    X_test.to_excel('../data./processed/21X_test' + '.xlsx', index=False)
    #y_train.to_excel('../data./processed/20y_train' + '.xlsx', index=False)
    y_test.to_excel('../data./processed/21y_test' + '.xlsx', index=False)

