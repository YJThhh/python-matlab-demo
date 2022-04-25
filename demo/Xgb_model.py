import csv
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor

from codes.models.Xgb.myXgb import xgb_predict_plot, xgb_importance
from codes.models.base_model import BaseModel
from codes.utils.util import feature_names_get, csv_preprocess, df_add_col, festival

from zhdate import ZhDate as lunar_date

class XgbModel(BaseModel):
    def __init__(self, opt):
        super(XgbModel, self).__init__(opt)

    def save(self):
        with open(os.path.join(self.opt['path']['experiments_root'], 'xgb.pkl'), 'wb') as fid:
            pickle.dump(self.xgb_model, fid)

    def load(self):
        pkl_file = open(os.path.join(self.opt['path']['experiments_root'], 'xgb.pkl'), 'rb')
        self.xgb_model = pickle.load(pkl_file)

    def prepare_data(self):
        feature_names = feature_names_get(self.opt)

        # prepare train and val datasets
        self.df_15month = csv_preprocess(self.opt, self.opt['datasets']['train'])
        self.df_15month = self.df_15month .resample(rule='1H').mean()
        self.df_15month=festival(self.df_15month)
        result = seasonal_decompose(self.df_15month.index.month, model='additive', period=4 * 24 * 7)
        self.df_15month['seasonal'] = result.seasonal

        #节假日
        # self.df_15month.loc[:, 'festival'] = 0
        # self.df_15month.loc[((self.df_15month.index.month == 1) & (self.df_15month.index.day == 1)), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == 4) & (self.df_15month.index.day == 4)), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == 5) & (self.df_15month.index.day == 1)), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == 7) & (self.df_15month.index.day == 1)), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == 10) & (self.df_15month.index.day < 2)), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == 12) & (self.df_15month.index.day == 25)), 'festival'] = 1
        # # self.df_15month.loc[((self.df_15month.index.month == lunar_date(2021, 1, 1).to_datetime().month) & (self.df_15month.index.day == (lunar_date(2021, 1, 1).to_datetime().day))), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == lunar_date(2021, 1, 1).to_datetime().month) & (
        #             self.df_15month.index.day < (lunar_date(2021, 1, 4).to_datetime().day))), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == lunar_date(2020 or 2021, 5, 5).to_datetime().month) & (
        #         self.df_15month.index.day == (lunar_date(2020 or 2021, 5, 5).to_datetime().day))), 'festival'] = 1
        # self.df_15month.loc[((self.df_15month.index.month == lunar_date(2021, 9, 9).to_datetime().month) & (
        #         self.df_15month.index.day == (lunar_date(2021, 9, 9).to_datetime().day))), 'festival'] = 1

        self.df_15month_lable = self.df_15month['CoolingLoad']
        df_15month_feature = self.df_15month.drop(columns=['ST_CoolingLoad', 'NT_CoolingLoad','CoolingLoad'])

        df_15month_feature = df_add_col(df_15month_feature, feature_names)

        self.df_15month_feature = df_15month_feature[feature_names]  # 排序



        df_rows_num = self.df_15month_feature.shape[0]

        test_with_label_opt=self.opt['datasets']['train']['test_with_label']['start']
        self.df_15month_feature_trainval = self.df_15month_feature[self.df_15month_feature.index[0]:test_with_label_opt]
        self.df_15month_lable_trainval = self.df_15month_lable[self.df_15month_feature.index[0]:test_with_label_opt]



        self.df_15month_feature_testlabel = self.df_15month_feature[test_with_label_opt:self.df_15month_feature.index[-1]]
        self.df_15month_lable_testlabel = self.df_15month_lable[test_with_label_opt:self.df_15month_feature.index[-1]]
        # self.df_15month_feature_trainval = self.df_15month_feature['2020-06-01 00:00:00':'2020-10-01 00:00:00']
        # self.df_15month_lable_trainval = self.df_15month_lable['2020-06-01 00:00:00':'2020-10-01 00:00:00']
        #
        # self.df_15month_feature_testlabel = self.df_15month_feature['2021-09-24 00:00:00':'2021-09-30 23:45:00']
        # self.df_15month_lable_testlabel = self.df_15month_lable['2021-09-24 00:00:00':'2021-09-30 23:45:00']
        # 划分 trainval   和   with label的test
        # xgb库的训练要用到两部分数据，第一部分part0是训练集，第二部分part1是验证集，所以这里有划分



        self.df_15month_feature_train, self.df_15month_feature_val, self.df_15month_lable_train, self.df_15month_lable_val = train_test_split(
            self.df_15month_feature_trainval, self.df_15month_lable_trainval,
            test_size=self.opt['datasets']['train'][
                'val_ratio'],
            random_state=42)

        self.dtrain = xgb.DMatrix(self.df_15month_feature_train, self.df_15month_lable_train)#,missing=-9999)
        self.dval = xgb.DMatrix(self.df_15month_feature_val, self.df_15month_lable_val)#,missing=-9999)
        self.watchlist = [(self.dtrain, 'train'), (self.dval, 'validate')]

        # prepare test datasets
        if not self.isTrain:
            name_order = self.xgb_model.feature_names  # 获取排序

            self.df_predict = csv_preprocess(self.opt, self.opt['datasets']['test'])

            df_predict_feature_test = df_add_col(self.df_predict, feature_names)
            df_predict_feature_test = df_predict_feature_test[name_order]
            df_15month_feature_testlabel = self.df_15month_feature_testlabel[name_order]

            self.dpredict_feature = xgb.DMatrix(df_predict_feature_test)#,missing=-9999)
            self.dtestlabel_feature = xgb.DMatrix(df_15month_feature_testlabel)#,missing=-9999)
            a=0

    def train(self):

        self.xgb_model = xgb.train(self.opt['xgb_params'], self.dtrain, num_boost_round=1000,#self.opt['xgb_params']['ntree'],
                                   evals=self.watchlist,
                                   early_stopping_rounds=self.opt['xgb_params']['early_stop'], verbose_eval=5)


    def eval(self):
        #预测集
        lable_predict_test = self.xgb_model.predict(self.dpredict_feature)#预测集
        lable_predict_test = pd.DataFrame(lable_predict_test, index=self.df_predict.index, columns=["test"])
        # xgb_predict_plot(lable_predict_test, 'test', self.opt['path']['results_root'])
        #测试集
        lable_predict_testlabel = self.xgb_model.predict(self.dtestlabel_feature)
        lable_predict_testlabel = pd.DataFrame(lable_predict_testlabel, index=self.df_15month_feature_testlabel.index,
                                               columns=['predict'])
        # xgb_predict_plot(lable_predict_testlabel, 'testlabel', self.opt['path']['results_root'],
        #                  plot_start=self.opt['plot']['start'], plot_end=self.opt['plot']['end'],
        #                  Y=self.df_15month_lable)
        #保存预测结果
        lable_predict_test=lable_predict_test.resample(rule='1H').mean()
        lable_predict_test.to_csv('../results/test.csv', sep=',', index=True, header=True)
        #测试集比较结果
        xgb_predict_plot(lable_predict_testlabel.resample(rule='1H').mean(), 'testlabel', self.opt['path']['results_root'],
                     plot_start=self.opt['plot']['start'], plot_end=self.opt['plot']['end'],
                     Y=self.df_15month_lable.resample(rule='1H').mean())

        Y_test=lable_predict_testlabel#['2021-09-24 00:00:00': '2021-09-30 23:45:00']
        Y=self.df_15month_lable['2021-07-01 00:00:00':'2021-09-30 23:45:00']
        RMSE = (metrics.mean_squared_error(Y_test,Y))** 0.5

        print("RMSE : " + str(RMSE))
        print("r2_score : " + str(r2_score(Y, Y_test)))

