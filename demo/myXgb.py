import operator

import matplotlib.pyplot as plt
import pandas as pd
import xgboost  as xgb
from sklearn.model_selection import train_test_split

from codes.utils.util import *


# get data for train, test, and forecast(unseen)


def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    # df.plot()
    df.plot(kind='barh', x='feature', y='fscore',
            legend=False, figsize=(12, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()


def xgb_importance(df, test_ratio, xgb_params, ntree, early_stop, plot_title):
    df = pd.DataFrame(df)
    # split the data into train/test set
    Y = df.iloc[:, -1]
    X = df.iloc[:, :-4]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'validate')]

    xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=True)

    importance = xgb_model.get_fscore()
    importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
    feature_importance_plot(importance_sorted, plot_title)


def xgb_predict_plot(Y_predict, title, save_path, plot_start=None, plot_end=None, Y=None):
    if Y is not None:
        Y = pd.concat([Y, Y_predict], axis=1)
    else:
        Y = Y_predict
    if plot_start is not None and plot_end is not None:
        Y[plot_start:plot_end].plot(ylabel='coolingLoad', figsize=(30, 10))
    else:
        Y.plot(ylabel='coolingLoad', figsize=(15, 10))

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, title + '.png'), dpi=300)
    plt.show()
