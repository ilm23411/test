# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:31:11 2018

@author: Origamii
"""

import lightgbm as lgb
import pandas as pd
import numpy as np


X = pd.DataFrame([0,0,1,1,1,1,2,3,1,1])

y = np.array([0,0,0,0,0,1,1,1,1,1])


lgb_train = lgb.Dataset(data=X, label=y)


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    #'num_leaves': 1,
    'max_depth':1,
    #'learning_rate': 0.05,
    #'feature_fraction': 0.9,
    #'bagging_fraction': 0.8,
    #'bagging_freq': 5,
    #'verbose': 0,
    'min_data':1,
    'min_data_in_bin':1,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5,
                #valid_sets=lgb_train,
                #valid_sets=lgb_eval,
                #early_stopping_rounds=5
                )

model = gbm.dump_model()
