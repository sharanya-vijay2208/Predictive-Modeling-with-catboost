# -*- coding: utf-8 -*-

# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool, cv

class CatbClassifier:
    '''
    Wrapper for CatBoostClassifier
    '''
    def __init__(self, train_df):
        self.train_df = train_df
        self.model = None
        self.X = None
        self.y = None
        self.categorical_features_indices = None

    def _fill_na(self, value, inplace=True):
        self.train_df.fillna(value, inplace=inplace)

    def _default_args_or_kwargs(self, **kwargs):
        params = {
            'iterations': 100,
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'use_best_model': True,
            'verbose' : False,
            'loss_function' : 'Logloss',
        }
        for k in kwargs.keys():
            if k in params:
                params[k] = kwargs.get(k)
        return params

    def generate_x_y(self, label, null_value=-999):
        self._fill_na(null_value)
        self.X = self.train_df.drop(label, axis=1)
        self.y = self.train_df[label]

    def model_create(self, **kwargs):
        params = self._default_args_or_kwargs(**kwargs)
        if not self.model:
            self.model = CatBoostClassifier(**params)
        else:
            raise ValueError("Cannot overwrite existing model")

    def model_train(self, train_size=0.85, random_state=1234, **kwargs):
        X_train, X_validation, y_train, y_validation = train_test_split(self.X, self.y, 
                                                                        train_size = train_size,
                                                                        random_state = random_state)
        if not self.categorical_features_indices:
            self.categorical_features_indices = np.where(self.X.dtypes != np.float)[0]
        if not self.model:
            self.model_create(**kwargs)
        self.model.fit(
            X_train, y_train,
            cat_features=self.categorical_features_indices,
            eval_set=(X_validation, y_validation),
            logging_level='Verbose'
        )   
        return accuracy_score(y_validation, self.model.predict(X_validation))

    def model_predict(self, dataframe, null_value=999, inplace=True):
        dataframe.fillna(null_value, inplace=inplace)
        results = self.model.predict(dataframe)
        return results
     
    def model_cross_validation(self ,**kwargs):
        cv_data = cv(
            Pool(self.X, self.y, 
                 cat_features=self.categorical_features_indices),
                 self.model.get_params()
        )   
        print(cv_data)    
        return np.max(cv_data['test-Accuracy-mean']) 
    
    def save_model(self, name):
        self.model.save_model('{}.dump'.format(name))

    def load_model(self, name):
        self.model.load_model('{}.dump'.format(name))

 