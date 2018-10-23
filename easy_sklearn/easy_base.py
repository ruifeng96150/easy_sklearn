# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import utils
from sklearn.model_selection import train_test_split
import copy
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, make_scorer, f1_score, accuracy_score, mean_squared_error


class EasySklearn(object):

    def __init__(self):
        self.es_models = {}
        self.es_scalers = {}
        self.X_train_, self.y_train_ = [], []
        self.X_test_, self.y_test_ = [], []
        self.best_model_ = None
        self.best_scaler_ = None
        self.cv_best_model_ = None
        self.scores = pd.DataFrame(columns=('model', 'scaler', 'train_score', 'valid_score', 'time'))

    def split_data(self, X, y, test_size=0.2, random_state=1):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @property
    def default_scalers_(self):
        return {
            'maxabs': preprocessing.MaxAbsScaler(),
            'robust': preprocessing.RobustScaler(),
            'scale': preprocessing.StandardScaler(),
            'norm': preprocessing.Normalizer(),
            'qt': preprocessing.QuantileTransformer(),
            'minmax': preprocessing.MinMaxScaler(),
        }

    @property
    def default_scalers_name_(self):
        return [scaler for scaler in self.default_scalers_]

    @property
    def default_models_(self):
        return {}

    @property
    def default_models_name_(self):
        return {}

    def set(self, models='', scalers=''):
        _models, _scalers = {}, {}
        if models is '':
            _models = self.default_models_
        else:
            _models[models] = self.default_models_[models]
        self.es_models = _models

        if scalers is '':
            _scalers = self.default_scalers_
        else:
            _scalers[scalers] = self.default_scalers_[scalers]
        self.es_scalers = _scalers

    def fit(self, X, y):
        print('begin train:------------------------------')
        print('traing data shape', X.shape)
        self.X_train_, self.y_train_ = X, y
        X_train, X_valid, y_train, y_valid = self.split_data(X, y)
        i, best_score = 0, 0
        for scaler_name in self.es_scalers:
            print('training scaler:', scaler_name)
            scaler = self.es_scalers[scaler_name].fit(X_train)
            X_train_trans = scaler.transform(X_train)
            X_valid_trans = scaler.transform(X_valid)
            for model in self.es_models:
                start_time = time.time()
                clf = self.es_models[model]['clf']
                clf.fit(X_train_trans, y_train)
                train_score = clf.score(X_train_trans, y_train)
                valid_score = clf.score(X_valid_trans, y_valid)
                self.scores.loc[i] = [model, scaler_name, train_score, valid_score, time.time() - start_time]
                if valid_score > best_score or self.best_model_ is None:
                    print('find best model', scaler_name, model)
                    self.best_model_ = copy.deepcopy(clf)
                    self.best_scaler_ = copy.deepcopy(scaler)
                    best_score = valid_score
                i += 1

        self.scores.sort_values(by=['valid_score'], ascending=0, inplace=True)
        self.scores.reset_index(inplace=True)
        del self.scores['index']
        print('\ntrain result:------------------------------')
        print(self.scores.head(10))
        print('\n group by model')
        print(self.scores.groupby('model').mean().sort_values(by=['valid_score'], ascending=0))
        print('\n group by scaler')
        print(self.scores.groupby('scaler').mean().sort_values(by=['valid_score'], ascending=0))

    def getModel(self, model, scaler, best_model=False):
        if model is '':
            _model = self.best_model_
            _model_name = self.scores.loc[0, 'model']
        else:
            _model = self.es_models[model]
            _model_name = model

        if scaler == '':
            _scaler = self.best_scaler_
            _scaler_name = self.scores.loc[0, 'scaler']
        else:
            _scaler = self.es_scalers[scaler]
            _scaler_name = scaler

        _defautl_param = self.es_models[_model_name]['param']
        if best_model and self.cv_best_model_ is not None:
            _model, _scaler = self.cv_best_model_, self.best_scaler_
        return _model, _model_name, _scaler, _scaler_name, _defautl_param

    def score(self, X, y, model='', scaler=''):
        print('\ntest result:------------------------------')
        self.X_test_, self.y_test_ = X, y
        _model, _model_name, _scaler, _scaler_name, _ = self.getModel(model, scaler)
        score = _model.score(_scaler.transform(X), y)
        print('test data shape:', X.shape, ' test with:', _model_name, _scaler_name)
        print(' score:', score)
        return score

    def optimize(self, X=None, y=None, model='', scaler='', param_grid=None, scoring=None, n_splits=5, n_iter=20,
                 n_jobs=-1):
        print('\noptimize:------------------------------')
        # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        X, y = X or self.X_train_, y or self.y_train_
        _model, _model_name, _scaler, _scaler_name, _default_param = self.getModel(model, scaler)
        param_grid = param_grid or _default_param
        cv = RandomizedSearchCV(estimator=_model, cv=n_splits, scoring=scoring,
                                param_distributions=param_grid, n_jobs=n_jobs,
                                n_iter=n_iter, verbose=1)
        X_ = _scaler.transform(X)
        cv.fit(X_, y)
        score = cv.best_estimator_.score(_scaler.transform(self.X_test_), self.y_test_)
        self.cv_best_model_ = cv.best_estimator_
        # score = cv.score(_scaler.transform(self.X_test_), self.y_test_)
        print ('best parameter ', cv.best_params_)
        print('The score after optimize', score)

    def predict(self, X, best_model=True, model='', scaler=''):
        _model, _model_name, _scaler, _scaler_name, _ = self.getModel(model, scaler, best_model)
        print('\n predict with model=%s and scaler=%s ------------------------------' % (_model_name, _scaler_name))
        return _model.predict(_scaler.transform(X))

    def plot_learning_curve(self, model='', scaler='', best_model=False):
        print('\nplot learning curve:------------------------------')
        _model, _model_name, _scaler, _scaler_name, _ = self.getModel(model, scaler, best_model)
        title = '%s-%s' % (_model_name, _scaler_name)
        X_train_trans = _scaler.transform(self.X_train_)
        utils.plot_learning_curve(_model, X_train_trans, self.y_train_, title=title)

    def plot_diff(self, model='', scaler='', best_model=True):
        print('\nplot plot_different:------------------------------')
        _model, _model_name, _scaler, _scaler_name, _ = self.getModel(model, scaler, best_model)
        y_pred = _model.predict(_scaler.transform(self.X_test_))
        utils.plot_pred_diff(self.y_test_, y_pred)
