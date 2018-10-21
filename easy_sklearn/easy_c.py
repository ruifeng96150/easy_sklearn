# -*- coding: utf-8 -*-
from easy_base import EasySklearn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


class EasySklearnClassifier(EasySklearn):
    def __init__(self):
        EasySklearn.__init__(self)

    @property
    def default_models_(self):
        return {
            'SVC': {'clf': SVC(kernel='rbf', probability=False, random_state=1),
                    'param': {}},
            'ORSVC': {'clf': OneVsRestClassifier(SVC(kernel='rbf', probability=False, random_state=1)),
                      'param': {}},
            'GBDT': {'clf': GradientBoostingClassifier(random_state=1),
                     'param': {'n_estimators': [70, 100], 'max_depth': [7], 'subsample': [0.7, 0.9]}},
            'Tree': {'clf': tree.DecisionTreeClassifier(random_state=1),
                     'param': {}},
            'RF': {'clf': RandomForestClassifier(random_state=1),
                   'param': {'n_estimators': [50, 100, 150]}},
            'LR2': {'clf': LogisticRegression(penalty='l2'),
                    'param': {}},
            'KN': {'clf': KNeighborsClassifier(),
                   'param': {}},
            'Bag': {'clf': BaggingClassifier(random_state=1),
                    'param': {}},
            'MLP': {'clf': MLPClassifier(max_iter=1000, random_state=1),
                    'param': {}}
        }

    @property
    def default_models_name_(self):
        return [model for model in self.default_models_]
