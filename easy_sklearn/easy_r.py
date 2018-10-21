# -*- coding: utf-8 -*-
from easy_base import EasySklearn
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import tree, ensemble, linear_model, svm, neighbors


class EasySklearnRegressor(EasySklearn):
    def __init__(self):
        EasySklearn.__init__(self)

    @property
    def default_models_(self):
        return {
            'Tree': {'clf': tree.DecisionTreeRegressor(),
                     'param': {}},
            'GBDT': {'clf': ensemble.GradientBoostingRegressor(),
                     'param': {
                         'learning_rate': [0.1],
                         'max_depth': [4, 6],
                         'alpha': [0.9],
                         'max_leaf_nodes': [10, 20],
                         'min_samples_split': [2, 4]
                     }},
            'Lin': {'clf': linear_model.LinearRegression(),
                    'param': {
                        'fit_intercept': [True, False],
                        'normalize': [True, False]
                    }},
            'Ridge': {'clf': linear_model.Ridge(),
                      'param': {}},
            'Lasso': {'clf': linear_model.Lasso(),
                      'param': {}},
            'ElasN': {'clf': linear_model.ElasticNet(),
                      'param': {}},
            'Lars': {'clf': linear_model.Lars(),
                     'param': {}},
            'Bayers': {'clf': linear_model.BayesianRidge(),
                       'param': {}},
            'Poly2': {'clf': Pipeline([('poly', PolynomialFeatures(degree=2)),
                                       ('std_scaler', StandardScaler()),
                                       ('line_reg', linear_model.LinearRegression())
                                       ]),
                      'param': {}},
            'SGD': {'clf': linear_model.SGDRegressor(),
                    'param': {}},
            'SVM': {'clf': svm.SVR(kernel='rbf', C=1.0, epsilon=1),
                    'param': {}},
            'Knn': {'clf': neighbors.KNeighborsRegressor(),
                    'param': {}},
            'RF': {'clf': ensemble.RandomForestRegressor(),
                   'param':
                       {'n_estimators': [3, 10, 30],
                        'bootstrap': [True, False],
                        'max_features': [3, 7]
                        }},
            'ADA': {'clf': ensemble.AdaBoostRegressor(n_estimators=100),
                    'param': {}},
            'BAG': {'clf': BaggingRegressor(bootstrap=True),
                    'param': {'n_estimators': [50, 100, 200]}},
            'ET': {'clf': tree.ExtraTreeRegressor(),
                   'param': {}},
        }

    @property
    def default_models_name_(self):
        return [model for model in self.default_models_]
