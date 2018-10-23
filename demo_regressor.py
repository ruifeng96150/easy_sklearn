from sklearn.datasets import load_boston
from easy_sklearn import EasySklearnRegressor
import warnings

warnings.filterwarnings("ignore")
ds = load_boston()
X = ds.data
y = ds.target
c = ds.feature_names

esc = EasySklearnRegressor()
X_train, X_test, y_train, y_test = esc.split_data(X, y)
print(X_train.shape, X_test.shape)
esc.set(models='GBDT',scalers='norm')
esc.fit(X_train, y_train)
esc.score(X_test, y_test)
esc.plot_learning_curve()
esc.plot_diff()

# ------------------
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'learning_rate': [0.1, 0.02],
    'max_depth': [4, 6, 8, 10, 20, 30],
    'alpha': [0.7, 0.8, 0.9],
    'max_leaf_nodes': [10, 20, 30, 40],
    'min_samples_split': [2, 4, 7]
}

esc.optimize(scoring='r2', param_grid=param_grid, n_splits=5, n_iter=30)
esc.plot_learning_curve(best_model=True)
