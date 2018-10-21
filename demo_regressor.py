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
esc.set()
esc.fit(X_train, y_train)
esc.score(X_test, y_test)
esc.plot_learning_curve()
