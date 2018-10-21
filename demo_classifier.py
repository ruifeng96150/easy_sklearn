from sklearn.datasets import load_iris
from easy_sklearn import EasySklearnClassifier

ds = load_iris()
X = ds.data
y = ds.target
c = ds.feature_names

esc = EasySklearnClassifier()
X_train, X_test, y_train, y_test = esc.split_data(X, y)
esc.set()
esc.fit(X_train, y_train)
esc.score(X_test, y_test)
esc.plot_learning_curve()

clf = esc.best_model_
print(clf)
