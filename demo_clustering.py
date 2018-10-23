from sklearn.datasets import load_iris
from easy_sklearn import EasySklearnClustering

ds = load_iris()
X = ds.data
y = ds.target
c = ds.feature_names

esc = EasySklearnClustering()
X_train, X_test, y_train, y_test = esc.split_data(X, y)
esc.n_clusters = 3
esc.set()
esc.fit(X_train, y_train)
esc.score(X_test, y_test)
esc.plot_learning_curve()
clf = esc.best_model_
# print(clf)
esc.optimize(scoring='f1_weighted')
y_pred = esc.predict(X_test)
esc.plot_cluster(X_train, clf)
