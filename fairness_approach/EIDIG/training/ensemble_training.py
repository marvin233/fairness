from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census, pre_compas, pre_bank


# collect datasets
datasets = [(pre_census.X, pre_census.y), (pre_bank.X, pre_bank.y), (pre_compas.X, pre_compas.y)]
names = ['census', 'bank', 'compas']

# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
eclf = VotingClassifier(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)], voting='soft')


# set a pipeline to handle the prediction process
clf = Pipeline([('scaler', StandardScaler()), ('ensemble', eclf)])


# train, evaluate and save ensemble models for each dataset
for i, ds in enumerate(datasets):
    model = clone(clf)
    X, y = ds
    model.fit(X, y)
    joblib.dump(model, '../models/ensemble_models/' + names[i] + '_ensemble.pkl')

