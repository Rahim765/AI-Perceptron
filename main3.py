import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
import warnings
from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression
from convert_to_numeric import convert_to_numeric

warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
data = pd.concat([train, test])

data = data.drop('Unnamed: 0', axis=1)
data = data.drop('id', axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['satisfaction']):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]


X_train = strat_train_set.drop("satisfaction", axis=1)
y_train = np.where(strat_train_set["satisfaction"] == "satisfied", 1, 0)

X2_train = strat_test_set.drop("satisfaction", axis=1)
y2_train = np.where(strat_test_set["satisfaction"] == "satisfied", 1, 0)


pipeline = Pipeline([
    ('to_numeric', convert_to_numeric()),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

pipeline.fit(X_train)
X_train = pipeline.transform(X_train)

pipeline.fit(X2_train)
X2_train = pipeline.transform(X2_train)

clf = Perceptron()
clf.fit(X_train, y_train)

print(clf.score(X2_train , y2_train))

clf2 = SVC(kernel="rbf");
clf2.fit(X_train[0:10000], y_train[0:10000])

print(clf2.score(X2_train , y2_train))
