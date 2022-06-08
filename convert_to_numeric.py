import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class convert_to_numeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ordinal_encoder = OrdinalEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in X:
            if len(np.unique(X[column])) < 10 and type(X[column].iloc[0]) == str:
                reshaped = X[column].values.reshape(-1,1)
                X[column] = self.ordinal_encoder.fit_transform(reshaped)
        return(X)