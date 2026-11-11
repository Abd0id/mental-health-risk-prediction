import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """Caps numeric values at Q1 - 1.5*IQR and Q3 + 1.5*IQR (bounds learned on the training data only)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lower_, self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
