from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ClusteringEnsemble(BaseEstimator, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X):
        for model in self.models:
            model.fit(X)

    def predict(self, X):
        predictions = np.column_stack([model.labels_ for model in self.models])
        majority_voting = []
        for sample in predictions:
            counts = np.bincount(sample)
            majority_voting.append(np.argmax(counts))
        return majority_voting
