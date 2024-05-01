import itertools
from typing import Literal

import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from hdbscan import HDBSCAN
from sklearn import metrics
from umap import UMAP

from MLP import MLP
from Autoencoder import Autoencoder

approved_reducers = ["none", "svd", "nmf", "umap", "auto"]
approved_clusterers = ["kmeans", "agglom", "hdbscan", "mlp"]

class ClusteringExperiment:
    def __init__(self) -> None:
        self.reducers = []
        self.r_experiments: list[dict] = []
        self.clusterers = []
        self.c_experiments: list[dict] = []
        self.reduced_features = []
        self.clustered_labels = []
        self.results = None

    def _get_reducer(self, reducer_name):
        if reducer_name == "none":
            return None
        elif reducer_name == "svd":
            return TruncatedSVD(random_state=0)
        elif reducer_name == "nmf":
            return NMF(random_state=0)
        elif reducer_name == "umap":
            return UMAP(random_state=0)
        elif reducer_name == "auto":
            return Autoencoder()
        else:
            raise ValueError("reducer not approved")
        
    def _get_clusterer(self, clusterer_name):
        if clusterer_name == "kmeans":
            return KMeans(random_state=0)
        elif clusterer_name == "agglom":
            return AgglomerativeClustering()
        elif clusterer_name == "hdbscan":
            return HDBSCAN()
        elif clusterer_name == "mlp":
            return MLP()
        else:
            raise ValueError("clusterer not approved")

    def add_reducer(self, reducer: Literal["none", "svd", "nmf", "umap", "auto"], arg_dict):
        """if reducer is "none" arg_dict is irrelevant"""
        if reducer not in approved_reducers:
            raise ValueError("Reducer not approved")
        if reducer == "none":
            self.reducers.append((reducer, None))
        else:
            self.reducers.append((reducer, arg_dict))

    def add_clusterer(self, clusterer: Literal["kmeans", "agglom", "hdbscan", "mlp"], arg_dict):
        if clusterer not in approved_clusterers:
            raise ValueError("Clusterer not approved")
        self.clusterers.append((clusterer, arg_dict))

    def _design(self):
        self.r_experiments = []
        self.c_experiments = []

        for reducer_config in self.reducers:
            if reducer_config[0] == "none" or len(reducer_config[1]) == 0:
                self.r_experiments.append({"dim_reduce": reducer_config[0]})
                continue
            r_keys, r_values = zip(*reducer_config[1].items())
            arg_experiments = [dict(zip(r_keys, v)) for v in itertools.product(*r_values)]
            self.r_experiments.append([{"dim_reduce": reducer_config[0]} | a for a in arg_experiments])
        for cluster_config in self.clusterers:
            if len(cluster_config[1]) == 0:
                self.c_experiments.append({"method": cluster_config[0]})
                continue
            c_keys, c_values = zip(*cluster_config[1].items())
            c_arg_experiments = [dict(zip(c_keys, v)) for v in itertools.product(*c_values)]
            self.c_experiments.append([{"method": cluster_config[0]} | a for a in c_arg_experiments])

    def get_total_experiments(self):
        return len(self.r_experiments) * len(self.c_experiments)

    def run(self, features):
        self._design()
        if len(self.r_experiments) == 0 or len(self.c_experiments) == 0:
            raise ValueError("design not called before run or invalid config")
        
        for r_experiment in self.r_experiments:
            reducer_name = r_experiment.pop('dim_reduce')
            reducer = self._get_reducer(reducer_name)
            if reducer is None:
                self.reduced_features.append((features, {'dim_reduce': reducer_name} | r_experiment))
                continue

            reducer.set_params(**r_experiment)
            dim_reduced_feats = reducer.fit_transform(features)
            self.reduced_features.append((dim_reduced_feats, {"dim_reduce": reducer_name} | r_experiment))
        
        for c_experiment in self.c_experiments:
            cluster_name = c_experiment.pop('method')
            
            for feature_set in self.reduced_features:
                clusterer = self._get_clusterer(cluster_name)
                clusterer.set_params(**c_experiment)
                clusterer.fit(feature_set[0])
                self.clustered_labels.append((clusterer.labels_), {"method": cluster_name} | c_experiment | feature_set[1])

    def eval(self, labels):
        results = []
        for strategy in self.clustered_labels:
            pred_labels = strategy[0]
            metadata = strategy[1]
            scores = {}
            scores["Homogeneity"] = metrics.homogeneity_score(labels, pred_labels)
            scores["Completeness"] = metrics.completeness_score(labels, pred_labels)
            scores["V-measure"] = metrics.v_measure_score(labels, pred_labels)
            scores["Adjusted rand index"] = metrics.adjusted_rand_score(labels, pred_labels)
            scores["Adjusted mutual information score"] = metrics.adjusted_mutual_info_score(labels, pred_labels)
            scores["Contingency matrix"] = metrics.cluster.contingency_matrix(labels, pred_labels)
            results.append(metadata | scores)
        self.results = pd.DataFrame(results)
        return self.results

