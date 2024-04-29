import itertools

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from umap import UMAP

class ClusteringExperiment:
    reducers = []
    clusterers = []
    reduced_features = []
    clustered_labels = []

    def __init__(self) -> None:
        pass

    def add_reducer(self, reducer, arg_dict):
        approved_reducers = {
            "none": "passthrough",
            "svd": TruncatedSVD(random_state=0),
            "nmf": NMF(random_state=0),
            "umap": UMAP(random_state=0)
        }
        if reducer not in approved_reducers:
            raise ValueError("Reducer not approved")
        self.reducers.append((approved_reducers[reducer], arg_dict))

    def add_clusterer(self, clusterer, arg_dict):
        approved_clusterers = {
            "kmeans": KMeans(random_state=0),
            "agglom": AgglomerativeClustering(),
           # "hdbscan": HDBSCAN()
        }
        if clusterer not in approved_clusterers:
            raise ValueError("Clusterer not approved")
        self.clusterers.append((approved_clusterers[clusterer], arg_dict))

    def run(self, features):
        for reducer_config in self.reducers:
            r_keys, r_values = zip(*reducer_config[1].items())
            r_experiments = [dict(zip(r_keys, v)) for v in itertools.product(*r_values)]
            for r_config in r_experiments:
                reducer = reducer_config[0].set_params(**r_config)
                dim_reduced_feats = reducer.fit_transform(features)
                self.reduced_features.append((dim_reduced_feats, r_config))
        for cluster_config in self.clusterers:
            c_keys, c_values = zip(*cluster_config[1].items())
            c_experiments = [dict(zip(c_keys, v)) for v in itertools.product(*c_values)]
            for c_config in c_experiments:
                clusterer = cluster_config[0].set_params(**c_config)
                for feature_set in self.reduced_features:
                    clusterer.fit(feature_set[0])
                    self.clustered_labels.append((clusterer.labels_, feature_set[1] | c_config))

