import itertools

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from umap import UMAP

class ClusteringExperiment:
    reducers = []
    clusterers = []
    reduced_features = []
    clustered_labels = []
    approved_reducers = {
        "none": "passthrough",
        "svd": TruncatedSVD(random_state=0),
        "nmf": NMF(random_state=0),
        "umap": UMAP(random_state=0)
    }
    approved_clusterers = {
        "kmeans": KMeans(random_state=0),
        "agglom": AgglomerativeClustering(),
        # "hdbscan": HDBSCAN()
    }

    def __init__(self) -> None:
        pass

    def add_reducer(self, reducer, arg_dict):
        """Any reducer added with an empty arg_dict is ignored, passthrough arg_dict is irrelevant"""
        if reducer not in self.approved_reducers:
            raise ValueError("Reducer not approved")
        if reducer == "none":
            self.reducers.append((reducer, None))
        elif len(arg_dict) == 0:
            raise ValueError("Must supply a nonempty arg_dict")
        else:
            self.reducers.append((reducer, arg_dict))

    def add_clusterer(self, clusterer, arg_dict):
        """Any clusterer added with an empty arg_dict is ignored"""
        if clusterer not in self.approved_clusterers:
            raise ValueError("Clusterer not approved")
        if len(arg_dict) == 0:
            raise ValueError("Must supply a nonempty arg_dict")
        self.clusterers.append((clusterer, arg_dict))

    def run(self, features):
        for reducer_config in self.reducers:
            if reducer_config[0] == "none":
                self.reduced_features.append((features, {"dim_reduce": "none"}))
                continue
            if len(reducer_config[1]) == 0:
                raise ValueError("Empty arg_dict found in reducer_config")
            r_keys, r_values = zip(*reducer_config[1].items())
            r_experiments = [dict(zip(r_keys, v)) for v in itertools.product(*r_values)]
            for r_config in r_experiments:
                reducer = self.approved_reducers[reducer_config[0]].set_params(**r_config)
                dim_reduced_feats = reducer.fit_transform(features)
                self.reduced_features.append((dim_reduced_feats, {"dim_reduce": reducer_config[0]} | r_config))
        for cluster_config in self.clusterers:
            if len(reducer_config[1]) == 0:
                raise ValueError("Empty arg_dict found in cluster_config")
            c_keys, c_values = zip(*cluster_config[1].items())
            c_experiments = [dict(zip(c_keys, v)) for v in itertools.product(*c_values)]
            for c_config in c_experiments:
                clusterer = self.approved_clusterers[cluster_config[0]].set_params(**c_config)
                for feature_set in self.reduced_features:
                    clusterer.fit(feature_set[0])
                    self.clustered_labels.append((clusterer.labels_, {"clusterer": cluster_config[0]} | feature_set[1] | c_config))

