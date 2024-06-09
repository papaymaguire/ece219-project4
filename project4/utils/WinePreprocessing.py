import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from project4.utils.WineDataset import WineDataset
from project4.utils.DataIO import DataIO
class WinePreprocessing:
    def __init__(self, wine_data: WineDataset, num_cols, cat_cols, target_col,  cache_io:DataIO, stratify_col=None, cache_name="processed_data") -> None:
        path = f'{cache_io.data_path}/{cache_name}'
        if os.path.exists(path):
            self.data = cache_io.load(cache_name)
        else:
            target = wine_data.data[target_col]
            feature_cols = num_cols + cat_cols
            features = wine_data.data[feature_cols]
            stratify = None
            if stratify_col:
                stratify = wine_data.data[stratify_col]
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=stratify, random_state=0)
            feature_cols = num_cols + cat_cols
            discrete_features_idx = [len(feature_cols)-i-1 for i in range(len(cat_cols))]
            scorer = lambda X, y: self._scorer(X, y, discrete_features_idx)
            reduce_pipe = Pipeline(steps=[
                ("scale", StandardScaler()),
                ("select", SelectKBest(scorer, k=10))
            ])
            reduce_pipe.fit(X_train, y_train)
            X_train_scaled = reduce_pipe['scale'].transform(X_train)
            X_test_scaled = reduce_pipe['scale'].transform(X_test)
            X_train_transformed = reduce_pipe.transform(X_train)
            transformed_features = reduce_pipe.get_feature_names_out()
            X_test_transformed = reduce_pipe.transform(X_test)
            self.data = {
                "train": {
                    "features": {
                        "original": X_train.to_numpy(),
                        "scaled": X_train_scaled,
                        "reduced": X_train_transformed
                    },
                    "target": y_train.to_numpy()
                },
                "test": {
                    "features": {
                        "original": X_test.to_numpy(),
                        "scaled": X_test_scaled,
                        "reduced": X_test_transformed
                    },
                    "target": y_test.to_numpy()
                },
                "metadata": {
                    "features":{
                        "original": feature_cols,
                        "reduced": transformed_features,
                        "num_categorical": len(cat_cols)
                    }
                }
            }
            cache_io.save(self.data, cache_name)

    def _scorer (self, X, y, discrete_features_idx):
        return mutual_info_regression(X, y, discrete_features=discrete_features_idx, random_state=0)
    
    def mi_scores(self):
        feature_cols = self.data['metadata']['features']['original']
        num_categorical = self.data['metadata']['features']['num_categorical']
        discrete_features_idx = [len(feature_cols)-i-1 for i in range(num_categorical)]
        X = self.data['train']['features']['original']
        y = self.data['train']['target']
        scores = self._scorer(X, y, discrete_features_idx)
        names = self.data['metadata']['features']['original']
        return list(zip(scores, names))