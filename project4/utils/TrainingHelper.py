from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import root_mean_squared_error
from project4.utils.WinePreprocessing import WinePreprocessing

RMSE_SCORING = "neg_root_mean_squared_error"

class TrainingHelper:
    def __init__(self, estimator, dataset: WinePreprocessing, subset: Literal["original", "scaled", "reduced"]) -> None:
        self.estimator = estimator
        self.dataset = dataset
        self.subset = subset
        self.X_train = dataset.data['train']['features'][subset]
        self.y_train = dataset.data['train']['target']
        self.X_test = dataset.data['test']['features'][subset]
        self.y_test = dataset.data['test']['target']

    def cv(self):
        results = cross_validate(self.estimator, self.X_train, self.y_train,
                               cv=10, scoring=RMSE_SCORING, n_jobs=-3, return_estimator=True, return_train_score=True)
        
        train_scores = results['train_score']
        val_scores = results['test_score']
        best_index = np.argmax(val_scores)
        best_estimator = results['estimator'][best_index]

        scores = list(zip(train_scores, val_scores))
        index = [f"Fold{i}" for i in range(1, 11)]
        scores_df = pd.DataFrame(scores, index=index, columns=['train_RMSE', "val_RMSE"])

        preds = best_estimator.predict(self.X_test)
        test_rmse = root_mean_squared_error(self.y_test, preds)

        return best_estimator, scores_df, test_rmse
