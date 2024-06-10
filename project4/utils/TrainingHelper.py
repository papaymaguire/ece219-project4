from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from skopt import BayesSearchCV
from project4.utils.WinePreprocessing import WinePreprocessing
from project4.utils.TweetHelper import TweetHelper

RMSE_SCORING = "neg_root_mean_squared_error"

class TrainingHelper:
    def __init__(self, estimator, dataset: WinePreprocessing | TweetHelper, subset: Literal["original", "scaled", "reduced"]) -> None:
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
    
    def grid(self, grid):
        search = GridSearchCV(self.estimator, grid,
                               scoring=RMSE_SCORING, n_jobs=-3, refit=True, cv=10, return_train_score=True)
        search.fit(self.X_train, self.y_train)
        best_estimator = search.best_estimator_
        grid_results = pd.DataFrame(search.cv_results_)
        preds = search.predict(self.X_test)
        test_rmse = root_mean_squared_error(self.y_test, preds)
        return best_estimator, grid_results, test_rmse
    
    def bayes(self, space, n_iter):
        search = BayesSearchCV(self.estimator, space, n_iter=n_iter,
                               scoring=RMSE_SCORING, n_jobs=-3, cv=10, refit=True, random_state=0, return_train_score=True)
        search.fit(self.X_train, self.y_train)
        best_estimator = search.best_estimator_
        grid_results = pd.DataFrame(search.cv_results_)
        preds = search.predict(self.X_test)
        test_rmse = root_mean_squared_error(self.y_test, preds)
        return best_estimator, grid_results, test_rmse
