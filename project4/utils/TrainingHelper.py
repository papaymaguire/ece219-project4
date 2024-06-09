from typing import Literal
from sklearn.model_selection import cross_validate
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
        return cross_validate(self.estimator, self.X_train, self.y_train,
                               cv=10, scoring=RMSE_SCORING, n_jobs=-3, return_estimator=True, return_train_score=True)