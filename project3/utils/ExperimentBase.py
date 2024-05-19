import numpy as np
from joblib import Parallel, delayed
from surprise import accuracy
from surprise.model_selection import KFold
class ExperimentBase ():
    def __init__(self) -> None:
        self.algo = None
        self.avg_rmse = None
        self.avg_mae = None

    def fit_and_predict(self, trainset, testset):
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)
        return predictions

    def _fit_and_score(self, trainset, testset):
        self.algo.fit(trainset)
        predictions = self.algo.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae

    def run(self, dataset):
        '''
        Should return average RMSE and average MAE for a 10 fold cross validation
        '''
        kf = KFold(n_splits=10, random_state=0)

        delayed_list = (
            delayed(self._fit_and_score)(trainset, testset) for (trainset, testset) in kf.split(dataset)
        )
            
        results = Parallel(n_jobs=-2)(delayed_list)
        rmses, maes = zip(*results)
        self.avg_rmse = np.average(rmses)
        self.avg_mae = np.average(maes)
        return {
            'rmse': self.avg_rmse,
            'mae': self.avg_mae
        }