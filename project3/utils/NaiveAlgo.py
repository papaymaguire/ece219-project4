import numpy as np
from surprise import Trainset
from surprise import AlgoBase

class NaiveAlgo(AlgoBase):
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, trainset: Trainset):
        super().fit(trainset)

        self.mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self
    
    def estimate(self, u, i):
        return self.mean