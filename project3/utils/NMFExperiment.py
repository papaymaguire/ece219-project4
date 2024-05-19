import numpy as np
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.model_selection import cross_validate
from project3.utils.ExperimentBase import ExperimentBase
class NMFExperiment(ExperimentBase):
    def __init__(self, n_factors) -> None:
        super().__init__()
        self.algo = NMF(n_factors=n_factors)