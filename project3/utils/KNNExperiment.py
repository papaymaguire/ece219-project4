from surprise.prediction_algorithms.knns import KNNWithMeans
from project3.utils.ExperimentBase import ExperimentBase
class KNNExperiment(ExperimentBase):
    def __init__(self, k) -> None:
        super().__init__()
        sim_options = {
            'name': 'pearson'
        }
        self.algo = KNNWithMeans(k=k, sim_options=sim_options)