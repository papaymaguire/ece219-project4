from project3.utils.ExperimentBase import ExperimentBase
from project3.utils.NaiveAlgo import NaiveAlgo
class NaiveExperiment(ExperimentBase):
    def __init__(self) -> None:
        super().__init__()
        self.algo = NaiveAlgo()