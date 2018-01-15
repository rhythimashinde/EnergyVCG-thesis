import numpy as np
from src.utils import *
class NegoEvaluationLogic():
    def __init__(self):
        np.random.seed()

    def get_evaluation(self,decisions,rewards,threshold):
        """
        Computes the measures for this round

        Args:
        decisions: the list of actions for all agents
        rewards: a list of rewards for all agents
        threshold: the value of the success threshold (for contribution)

        Returns:
        A list of dictionaries containing the evaluation of the population behavior
        """
        contributions,values,costs=zip(*[(d["contributed"],d["contribution"],d["cost"]) for d in decisions])
        tc=tot_contributions([int(c) for c in contributions])
        return [{"gini":gini(values),
                "efficiency":efficiency(self.model.N,tc),
                "tot_contrib":tc}]
