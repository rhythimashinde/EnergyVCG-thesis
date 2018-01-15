import numpy as np
from src.utils import *
class BaseEvaluationLogic():
    def __init__(self):
        np.random.seed()

    def get_evaluation(self,decisions,rewards,threshold):
        """
        Computes the measures for this round

        Args:
        decisions: the list of dictionaries for all agents
        rewards: a list of rewards for all agents
        threshold: the value of the success threshold (for contribution)

        Returns:
        A list of dictionaries containing the evaluation of the population behavior
        """
        contributions,values,costs=zip(*[(d["contributed"],d["contribution"],d["cost"]) for d in decisions])
        rews=[i["reward"] for i in rewards]
        tc=tot_contributions([int(c) for c in contributions])
        return [{"gini":gini(values),
                "cost":cost(costs),
                "social_welfare":social_welfare(costs,rews),
                "efficiency":efficiency(threshold,tc),
                "success":success(threshold,tc),
                "tot_contrib":tc}]
