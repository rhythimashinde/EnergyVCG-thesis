import numpy as np
from nego.src.utilsnego import *
from src.EvaluationLogic import BaseEvaluationLogic
class NegoEvaluationLogic(BaseEvaluationLogic):
    def __init__(self,model):
        np.random.seed()
        self.model = model

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
        #print(self.model)
        #contributions,values,costs=zip(*[(d["contributed"],d["contribution"],d["cost"]) for d in decisions])
        #tc=tot_contributions([int(c) for c in contributions])
        # return [{"gini":gini(values),
        #         "efficiency":efficiency(self.model.N,tc),
        #         "tot_contrib":tc}]
        N = self.model.N
        costs = [a.current_state["cost"] for a in self.model.schedule.agents]
        rewards = [a.current_state["reward"]["reward"] for a in self.model.schedule.agents]
        return [{"social_welfare":social_welfare(costs,rewards,N)}]
