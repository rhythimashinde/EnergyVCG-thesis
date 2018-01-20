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
        actions = [a.current_state["action"] for a in self.model.schedule.agents]
        N = self.model.N
        costs = [a.current_state["cost"] for a in self.model.schedule.agents]
        rewards = [a.current_state["reward"]["reward"] for a in self.model.schedule.agents]
        tot_agents = 0
        production_tot = 0
        consumption_met = 0
        for a in self.model.schedule.agents:
            if a.current_state["partner"] != None:
                tot_agents = tot_agents + 1
                if a.current_state["type"] == "seller":
                    partner = a.current_state["partner"]
                    produce = a.current_state["perception"]["production"]
                    consume = partner.current_state["perception"]["consumption"]
                    if a.current_state["perception"]["production"] >= partner.current_state["perception"]["consumption"]:
                        # if the consumption is more and the seller is able to meet only a limited consumption,
                        # then the consumption met would be equivalent to the production available by seller
                        production_tot = production_tot + produce
                        consumption_met = consumption_met + consume
                    else:
                        production_tot = production_tot + produce
                        consumption_met = consumption_met + produce
        return [{"social_welfare":social_welfare(costs,rewards,N),"gini":gini(actions),
                 "success":success_nego(N,tot_agents),"efficiency":efficiency_nego(consumption_met,production_tot)}]
