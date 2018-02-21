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
        actions = [a.current_state["action"] for a in self.model.schedule.agents]
        rewards = [a.current_state["reward"]["reward"] for a in self.model.schedule.agents]
        costs= [a.current_state["cost"]for a in self.model.schedule.agents]
        N_low = self.model.N
        N_high = self.model.N
        costs_low = [a.current_state["cost"] for a in self.model.schedule.agents
                 if a.current_state["perception"]["social_type"]==1]
        costs_high = [a.current_state["cost"] for a in self.model.schedule.agents
                 if a.current_state["perception"]["social_type"]==2]
        rewards_low = [a.current_state["reward"]["reward"] for a in self.model.schedule.agents
                   if a.current_state["perception"]["social_type"]==1]
        rewards_high = [a.current_state["reward"]["reward"] for a in self.model.schedule.agents
                   if a.current_state["perception"]["social_type"]==2]
        tot_low_agents = 0
        tot_high_agents = 0
        eff = []

        # base with bilateral
        # set_new =[({"agent":a,"partner":a.partner_selection_orderbid()}) for a in self.model.schedule.agents]

        # exp 2 and 4 with mediation
        set_new = self.model.decision_fct.get_partner()

        # exp 3 and 5 with bid split mediation
        # set_new = self.model.decision_fct.get_partner_bidsplit()

        for i in range(self.model.N):
            x = set_new[i]["agent"]
            if x.current_state["perception"]["social_type"]==1:
                N_low=N_low-1
                if set_new[i]["partner"]!=None:
                    tot_low_agents+=1
            else:
                N_high=N_high-1
                if set_new[i]["partner"]!=None:
                    tot_high_agents+=1

            if set_new[i]["partner"] != None:
                x = set_new[i]["agent"]
                partner = set_new[i]["partner"]
                if x.current_state["type"] == "buyer":
                    consume = x.current_state["perception"]["consumption"]
                    old_produce = partner.current_state["perception"]["old_production"]
                    if (consume+old_produce)!=0 and old_produce>=0:
                        eff.append(old_produce/(old_produce+consume))
                else:
                    produce = x.current_state["perception"]["production"]
                    old_consume = partner.current_state["perception"]["old_consumption"]
                    if (produce+old_consume)!=0 and old_consume>=0:
                        eff.append(old_consume/(old_consume+produce))

        return [{"social_welfare_cost":social_welfare_costs(costs,rewards,self.model.N),
                 "social_welfare":social_welfare(costs,rewards,self.model.N),
                 "social_welfare_high":social_welfare(costs_high,rewards_high,N_low),
                 "social_welfare_low":social_welfare(costs_low,rewards_low,N_high),
                 "gini":gini(actions),"efficiency":efficiency_nego(eff,(tot_high_agents+tot_low_agents)),
                 "market_access":success_nego(self.model.N,(tot_high_agents+tot_low_agents)),
                 "market_access_low":success_nego(N_low,tot_high_agents),
                 "wealth_distribution":gini(rewards),
                 "wealth_distribution_high":gini(rewards_high),
                 "wealth_distribution_low":gini(rewards_low),
                 "market_access_high":market_access(N_high,tot_low_agents)}]
