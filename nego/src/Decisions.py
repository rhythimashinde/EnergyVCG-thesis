import random
from src.DecisionLogic import BaseDecisionLogic
from src.Supervisor import BaseSupervisor
import operator

class NegoDecisionLogic(BaseDecisionLogic):
    def get_decision(self,perceptions):
        """
        Args:
        perceptions: a list of dictionaries, each containing the perception vector associated with one agent
        Returns: a list of dictionaries representing the contributions of the agents.
        Must contain keys ["agentID", "timestep"]
        """
        # decs=[a.decisions(p) for a,p in zip(self.model.schedule.agents,perceptions)]
        # call each agent's decision fct with the appropriate perception
        # partner = self.model.partner_selection_orderbid()
        self.last_actions=[{"production":p["production"],"tariff":p["tariff"],
                            "consumption":p["consumption"],"agentID":a.unique_id,
                            "contribution":0,"contributed":False,"cost":a.current_state["cost"],
                            "reward":a.current_state["reward"],"action":a.current_state["action"],
                            "partner":a.current_state["partner"],"social_type":p["social_type"],
                            "new_production":a.current_state["perception"]["production"],
                            "biased":p["biased"],"new_consumption":a.current_state["perception"]["consumption"]}
                           for a,p in zip(self.model.schedule.agents,perceptions)]
        # print(self.last_actions)
        return self.last_actions

    def get_partner(self):
        other = self.model.schedule.agents
        sellers = []
        buyers = []
        for a in other:
            perc=a.current_state["perception"]
            a.seller_buyer()
            perc_other = a.current_state["perception"]
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"agent_bid":perc_other["tariff"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"agent_bid":perc_other["tariff"]})

        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))  # ascending sorted sellers as bids
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)  # descending sorted buyers as bids

        if len(sellers_sorted)<=len(buyers_sorted): # the remaining energy is wasted
            sorted_list = sellers_sorted
            other_list = buyers_sorted
        else:
            sorted_list = buyers_sorted
            other_list = sellers_sorted

        partner_set =[]
        for i in range(len(sorted_list)):
            x = sorted_list[i]["agent"]
            y = other_list[i]["agent"]
            x.current_state.update({"partner":y})
            partner_set.append({"agent":x,"partner":y})

        return partner_set

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        if perceptions is None:
            perceptions=self.model.current_state["perception"]
        # partner_set = self.model.partner_selection_mediated()  # mediated
        # a= None
        # for i in partner_set:
        #     if i["agent"]==self:
        #         a = i["partner"]
        a = self.model.partner_selection_orderbid()  # bilateral
        print(a)
        other = self.model.model.schedule.agents
        cost = self.model.transactions()
        if a!= None:
            perc=a.current_state["perception"]
            for other in other:
                perc_other=other.current_state["perception"]
                if self.model.current_state["perception"]["biased"] == 0 \
                        or (self.model.current_state["perception"]["biased"] == 1
                            and self.model.current_state["perception"]["social_type"] == a.current_state["perception"]["social_type"]):
                    # only if buyer is not biased transaction would happen or if biased and has an allocated partner as the same caste
                    if self.model.current_state["type"] == "buyer" \
                            and a.current_state["type"] == "seller" \
                            and perc["consumption"] <= (perc_other["production"] - perc_other["consumption"]):
                        self.model.current_state.update({"action": 1})  # buy
                        a.current_state.update({"action": 2}) # sell
                        if perc["consumption"] <= perc_other["production"]:
                            a.current_state["perception"].update({"production":perc_other["production"]-perc["consumption"]})
                            self.model.current_state["perception"].update({"consumption": 0})
                        # allocate this remaining energy as surplus in second round
                        else:           # not all needs are satisfied
                            a.current_state["perception"].update({"production": 0})
                            self.model.current_state["perception"].update(
                                {"consumption": perc["consumption"]-perc_other["production"]})
                    elif self.model.current_state["type"] == "seller" \
                            and a.current_state["type"] == "buyer" \
                            and (perc["production"]-perc["consumption"]) >= perc_other["consumption"]:
                        self.model.current_state.update({"action": 2})  # sell
                        a.current_state.update({"action": 1}) # buy
                        if perc["production"] >= perc_other["consumption"]:
                            self.model.current_state["perception"].update(
                                {"production": perc["production"] - perc_other["consumption"]})
                            a.current_state["perception"].update({"consumption": 0})
                        else:           # not all needs are satisfied
                            self.model.current_state["perception"].update({"production": 0})
                            a.current_state["perception"].update(
                                {"consumption":perc_other["consumption"] - perc["production"]})
        return self.model.current_state["action"]

    def feedback(self,perceptions,reward):
        rew = self.model.current_state["reward"]
        cost = self.model.current_state["cost"]
        rew1 = self.model.current_state["perception"]
        self.model.seller_buyer()
        if self.model.current_state["type"]=="seller":
            rew.update({"reward":(rew1["production"]-rew1["consumption"])*rew1["tariff"]})
        if self.model.current_state["type"]=="buyer":
            if rew1["production"]<=0:
                rew.update({"reward":0})
            else:
                rew.update({"reward":rew1["production"]*rew1["tariff"]})
        rew.update({"reward":rew["reward"]-cost})
        # print(self.model.current_state["type"], self.model.current_state["reward"],self.model.current_state["cost"])
        return self.model.current_state["reward"]