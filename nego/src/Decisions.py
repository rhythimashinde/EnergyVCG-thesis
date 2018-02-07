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
                            "biased":p["biased"],"bias_degree":p["bias_degree"],
                            "new_consumption":a.current_state["perception"]["consumption"]}
                           for a,p in zip(self.model.schedule.agents,perceptions)]
        return self.last_actions

    def get_partner(self):
        pass

    def get_partner_bidsplit(self):
        pass

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_partner(self):
        other = self.model.model.schedule.agents
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

        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)

        if len(sellers_sorted)<=len(buyers_sorted):
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

    def get_partner_bidsplit(self):
        other = self.model.model.schedule.agents
        perc=self.model.current_state["perception"]
        sellers = []
        buyers = []
        for a in other:
            a.seller_buyer()
            perc_other = a.current_state["perception"]
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"agent_bid":perc_other["tariff"],
                                "produce":a.current_state["perception"]["production"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"agent_bid":perc_other["tariff"],
                               "consume":a.current_state["perception"]["consumption"]})
        if not sellers or all(sellers[i]["produce"] == 0 for i in range(len(sellers))):
            pass
        else:
            produce_smallest = min (sellers[i]["produce"] for i in range(len(sellers)) if sellers[i]["produce"]>0)
            # get smallest element of produce greater than 0
            for i in range(len(sellers)):
                sellers[i]["produce"] =round((sellers[i]["produce"]/produce_smallest),0)
                for a in other:
                    if sellers[i]["agent"] == a:
                        a.current_state["perception"].update({"production":sellers[i]["produce"]})
        if not buyers or all(buyers[i]["consume"] == 0 for i in range(len(buyers))):
            pass
        else:
            consume_smallest = min (buyers[i]["consume"] for i in range(len(buyers)) if buyers[i]["consume"]>0)
            for i in range(len(buyers)):
                buyers[i]["consume"] = round((buyers[i]["consume"]/consume_smallest),0)
                for a in other:
                    if buyers[i]["agent"] == a:
                        a.current_state["perception"].update({"consumption":buyers[i]["consume"]})
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)
        if len(sellers_sorted)<=len(buyers_sorted):
            sorted_list = sellers_sorted
            other_list = buyers_sorted
        else:
            sorted_list = buyers_sorted
            other_list = sellers_sorted

        partner_set = []
        for i in range(len(sorted_list)):
            x = sorted_list[i]["agent"]
            y = other_list[i]["agent"]
            x.current_state.update({"partner":y})
            partner_set.append({"agent":x,"partner":y})
        return partner_set

    def get_decision(self,perceptions):
        if perceptions is None:
            perceptions=self.model.current_state["perception"]

        # base: bilateral partner selection, include below snippet only
        a = self.model.partner_selection_orderbid()

        # exp 1: bilateral bid split partner selction, include below snippet only
        # a = self.model.partner_selection_orderbid_bidsplit()

        # for exp 2 and 4: mediated partner selection, include below snippet only
        # partner_set = self.model.partner_selection_orderbid_mediated()
        # a= None
        # if partner_set !=None:
        #     for i in partner_set:
        #         selfID = self.model.current_state["agentID"]
        #         agenti= i["agent"]
        #         agentID = agenti.current_state["agentID"]
        #         if agentID==selfID:
        #             a = i["partner"]

        # exp 3 and 5: mediated bid split partner selection, include below snippet only
        # partner_set = self.model.partner_selection_orderbid_bidsplit_mediated()
        # a= None
        # if partner_set !=None:
        #     for i in partner_set:
        #         selfID = self.model.current_state["agentID"]
        #         agenti= i["agent"]
        #         agentID = agenti.current_state["agentID"]
        #         if agentID==selfID:
        #             a = i["partner"]

        if a!= None:
            p_p=a.current_state["perception"]
            pc_p=a.current_state
            p = self.model.current_state["perception"]
            pc = self.model.current_state

            # for exp 4, 5 with mediator discrimination, include everything below
            # if p["bias_degree"] == 0 or (p["bias_degree"]==1 and p["social_type"] == p_p["social_type"]):

            # for base, exp 1 with agents discrimination, include everything below, comment above
            if p["biased"] == 0 or (p["biased"] == 1 and p["social_type"] == p_p["social_type"]):

                # for exp 2, 3 with no discrimination, include everything below, comment above
                if pc["type"] == "buyer" and pc_p["type"] == "seller":
                    if p["consumption"] <= (p_p["production"] - p_p["consumption"]):
                        pc.update({"action": 1})  # buy
                        pc_p.update({"action": 2}) # sell
                        p_p.update({"production":p_p["production"]-p["consumption"]})
                        p.update({"consumption": 0})
                    else:
                        p_p.update({"production": 0})
                        p.update({"consumption": p["consumption"]-p_p["production"]})
                elif pc["type"] == "seller" and pc_p["type"] == "buyer":
                    if (p["production"]-p["consumption"]) >= p_p["consumption"]:
                        pc.update({"action": 2})  # sell
                        pc_p.update({"action": 1}) # buy
                        p.update({"production": p["production"] - p_p["consumption"]})
                        p_p.update({"consumption": 0})
                    else:
                        p.update({"production": 0})
                        p_p.update({"consumption":p_p["consumption"] - p["production"]})
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
        return self.model.current_state["reward"]