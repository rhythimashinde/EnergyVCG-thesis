from src.DecisionLogic import BaseDecisionLogic
import operator
import math

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
        self.last_actions=[{"old_production":p["old_production"],"old_consumption":p["old_consumption"],
                            "production":p["production"],"tariff":p["tariff"],
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
        other = self.model.schedule.agents
        sellers = []
        buyers = []
        for a in other:
            a.seller_buyer()
            a.transactions()
            perc_other = a.current_state["perception"]
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"agent_bid":perc_other["tariff"],
                                "value":a.current_state["perception"]["production"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"agent_bid":perc_other["tariff"],
                               "value":a.current_state["perception"]["consumption"]})
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)
        i=0
        j=0
        len_s=len(sellers_sorted)
        len_b=len(buyers_sorted)
        r = min(len_b,len_s)
        while True:
            if i<r and j<r and r>=1:
                if sellers_sorted[i]["value"] !=0 and buyers_sorted[j]["value"]!=0:
                    k = (sellers_sorted[i]["value"])-(buyers_sorted[j]["value"])
                    if k==0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[i]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        x.current_state["perception"].update({"production":0})
                        y.current_state["perception"].update({"consumption":0})
                        sellers_sorted[i]["value"] = 0
                        buyers_sorted[j]["value"] = 0
                        j+=1
                        i+=1
                    if k>0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[i]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        y.current_state["perception"].update({"old_consumption":
                                                                  y.current_state["perception"]["consumption"]})
                        x.current_state["perception"].update({"production":
                                                                  x.current_state["perception"]["production"]-
                                                                  y.current_state["perception"]["consumption"]})
                        sellers_sorted[i]["value"] = sellers_sorted[i]["value"]-buyers_sorted[j]["value"]
                        buyers_sorted[j]["value"] = 0
                        j+=1
                        if sellers_sorted[i]["value"]<=0:
                            i+=1
                    if k<0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[j]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        x.current_state["perception"].update({"old_production":
                                                                  x.current_state["perception"]["production"]})
                        y.current_state["perception"].update({"consumption":
                                                                  y.current_state["perception"]["consumption"]-
                                                                  x.current_state["perception"]["production"]})
                        buyers_sorted[j]["value"] = buyers_sorted[j]["value"]-sellers_sorted[i]["value"]
                        sellers_sorted[i]["value"] = 0
                        i+=1
                        if buyers_sorted[j]["value"]<=0:
                            j+=1
                else:
                    break
            else:
                break
        return [({"agent":a,"partner":a.current_state["partner"]}) for a in self.model.schedule.agents]

    def get_partner_bidsplit(self):
        other = self.model.schedule.agents
        sellers = []
        buyers = []
        for a in other:
            a.seller_buyer()
            a.transactions()
            perc_other = a.current_state["perception"]
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"agent_bid":perc_other["tariff"],
                                "value":a.current_state["perception"]["production"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"agent_bid":perc_other["tariff"],
                               "value":a.current_state["perception"]["consumption"]})
        if not sellers or all(sellers[i]["value"] == 0 for i in range(len(sellers))):
            pass
        else:
            for i in range(len(sellers)):
                length = math.ceil(sellers[i]["value"])
                x = sellers[i]["value"]
                sellers[i]["value"] = [1 for i in range(length-1)]
                sellers[i]["value"].append(length-x)
        if not buyers or all(buyers[i]["value"] == 0 for i in range(len(buyers))):
            pass
        else:
            for i in range(len(buyers)):
                length = math.ceil(buyers[i]["value"])
                x = buyers[i]["value"]
                buyers[i]["value"] = [1 for i in range(length-1)]
                buyers[i]["value"].append(length-x)
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid'))
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True)
        i=0
        j=0
        len_s=len(sellers_sorted)
        len_b=len(buyers_sorted)
        r = min(len_b,len_s)
        while True:
            if i<r and j<r and r>=1:
                if sellers_sorted[i]["value"] !=0 and buyers_sorted[j]["value"]!=0:
                    k = len(sellers_sorted[i]["value"])-len(buyers_sorted[j]["value"])
                    if k==0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[i]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        sellers_sorted[i]["value"] = \
                            [sellers_sorted[i]["value"][len(sellers_sorted[i]["value"])-1]-
                             buyers_sorted[j]["value"][len(buyers_sorted[j]["value"])-1]]
                        buyers_sorted[j]["value"] = \
                            [buyers_sorted[j]["value"][len(buyers_sorted[j]["value"])-1]-
                             sellers_sorted[i]["value"][len(sellers_sorted[i]["value"])-1]]
                        if sellers_sorted[i]["value"][0]<0:
                            x.current_state["perception"].update({"production":0})
                        else:
                            x.current_state["perception"].update({"production":sellers_sorted[i]["value"][0]})
                        if sellers_sorted[i]["value"][0]<0:
                            y.current_state["perception"].update({"consumption":0})
                        else:
                            y.current_state["perception"].update({"consumption":buyers_sorted[j]["value"][0]})
                        j+=1
                        i+=1
                    if k>0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[i]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        s = sellers_sorted[i]["value"]
                        s = s[len(buyers_sorted[j]["value"]):]
                        sellers_sorted[i]["value"] = s
                        buyers_sorted[j]["value"] = \
                            [buyers_sorted[j]["value"][len(buyers_sorted[j]["value"])-1]-0.1]
                        y.current_state["perception"].update({"old_consumption":
                                                                  y.current_state["perception"]["consumption"]})
                        y.current_state["perception"].update({"consumption":0})
                        x.current_state["perception"].update({"production":
                                                                  x.current_state["perception"]["production"]-
                                                                  y.current_state["perception"]["consumption"]})
                        j+=1
                        if len(sellers_sorted[i]["value"])==1:
                            i+=1
                    if k<0:
                        x = sellers_sorted[i]["agent"]
                        y = buyers_sorted[j]["agent"]
                        x.current_state.update({"partner":y})
                        y.current_state.update({"partner":x})
                        x.current_state["perception"].update({"old_production":
                                                                  x.current_state["perception"]["production"]})
                        x.current_state["perception"].update({"production":0})
                        y.current_state["perception"].update({"consumption":
                                                                  y.current_state["perception"]["consumption"]-
                                                                  x.current_state["perception"]["production"]})
                        b = buyers_sorted[j]["value"]
                        if sellers_sorted[i]["value"]!=0:
                            b = b[len(sellers_sorted[i]["value"]):]
                        buyers_sorted[j]["value"] = b
                        sellers_sorted[i]["value"] = \
                            [sellers_sorted[i]["value"][len(sellers_sorted[i]["value"])-1]-0.1]
                        i+=1
                        if len(buyers_sorted[j]["value"])==1:
                            j+=1
                else:
                    break
            else:
                break
        return [({"agent":a,"partner":a.current_state["partner"]}) for a in self.model.schedule.agents]

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """

    def get_decision(self,perceptions):
        if perceptions is None:
            perceptions=self.model.current_state["perception"]

        # base: bilateral partner selection
        # partner_set = self.model.partner_selection_orderbid()
        # a = self.model.current_state["partner"]

        # for exp 1 and 3:
        # partner_set = self.model.model.decision_fct.get_partner()
        # a = self.model.current_state["partner"]

        # for exp 2 and 4:
        partner_set = self.model.model.decision_fct.get_partner_bidsplit()
        a = self.model.current_state["partner"]

        if a!= None:
            p_p=a.current_state["perception"]
            pc_p=a.current_state
            p = perceptions
            pc = self.model.current_state

            # old discrmination definition: for base with agents discrimination, include only this snippet
            # if p["biased"] == 0 or (p["biased"] == 1 and p["social_type"]==p_p["social_type"]):

            # new discrimination definition: for base with agents discrimination, include only this snippet
            # if abs(p["production"]-p_p["production"]) > p["discrimination"]:

            # newest discrimination: for base with agents discrimination, include only this snippet
            # if (p["chance_rich"]==True and p_p["income_excess"]<0 and p["income"]>=0.66) \
            #         or (p["chance_average"]==True and p_p["income_excess"]<0 and
            #                     p["income"]<0.66 and p["income"]>=0.33) \
            #         or (p["income"]<0.33):

            # old discrmination definition: for exp 3, 4 with mediator discrimination, include only this snippet
            # if p["bias_degree"] == False or (p["bias_degree"]==True and p["social_type"] == p_p["social_type"]):

            # new discrmination definition: for exp 3, 4 with mediator discrimination, include only this snippet
            # if p["bias_degree"] == False or (p["bias_degree"]==True and
            #                                          abs(p["production"]-p_p["production"]) > p["discrimination"]):

            # newest discrimination: for exp 3, 4 with mediator discrimination, include only this snippet
            if p["bias_degree"] == False or \
                    (p["chance_rich"]==True and p_p["income_excess"]<0 and p["income"]>=0.66) or \
                    (p["chance_average"]==True and p_p["income_excess"]<0 and p["income"]<0.66 and
                             p["income"]>=0.33) or (p["income"]<0.33):

                # for all experiments (with 1,2), or base without discrimination, include everything below
                if pc["type"] == "buyer" and pc_p["type"] == "seller":

                    if p["consumption"] <= (p_p["production"] - p_p["consumption"]):
                        pc.update({"action": 1})  # buy
                        pc_p.update({"action": 2}) # sell
                        p_p.update({"production":(p_p["production"]-p_p["consumption"])-p["consumption"]})
                        p.update({"consumption": 0})
                        p_p.update({"consumption": 0})
                    else:
                        p_p.update({"production": 0})
                        p.update({"consumption": p["consumption"]-(p_p["production"]-p_p["consumption"])})
                elif pc["type"] == "seller" and pc_p["type"] == "buyer":
                    if (p["production"]-p["consumption"]) >= p_p["consumption"]:
                        pc.update({"action": 2})  # sell
                        pc_p.update({"action": 1}) # buy
                        p.update({"production": (p["production"]-p["consumption"]) - p_p["consumption"]})
                        p_p.update({"consumption": 0})
                        p.update({"consumption": 0})
                    else:
                        p.update({"production": 0})
                        p_p.update({"consumption":p_p["consumption"] - (p["production"]-p["consumption"])})
        return self.model.current_state["action"]

    def feedback(self,perceptions,reward):
        rew = self.model.current_state["reward"]
        rew1 = self.model.current_state["perception"]
        partner = self.model.current_state["partner"]
        if partner!=None:
            rew1_p = partner.current_state["perception"]
            self.model.seller_buyer()
            if self.model.current_state["type"]=="seller":
                rew.update({"reward":(rew1["old_production"]-rew1_p["old_consumption"])*2})
        return self.model.current_state["reward"]