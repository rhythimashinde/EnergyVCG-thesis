from src.Agent import BaseAgent
from mesa import Agent
from nego.src.Decisions import NegoDecisionLogicAgent
import operator

class NegoAgent(BaseAgent):
    def __init__(self,unique_id,model,decision_fct=NegoDecisionLogicAgent):
        """
        The state contains:
        action: last action, 0 for buy 1 for sell
        reward: last reward obtained by the supervisor
        tariff: the preferred prico of sale
        type: whether the agent is a seller or a buyer in this turn
        partner: the selected partner
        perception: last measurement, containing:
            production: how much energy is produced
            consumption: how much energy is requireds
            cost: the cost of contribution
        """
        super().__init__(unique_id,model,decision_fct=decision_fct)
        self.current_state={"perception":{"production":0,"consumption":0,"tariff":0,"social_type":0},
                            "partner":None,"action":0,"cost":0,"reward":0,"agentID":self.unique_id}

    def seller_buyer(self):
        state=self.current_state["perception"]
        if state["production"] > state["consumption"] and state["production"]!=0:
            self.current_state.update({"type":"seller"})
        if state["production"] < state["consumption"] and state["consumption"]!=0:
            self.current_state.update({"type":"buyer"})
        return self.current_state["type"]

    def partner_selection(self):
        other = self.model.schedule.agents
        perc=self.current_state["perception"]
        self.seller_buyer()
        for a in other:
            if a != self: # making sure that the agent doesn't select itself
                perc_other = a.current_state["perception"]
                if self.current_state["type"] != a.current_state["type"]:
                    # modify tariff rule: # of different types
                    self.current_state["partner"] = a
        return self.current_state["partner"]

    def partner_selection_orderbid(self):
        other = self.model.schedule.agents
        perc=self.current_state["perception"]
        sellers = []
        buyers = []
        for a in other:
            a.seller_buyer()
            perc_other = a.current_state["perception"]
            if a.current_state["type"] == "seller":
                sellers.append({"agent":a,"agent_bid":perc_other["tariff"]})
            elif a.current_state["type"] == "buyer":
                buyers.append({"agent":a,"agent_bid":perc_other["tariff"]})
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid')) # ascending sorted sellers as bids
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True) # descending sorted buyers as bids
        if len(sellers_sorted)<=len(buyers_sorted): # the remaining energy is wasted
            sorted_list = sellers_sorted
            other_list = buyers_sorted
        else:
            sorted_list = buyers_sorted
            other_list = sellers_sorted
        for i in range(len(sorted_list)):
            x = sorted_list[i]["agent"]
            y = other_list[i]["agent"]
            x.current_state.update({"partner":y})
        return self.current_state["partner"]

    def partner_selection_orderbid_bidsplit(self):
        #TODO do I need to move this generalised part to the supervisor? Or just do that for mediated negotiation
        other = self.model.schedule.agents
        perc=self.current_state["perception"]
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
            # get smallest element of consumption greater than 0
            for i in range(len(buyers)):
                buyers[i]["consume"] = round((buyers[i]["consume"]/consume_smallest),0)
                for a in other:
                    if buyers[i]["agent"] == a:
                        a.current_state["perception"].update({"consumption":buyers[i]["consume"]})
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid')) # ascending sorted sellers as bids
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True) # descending sorted buyers as bids
        if len(sellers_sorted)<=len(buyers_sorted): # the remaining energy is wasted
            sorted_list = sellers_sorted
            other_list = buyers_sorted
        else:
            sorted_list = buyers_sorted
            other_list = sellers_sorted
        for i in range(len(sorted_list)):
            x = sorted_list[i]["agent"]
            y = other_list[i]["agent"]
            x.current_state.update({"partner":y})
        return self.current_state["partner"]

    def transactions(self):
        if self.current_state["type"] == "seller":
            if self.current_state["perception"]["production"] != 0:
                self.current_state.update({"cost":(self.current_state["perception"]["production"]-
                                                   self.current_state["perception"]["consumption"])/
                                                    self.current_state["perception"]["production"]})
        return self.current_state["cost"]

    def feedback(self,reward,timestep,perceptions=None):
        super().feedback(reward,timestep,perceptions)
        reward_new = self.decision_fct.feedback(perceptions,reward)
        if reward_new["reward"] <= 0: #if there is no rewards in the system then the agent decides to have no partner
            self.current_state.update({"partner":None})

    def perception(self,perceptions,population=[]):
        super().perception(perceptions,population=[])
        self.current_state["type"]=self.seller_buyer()

