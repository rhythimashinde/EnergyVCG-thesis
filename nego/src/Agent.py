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
        self.current_state={"perception":{"production":0,"consumption":0,"tariff":0},"partner":None,"action":0}
        #self.seller_buyer()

    def seller_buyer(self):
        state=self.current_state["perception"]
        #print(state)
        if state["production"] > state["consumption"]:
            self.current_state.update({"type":"seller"})
        if state["production"] < state["consumption"]:
            self.current_state.update({"type":"buyer"})
        #print(self.current_state)

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
        #print(self.current_state)
        print(self.current_state["partner"])
        return self.current_state["partner"] # TODO bring this partner as a value of dictionary of decision for plots

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
        sellers_sorted = sorted(sellers,key=operator.itemgetter('agent_bid')) #ascending sorted sellers as bids
        buyers_sorted = sorted(buyers,key=operator.itemgetter('agent_bid'),reverse=True) #descending sorted buyers as bids
        if len(sellers_sorted)<=len(buyers_sorted): #the remaining energy is wasted
            sorted_list = sellers_sorted
            other_list = buyers_sorted
        else:
            sorted_list = buyers_sorted
            other_list = sellers_sorted
        for i in range(len(sorted_list)):
            x = sorted_list[i]["agent"]
            y = other_list[i]["agent"]
            x.current_state.update({"partner":y})
            #print(x,y)
        print(self.current_state["partner"])
        return self.current_state["partner"]


    # def transactions(self):
    #     if self.current_state["type"] == "seller":
    #         return 1  # modify this cost with every transaction

    def feedback(self,reward,timestep,perceptions=None):
        perc = self.current_state["perception"]
        perc.update({"tariff":perc["tariff"]+1})
        super().feedback(reward,timestep,perceptions)

    def perception(self,perceptions,population=[]):
        super().perception(perceptions,population=[])
        self.current_state["type"]=self.seller_buyer()

