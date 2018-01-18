import random
from src.DecisionLogic import BaseDecisionLogic
from src.Supervisor import BaseSupervisor

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
                            "action":a.current_state["action"],"partner":a.current_state["partner"],
                            "new_production":a.current_state["perception"]["production"],
                            "new_consumption":a.current_state["perception"]["consumption"]}
                           for a,p in zip(self.model.schedule.agents,perceptions)]
        #print(self.last_actions)
        return self.last_actions

class NegoDecisionLogicAgent(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        if perceptions is None:
            perceptions=self.model.current_state["perception"]
        a = self.model.partner_selection()
        other = self.model.model.schedule.agents
        cost = self.model.transactions()
        if a != None:
            perc=a.current_state["perception"]
            for other in other:
                perc_other=other.current_state["perception"]
                if self.model.current_state["type"] == "buyer" and a.current_state["type"] == "seller":
                    self.model.current_state.update({"action": 1})  # buy
                    a.current_state.update({"action": 2}) # sell
                    if perc["consumption"] <= perc_other["production"]:
                        a.current_state["perception"].update({"production":perc_other["production"]-perc["consumption"]})
                        self.model.current_state["perception"].update({"consumption": 0})
                        # allocate this remaining energy as surplus in second round
                    else:           # not all needs are satisfied
                        a.current_state["perception"].update({"production": 0})
                        self.model.current_state["perception"].update({"consumption": perc["consumption"]-perc_other["production"]})
                elif self.model.current_state["type"] == "seller" and a.current_state["type"] == "buyer":
                    self.model.current_state.update({"action": 2})  # sell
                    a.current_state.update({"action": 1}) # buy
                    if perc["production"] >= perc_other["consumption"]:
                        self.model.current_state["perception"].update({"production": perc["production"] - perc_other["consumption"]})
                        a.current_state["perception"].update({"consumption": 0})
                    else:           # not all needs are satisfied
                        self.model.current_state["perception"].update({"production": 0})
                        a.current_state["perception"].update({"consumption":perc_other["consumption"] - perc["production"]})
                # else:
                #     raise(AssertionError,"Invalid partner selected: types not matching")
        #print(self.model.current_state["action"])
        return self.model.current_state["action"]

    def get_feedback(self,perceptions,reward):
        pass