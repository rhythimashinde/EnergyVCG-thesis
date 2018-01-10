import numpy as np
class BaseDecisionLogic():

    def __init__(self,model):
        np.random.seed()
        self.last_actions=None
        self.model=model

    def get_decision(self,perceptions):
        """
        Args:
        perceptions: a list of dictionaries, each containing the perception vector associated with one agent
        Returns: a list of dictionaries representing the contributions of the agents. Must contain keys ["agentID", "timestep"]
        """
        if not isinstance(perceptions,list):
            perceptions=[perceptions] # make sure it is a list
        self.last_actions=[{"contribution":0,"cost":0,"agentID":a["agentID"],"contributed":False,"timestep":a["timestep"]} for a in perceptions]
        return self.last_actions

    def feedback(self,perceptions,reward):
        pass
