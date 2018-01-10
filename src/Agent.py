import numpy as np
from mesa import Agent
from DecisionLogic import *

class BaseAgent(Agent):
    """
    Base agent
    """
    def __init__(self,unique_id,model,decision_fct=BaseDecisionLogic):
        """
        create agent
        """
        super().__init__(unique_id,model)
        np.random.seed()
        self.decision_fct=decision_fct(self) # new instance
        self.current_state={}
        self.log=[]

    def __log(self,timestep):
        """
        Save params to file

        Args:
        timestep: the index of the current round
        """
        dct=self.current_state.copy()
        dct.update({"timestep":timestep,"action":self.get_decision()})
        self.log.append(dct)

    def __learn(self,perceptions,reward):
        """
        Update decision process.
        Private, called from within feedback

        Kwargs:
        perceptions: the state of the system.
        perceptions: the reward.
        """
        self.decision_fct.feedback(perceptions,reward) # update state of decision fct


    def feedback(self,reward,timestep,perceptions=None):
        """
        Updates own state according to feedback obtained this round

        Args:
        reward: a dictionary containing the reward
        timestep: the time
        perceptions: a dictionary containing the current perception. It can be used as the state for a learning algorithm
        """
        self.current_state.update({"reward":reward})
        if perceptions is None:
            perceptions=self.current_state["perception"]
        self.__learn(perceptions,reward)
        self.__log(timestep)

    def get_decision(self):
        return self.decision_fct.last_actions

    def decisions(self,perceptions=None):
        """
        Decides the action for the given perceptions

        Kwargs:
        perceptions: the state of all agents in the population
        """
        if perceptions is None:
            perceptions=self.current_state["perception"]
        self.decision_fct.get_decision(perceptions)
        return self.get_decision()

    def perception(self,perceptions,population=[]):
        """
        Updates the state of the agent based on the current state

        Args:
        perceptions: the perception dictionaries for each agent
        population: the other agents, in case information about them is known
        """
        self.current_state.update({"perception":perceptions})

    def step(self):
        self.decisions()
