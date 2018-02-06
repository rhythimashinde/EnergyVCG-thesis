import pandas as pd
import numpy as np
from mesa import Model
from src.Agent import *
from src.MeasurementGen import BaseMeasurementGen
from src.DecisionLogic import BaseDecisionLogic
from src.RewardLogic import BaseRewardLogic
from src.EvaluationLogic import BaseEvaluationLogic
from mesa.time import RandomActivation
from src.utils import *

class BaseSupervisor(Model):

    def __init__(self,N,measurement_fct=BaseMeasurementGen,
                 decision_fct=BaseDecisionLogic,
                 agent_decision_fct=BaseDecisionLogic,
                 reward_fct=BaseRewardLogic,
                 evaluation_fct=BaseEvaluationLogic,
                 agent_type=BaseAgent):
        super().__init__(N)
        np.random.seed()
        # parameters
        self.measurement_fct=measurement_fct()
        self.decision_fct=decision_fct(self)
        self.reward_fct=reward_fct(self)
        self.evaluation_fct=evaluation_fct(self)
        self.agent_decision_fct=agent_decision_fct # keep it a class as it will be instanciated in the agent init
        self.agent_type=agent_type
        self.log=[]
        self.N=N
        if self.N<=0:
            raise AssertionError("Initializing empty population")
        self.schedule = RandomActivation(self)
        self.__create_agents()
        self.current_state={"perception":self.perception_dict(0),"reward":[0]*self.N}
        self.perception()       # initialize measurements

    def __create_agents(self):
        self.population=self.__init_population()
        for a in self.population:
            self.schedule.add(a)

    def __init_population(self,population=[],agent_type=None):
        """
        Creates the objects

        Args:
        N: the size of the population

        Kwargs:
        population: a list of agents with which to initialize the population,
        if the parameter is not provided a new population is initialized
        agent_type: the class of agents

        Returns:
        A list of pre-initialized agents
        """
        pop=[]
        if agent_type is None:
            agent_type=self.agent_type
        if population:
            pop=population
        else:
            pop=[agent_type(i, self, decision_fct=self.agent_decision_fct) for i in range(self.N)]
        return pop

    def measurements(self,timestep,population=None):
        """
        Produces the new measurement for each agent for the current round

        Args:
        timestep: the index of the current round

        Kwargs:
        population: the population for which to compute the perceptions. It might be useful to in case perception of a single agent depend on the state of other agents

        Returns:
        A list of dictionaries containing the perception for the current timestep
        """
        if population is None:
            population=self.population
        return self.measurement_fct.get_measurements(population,timestep)

    def decisions(self,perceptions=None):
        """
        Call the model's decision function, which would then query the individual agents if needed

        Kwargs:
        perceptions: the state of all agents in the population

        Returns:
        A list of actions of the same length as the population
        """
        if perceptions is None:
            perceptions=self.current_state["perception"]
        partner = self.partner_set()
        decisions=self.decision_fct.get_decision(perceptions) # call own decision fct
        return decisions

    def partner_set(self):
        return self.decision_fct.get_partner()

    def __learn(self,perceptions,reward):
        """
        Updates the decision function of the supervisor based on the current reward

        Kwargs:
        perceptions: a list of dictionaries containing the state of the system.
        perceptions: a list of dictionaries containing the rewards.
        """
        self.decision_fct.feedback(perceptions,reward) # update state of decision fct

    def perception_dict(self,timestep,population=None):
        """
        Computes a new perception vector for every agent in the population

        Args:
        timestep: the index of the current round

        Kwargs:
        population: the population for which to compute the perceptions. It might be useful to in case perception of a single agent depend on the state of other agents

        Returns:
        A list of dictionaries containing the perception for the current timestep
        """
        measurements=self.measurements(timestep) # obtain new measurements
        pdict=[]
        for i,m in zip(range(len(measurements)),measurements):
            d=m.copy()
            # d.update({"agentID":i})
            # add here any extra information to give to the agents
            pdict.append(d)
        return pdict

    def perception(self,perceptions=None):
        """
        Assigns the new perception to every agent in the population

        Kwargs:
        perceptions: the list of perception dictionaries for each agent
        """
        if perceptions is None:
            perceptions=self.current_state["perception"]
        if not (isinstance(perceptions,list)
           and all([isinstance(i,dict) for i in perceptions])
        ):
        #     # produce default values if lengths mismatch
        #     if(len(perceptions)==self.N):
        #         measurements=perceptions
        #     else:
        #         measurements=list(perceptions)+[{}]*(self.N-len(perceptions)) # add missing default values
        # else:
            raise TypeError("Malformed perception vector")
        for a,p in zip(self.schedule.agents,perceptions):
            a.perception(p)

    def feedback(self,decisions,perceptions=None,rewards=None):
        """
        Gives the feedback to every user

        Args:
        decisions: the list of actions for all agents, their contributions and costs

        Kwargs:
        perceptions: the state of the system
        rewards: a list of rewards, one for each user. If this is not provided, rewards are computed automatically

        Returns:
        A list of rewards, one for each user
        """
        if perceptions is None:
            perceptions=self.current_state["perception"]
        if rewards is None or len(rewards)!=self.N:
            if rewards is not None and len(rewards)!=self.N:
                print("Warning: invalid rewards provided, computing new ones")
            rewards=self.reward_fct.get_rewards(decisions) # recompute rewards
        self.__learn(perceptions,rewards) # feedback to the own decision fct
        # give feedback to the agents
        for a,r in zip(self.schedule.agents,rewards):
            a.feedback(r,self.schedule.steps)
        return rewards


    def run(self,T,params={}):
        """
        Executes the simulation

        Args:
        T: the simulation length
        """
        t=0
        while(t<T):
            self.step()
            self.get_log(params=params)
            t+=1

    def evaluate(self,decisions,rewards=None,threshold=None):
        """
        Computes the measures for this round

        Args:
        decisions: the list of actions for all agents

        Returns:
        A list of measures on the population behavior
        """
        if threshold is None:
            threshold=1
        if rewards is None:
            assert(self.current_state["reward"] is not None)
            rewards=self.current_state["reward"]
        evaluation=self.evaluation_fct.get_evaluation(decisions,rewards,threshold) # call own evaluation fct
        return evaluation

    def get_log(self,params):
        """
        To be called from outside so that the frequency of logging is independent
        """
        dct=self.current_state.copy()
        # convert contents to tables
        for k,v in dct.items():
            if isinstance(v,list) or isinstance(v,np.ndarray):
                if all([isinstance(i,dict) for i in v]):
                    dct[k]=pd.DataFrame(v)
                    for name,val in params.items():
                        dct[k][name]=val # add the extra columns
            # elif isinstance(v,dict):
            #     for name,val in params.items():
            #         dct[k][name]=val # add the extra columns
        agents=[a.log[-1] for a in self.schedule.agents] # get the last log for each agent
        assert(all([a["timestep"]==self.schedule.steps for a in agents]))
        # add extra information to log
        dct.update({"agents":agents})
        dct.update(params)
        self.log.append(dct)
        return dct

    def step(self):
        # update state
        self.current_state.update({"timestep":self.schedule.steps})
        perception=self.perception_dict(self.schedule.steps) # generate measurements
        self.current_state.update({"perception":perception}) # generate measurements
        self.perception()               # communicate them to agents
        self.schedule.step()    # agents decide
        decisions=self.decisions(perception) # collect decisions
        self.current_state.update({"decisions":decisions}) # collect decisions
        self.current_state.update({"reward":self.feedback(decisions)})
        self.current_state.update({"evaluation":self.evaluate(decisions)})
        #self.__log()
