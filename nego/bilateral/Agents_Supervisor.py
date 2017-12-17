from mesa import Agent, Model
from mesa.time import RandomActivation
from nego.utilsnego import *
from nego.bilateral.MeasurementGen import MeasurementGen
from nego.bilateral.Decisions import DecisionLogic
from nego.bilateral.Feedback_calculation import feedback
import pandas as pd
import numpy as np

class NegoModel(Model):
    def __init__(self, N):
        super().__init__(N)
        self.num_agents = N
        self.schedule = RandomActivation(self)

    def perception(self):
        """
        Returns:
            allocates the measurements to the model
        """
        m=MeasurementGen()
        measurements_new=[m.get_measurements(i) for i in range(self.num_agents)]
        return measurements_new

    def social_measurements(self,social_values):
        """
        Args:
            social_values: the return value from the perceptions
        Returns:
            gets the values related to social parameters from the perception/ measurements
        """
        return [social_values[i][3] for i in range(self.num_agents)]

    def decision_fct(self):
        """
        Returns:
            list of the decisions of every agent
        """
        all_actions=[i.decision_fct() for i in self.schedule.agents]
        return all_actions

    def create_agents(self,measurements_now,decisions,rewards):
        """
        Args:
            measurements_now: list of measurements
            decisions: the list of decisions
            rewards: the list of rewards
        Returns:
            creates agents with the allocated measurements, decisions and rewards
        """
        agents=self.init_agents(measurements_now,decisions,rewards)
        for a in agents:
            self.schedule.add(a)

    def init_agents(self,measurements_now,decisions,rewards):
        """
        Args:
            measurements_now: list of measurements
            decisions: the list of decisions
            rewards: the list of rewards
        Returns:
            initialises agents with the allocated measurements, decisions and rewards
        """
        # produce default values if lengths mismatch
        if(len(measurements_now)<self.num_agents):
            default_vals=[0]*len(measurements_now[0])
            measurements_now=list(measurements_now)+[default_vals]*(self.num_agents-len(measurements_now))
        if(len(decisions)<self.num_agents):
            default_vals=0
            decisions=list(decisions)+[default_vals]*(self.num_agents-len(decisions))
        ret=[NegoAgent(i, self, m,d,r) for i,m,d,r in zip(range(self.num_agents),measurements_now,decisions,rewards)]
        return ret

    def step(self,decisions,rewards,perceptions,timestep):
        """
        update perceptions, rewards
        Args:
            decisions: the list of all decisions (buy/sell) as per every agent
            rewards: the list of all rewards
            perceptions: the list of all measurements
            timestep: timestep of the model
        Returns: lays down the functions to be called when the model is forwarded by a step
        """
        self.schedule.step(self,decisions,rewards,perceptions,timestep)

    def evaluate(self,decisions,social_measurements,agents_total,ratio,total,costs,rewards,timestep):
        """
        Args:
            agents_total: total agents who want to meet demands (are either seller or buyer)
            ratio: record of every transaction gap/maximum of the consumption and production in transaction
            total: total number of transactions in the round
            timestep: timestep of the model
        Returns:
            the evaluation measures results e.g. efficiency, success and fairness
        """
        return dict(efficiency=efficiency_nego(ratio,total),success=success_nego(agents_total,total),
                    social_welfare=social_welfare(costs,rewards,self.num_agents),
                    fairness=fairness(social_measurements,decisions,self.num_agents),
                    gini=gini(decisions))

    def feedback(self):
        """
        Returns:
            rewards for every agent based on their decisions
        """
        rewards = [feedback.feedbackGen(i) for i in range(self.num_agents)]
        return rewards

    def transactions_all(self):
        return [i.transactions() for i in self.schedule.agents]

    def log_all(self):
        """
        Returns:
            getting agents data and storing int dataframe
        """
        d = [[a.unique_id,a.production,a.consumption,a.tariff,a.t,a.reward,a.partner] for a in self.schedule.agents]
        a_df = pd.DataFrame(data=d,columns=['id','production','consumption','tariff','type','reward','partner'])
        return a_df

    def log(self,full_log):
        """
        Args:
            full_log: gets the complete log from the log function
        Returns:
            appended dataframe with the data of the partners associated to every agent
        """
        a_df_partner = full_log.dropna()  # remove those agents with no partners
        partners = a_df_partner['partner'] # collecting partner data from the dataframe
        partners_id = [x.unique_id for x in partners] # adding the partners' data to every agent e.g. partners' id here
        partners_consumption = [x.consumption for x in partners]
        partners_production = [x.production for x in partners]
        a_df_partner= a_df_partner.assign(partner_id = partners_id,
                                          partners_consumption = partners_consumption,
                                          partners_production = partners_production)
        # further calculations on the partners data
        a_df_partner['gap_seller'] = np.abs(a_df_partner['partners_consumption']-a_df_partner['production'])
        a_df_partner['gap_buyer'] = np.abs(a_df_partner['partners_production']-a_df_partner['consumption'])
        a_df_partner['max_seller'] = a_df_partner[['partners_consumption','production']].max(axis=1)
        a_df_partner['max_buyer'] = a_df_partner[['partners_production','consumption']].max(axis=1)
        a_df_partner['ratio_seller'] = a_df_partner['gap_seller']/a_df_partner['max_seller']
        a_df_partner['ratio_buyer'] = a_df_partner['gap_buyer']/a_df_partner['max_buyer']
        return a_df_partner

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions,rewards):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.action = decisions
        self.reward = rewards
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]
        self.social_value = measurements[3]
        self.t = self.seller_buyer()
        self.cost = self.transactions() # every transaction leads to a cost for the agent
        self.partner = self.partner_selection()

    def step(self, model,decisions,rewards,perceptions,timestep):
        #
        """
        Args:
            model: model (NegoModel)
            timestep: timestep of the model
            perceptions: list of measurements
            decisions: the list of decisions
            rewards: the list of rewards
        Returns:
            updated state of agents
        """
        self.update_state(rewards,decisions,perceptions)

    def seller_buyer(self):
        """
        Returns:
            the seller or buyer role of every agent
        """
        if self.production > self.consumption:
            self.t = "seller"
        if self.production < self.consumption:
            self.t = "buyer"
        #return self.t #comment this out when running test files, keep it for the plots file.

    def partner_selection(self):
        """
        Returns:
            the partner for every agent and their updated consumption, production
        """
        other = self.model.schedule.agents
        for a in other:
            if a != self:  # making sure that the agent doesn't select itself
                if self.t == "buyer" and a.t == "seller" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.consumption <= a.production:
                        a.production = a.production-self.consumption
                        self.consumption = 0
                        # allocate this remaining energy as surplus in second round
                    else:
                        self.consumption = self.consumption-a.production
                        a.production = 0
                    return self.partner
                elif self.t == "seller" and a.t == "buyer" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.production <= a.consumption:
                        a.consumption = a.consumption - self.production
                        self.production = 0
                    else:
                        self.production = self.production - a.consumption
                        a.consumption = 0
                    return self.partner

    def transactions(self):
        """
        Returns:
            total number of transactions
        """
        if self.t == "seller":
            self.cost= 1
        else:
            self.cost= 0
        return self.cost
        # TODO modify the cost here with every transaction

    def decision_fct(self):
        """
        Returns:
            the decision of every agent
        """
        d=DecisionLogic()
        self.action = d.chose_action()
        return self.action

    def feedback(self):
        return 1

    def update_state(self,rewards,decisions,perceptions):
        """
        Args:
            rewards: the list of rewards
            decisions: the list of decisions
            perceptions: the list of measurements
        Returns:
            updated state of every agent after every step
        """
        self.reward = rewards[self.unique_id]+1
        self.production = perceptions[self.unique_id][0]+1
        self.consumption = perceptions[self.unique_id][1]+1
        self.tariff = perceptions[self.unique_id][2]+1