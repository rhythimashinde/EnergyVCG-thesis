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
        m=MeasurementGen()
        measurements_new=[m.get_measurements(i) for i in range(self.num_agents)]
        return measurements_new

    def decision_fct(self):
        # d=DecisionLogic()
        all_actions=[i.decision_fct() for i in self.schedule.agents]
        return all_actions

    def create_agents(self,measurements_now,decisions,rewards):
        agents=self.init_agents(measurements_now,decisions,rewards)
        for a in agents:
            self.schedule.add(a)

    def init_agents(self,measurements_now,decisions,rewards):
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
        # update perceptions, rewards
        self.feedback()
        # self.log(decisions,rewards,perceptions,timestep)
        # removed from here included in the plot file, can be updated by including here if needed
        self.schedule.step(self,decisions,rewards,perceptions,timestep)

    def evaluate(self,agents_total,ratio,total,timestep):
        return dict(efficiency=efficiency_nego(ratio,total),
                    success = success_nego(agents_total,total))

    def feedback(self):
        rewards = [feedback.feedbackGen(i) for i in range(self.num_agents)]
        return rewards

    def log_all(self):
        #getting agent data
        d = [[a.unique_id,a.production,a.consumption,a.tariff,a.t,a.reward,a.partner] for a in self.schedule.agents]
        a_df = pd.DataFrame(data=d,columns=['id','production','consumption','tariff','type','reward','partner'])
        return a_df

    def log(self,full_log):
        a_df_partner = full_log.dropna()# remove those agents with no partners
        # getting partner data
        partners = a_df_partner['partner']
        partners_id = [x.unique_id for x in partners]
        partners_consumption = [x.consumption for x in partners]
        partners_production = [x.production for x in partners]
        a_df_partner= a_df_partner.assign(partner_id = partners_id,
                                          partners_consumption = partners_consumption,
                                          partners_production = partners_production)
        # further calculations on the data
        a_df_partner['gap_seller'] = np.abs(a_df_partner['partners_consumption']-a_df_partner['production'])
        a_df_partner['gap_buyer'] = np.abs(a_df_partner['partners_production']-a_df_partner['consumption'])
        a_df_partner['max_seller'] = a_df_partner[['partners_consumption','production']].max(axis=1)
        a_df_partner['max_buyer'] = a_df_partner[['partners_production','consumption']].max(axis=1)
        a_df_partner['ratio_seller'] = a_df_partner['gap_seller']/a_df_partner['max_seller']
        a_df_partner['ratio_buyer'] = a_df_partner['gap_buyer']/a_df_partner['max_buyer']
        return a_df_partner
        # if the whole dataframe is needed
        # return agents_dataframe

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions,rewards):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.action = decisions
        self.reward = rewards  # no reward
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]
        self.t = self.seller_buyer()
        self.cost = self.transactions() # every transaction leads to a cost for the agent
        self.partner = self.partner_selection()

    def step(self, model,decisions,rewards,perceptions,timestep):
        # update decisions of agents is called in the model
        self.update_state(rewards,decisions,perceptions)

    def seller_buyer(self):
        if self.production > self.consumption:
            self.t = "seller"
        if self.production < self.consumption:
            self.t = "buyer"
        return self.t

    def partner_selection(self):
        other = self.model.schedule.agents
        # self.read_file()
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
        if self.t == "seller":
            return 1  # modify this cost with every transaction

    def decision_fct(self):
        # this function is called now for bilateral and not mediated.
        d=DecisionLogic()
        self.action = d.chose_action()

    def feedback(self):
        return 1

    def update_state(self,rewards,decisions,perceptions):
        self.reward = rewards[self.unique_id]+1
        self.production = perceptions[self.unique_id][0]+1
        self.consumption = perceptions[self.unique_id][1]+1
        self.tariff = perceptions[self.unique_id][2]+1