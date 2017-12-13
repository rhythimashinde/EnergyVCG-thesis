from mesa import Agent, Model
from mesa.time import RandomActivation
from src.utils import *
from nego.bilateral.MeasurementGen import MeasurementGen
from nego.bilateral.Decisions import DecisionLogic
from nego.bilateral.Feedback_calculation import feedback
import pandas as pd

class NegoModel(Model):
    def __init__(self, N):
        super().__init__(N)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        measurements = self.perception()
        decisions = self.decision_fct()
        rewards = self.feedback()
        self.evaluate(decisions,self.schedule.time)

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
        # self.log(decisions,rewards,perceptions,timestep) # removed from here included in the plot file, can be updated by including here if needed
        self.schedule.step(self,decisions,rewards,perceptions,timestep)

    def evaluate(self,decisions,timestep):
        # Stefano: I am not sure that these measures (as defined) are suitable for the energy scenario... we should discuss about this
         return dict(gini=gini(decisions),
                          efficiency=efficiency(self.num_agents,tot_contributions(decisions)),
                          success=success(self.num_agents,tot_contributions(decisions)),
                          tot_contrib = tot_contributions(decisions))

    def feedback(self):
        rewards = [feedback.feedbackGen(i) for i in range(self.num_agents)]
        return rewards

    def log(self):
        d = [[a.unique_id,a.production,a.consumption,a.tariff,a.t,a.reward] for a in self.schedule.agents]
        agents_dataframe = pd.DataFrame(data=d,columns=['id','production','consumption','tariff','type','reward'])
        #agents_dataframe.to_csv("out_log["+str(i+1)+"].csv",index=False)
        return agents_dataframe

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