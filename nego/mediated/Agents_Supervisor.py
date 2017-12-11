from mesa import Agent, Model
from mesa.time import RandomActivation
import pandas as pd
from src.utils import *
from Nego.mediated.MeasurementGen import MeasurementGen
from Nego.mediated.Decisions import DecisionLogic
from Nego.bilateral.Feedback_calculation import feedback

class NegoModel(Model):
    def __init__(self, N):
        super().__init__(N)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        measurements = self.perception()
        decisions = self.decision_fct()
        rewards = self.feedback()
        self.create_agents(measurements,decisions,rewards)
        self.evaluate(decisions,self.schedule.time)

    def perception(self):
        m=MeasurementGen()
        measurements_new=[m.get_measurements(i) for i in range(self.num_agents)]
        return measurements_new

    def decision_fct(self):
        # TODO the mediator should choose the actions, not the agents
        d=DecisionLogic()
        all_actions=[d.chose_action(i) for i in range(self.num_agents)]
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

    def step(self,decisions,rewards,timestep):
        # TODO call the log fct, update perceptions, <rewards
        self.schedule.step(self,decisions,rewards,timestep)

    def evaluate(self,decisions,timestep):
         return dict(gini=gini(decisions),
                          efficiency=efficiency(self.num_agents,tot_contributions(decisions)),
                          success=success(self.num_agents,tot_contributions(decisions)),
                          tot_contrib = tot_contributions(decisions))

    def feedback(self):
        rewards = [feedback.feedbackGen(i) for i in range(self.num_agents)]
        return rewards

    def log(self):
        model = NegoModel(self.num_agents)
        m=model.perception()
        decisions = model.decision_fct()
        rewards = model.feedback()
        agents=model.init_agents(m,decisions,rewards)
        partner = [a.partner for a in agents]
        partner_id = [a.unique_id for a in partner]
        d = [[a.unique_id,a.production,a.consumption,a.tariff,a.t,a.reward,a.state] for a in agents]
        agents_dataframe = pd.DataFrame(data=d,columns=['id','production','consumption','tariff','type','reward','state'])
        agents_dataframe_new = agents_dataframe.assign(partner_id = partner_id)
        return agents_dataframe_new

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions,rewards):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.action = decisions
        self.reward = rewards  # no reward
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]
        self.seller_buyer()
        self.cost = self.transactions() # every transaction leads to a cost for the agent
        self.partner = self.partner_selection()
        self.state = self.update_state(rewards)

    def step(self, model,decisions,rewards,timestep):
        # TODO update the state of agents
        pass

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
                if self.y == "buyer" and a.y == "seller" and self.tariff<=a.tariff: # modify tariff rule
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

    def chose_action(self):
        j = self.partner
        if j.consumption <= self.production:
            self.action = 1  # sell
        if j.production >= self.consumption:
            self.action = 0  # buy

    def decision_fct(self):
        # TODO this function is never called
        return 1

    def feedback(self):
        return 1

    def update_state(self,rewards):
        self.state = rewards+1
        return self.state
