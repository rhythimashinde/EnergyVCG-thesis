from mesa import Agent, Model
from mesa.time import RandomActivation
from src.utils import *
from nego.bilateral.MesurementGen import MeasurementGen
from nego.bilateral.Decisions import DecisionLogic
from nego.bilateral.Feedback_calculation import feedback

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
        measurements_new=[MeasurementGen.get_measurements(i) for i in range(self.num_agents)]
        return measurements_new

    def decision_fct(self):
        all_actions=[NegoAgent.decision_fct(i) for i in range(self.num_agents)]
        return all_actions

    def create_agents(self,measurements_now,decisions,rewards):
        agents=self.init_agents(measurements_now,decisions,rewards)
        for a in agents:
            self.schedule.add(a)

    def init_agents(self,measurements_now,decisions,rewards):
        ret=[NegoAgent(i, self, m,d,r) for i,m,d,r in zip(range(self.num_agents),measurements_now,decisions,rewards)]
        return ret

    def step(self,decisions,rewards,timestep):
        self.feedback()
        self.schedule.step(self,decisions,rewards,timestep)

    def evaluate(self,decisions,timestep):
         return dict(gini=gini(decisions),
                          efficiency=efficiency(self.num_agents,tot_contributions(decisions)),
                          success=success(self.num_agents,tot_contributions(decisions)),
                          tot_contrib = tot_contributions(decisions))

    def feedback(self):
        rewards = [feedback.feedbackGen(i) for i in range(self.num_agents)]
        return rewards

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions,rewards):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.action = decisions
        self.reward = rewards  # no reward
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]
        self.type = self.seller_buyer()
        self.cost = self.transactions() # every transaction leads to a cost for the agent
        self.partner = self.partner_selection()
        self.state = self.update_state(rewards)

    def step(self, model,decisions,rewards,timestep):
        self.update_state(rewards)

    def seller_buyer(self):
        if self.production > self.consumption:
            self.type = "seller"
        if self.production < self.consumption:
            self.type = "buyer"
        return self.type

    def partner_selection(self):
        other = self.model.schedule.agents
        # self.read_file()
        for a in other:
            if a != self:  # making sure that the agent doesn't select itself
                if self.type == "buyer" and a.type == "seller" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.consumption <= a.production:
                        a.production = a.production-self.consumption
                        self.consumption = 0
                        # allocate this remaining energy as surplus in second round
                    else:
                        self.consumption = self.consumption-a.production
                        a.production = 0
                    return self.partner
                elif self.type == "seller" and a.type == "buyer" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.production <= a.consumption:
                        a.consumption = a.consumption - self.production
                        self.production = 0
                    else:
                        self.production = self.production - a.consumption
                        a.consumption = 0
                    return self.partner

    def transactions(self):
        if self.type == "seller":
            return 1  # modify this cost with every transaction

    def decision_fct(self):
        # j = self.partner
        # if j.consumption <= self.production:
        #     self.action = 1  # sell
        # if j.production >= self.consumption:
        #     self.action = 0  # buy
        return DecisionLogic.chose_action(self)

    def feedback(self):
        return 1

    def update_state(self,rewards):
        self.state = rewards+1
        return self.state
