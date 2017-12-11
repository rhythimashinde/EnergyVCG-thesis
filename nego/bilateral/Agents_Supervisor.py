from mesa import Agent, Model
from mesa.time import RandomActivation
from src.utils import *
from Nego.bilateral.MeasurementGen import MeasurementGen
from Nego.bilateral.Decisions import DecisionLogic
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
        # TODO call the log fct, update perceptions, rewards
        self.feedback()
        self.schedule.step(self,decisions,rewards,timestep)

    def evaluate(self,decisions,timestep):
        # Stefano: I am not sure that these measures (as defined) are suitable for the energy scenario... we should discuss about this
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

        self.seller_buyer()
        self.cost = self.transactions() # every transaction leads to a cost for the agent
        #self.partner = self.partner_selection()
        self.state = self.update_state(rewards)

    def step(self, model,decisions,rewards,timestep):
        # TODO update decisions of agents
        self.update_state(rewards)

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

    def decision_fct(self, decision):
        # TODO this function is never called
        d=DecisionLogic()
        self.action = d.chose_action(decision)

    def feedback(self):
        return 1

    def update_state(self,rewards):
        self.state = rewards+1
        return self.state
