from mesa import Agent, Model
from mesa.time import RandomActivation
from Mesurements import MeasurementGen
from Decisions import DecisionLogic
import random
import utils

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.action = decisions
        self.reward = 0  # no reward
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]
        self.type = self.seller_buyer()
        self.energy = self.production
        self.partner = self.partner_selection()

    def step(self, model,decisions,timestep):
        self.decision_fct(NegoModel.chose_action(model))
        self.feedback(NegoModel.evaluate(model,decisions,timestep))  # this would not be called again separately here

    def seller_buyer(self):
        if self.production > self.consumption:
            self.type = "seller"
        if self.production < self.consumption:
            self.type = "buyer"
        return self.type

    def partner_selection(self):
        other = self.model.schedule.agents
        for a in other:
            if a != self:
                if self.type == "buyer" and a.type == "seller" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.consumption <= a.production:
                        a.energy = a.production-self.consumption
                        a.production = a.energy
                        # allocate this remaining energy as surplus in second round
                    else:
                        self.energy = self.consumption-a.production
                        self.consumption = self.energy
                    return self.partner
                if self.type == "seller" and a.type == "buyer" and self.tariff<=a.tariff: # modify tariff rule
                    self.partner = a
                    if self.production <= a.consumption:
                        a.energy = a.consumption - self.production
                        a.consumption = a.energy
                    else:
                        self.energy = self.production - a.consumption
                        self.production = self.energy
                    return self.partner

    def chose_action(self):
        agents = self.model.schedule.agents
        for i in agents:
            j = i.partner
            if j.energy < i.energy:
                i.action = 1
            else:
                i.action = 0

    def decision_fct(self, decision):
        self.action = DecisionLogic.chose_action(decision)

    def feedback(self, rewards):
        self.reward = NegoModel.feedback(rewards)

class NegoModel(Model):
    def __init__(self, N):
        super().__init__(N)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        m = self.perception()
        decisions = self.chose_action()
        self.create_agents(m,decisions)
        self.evaluate(decisions,self.schedule.time)

    def perception(self):
        measurements_new = [0 for x in range(self.num_agents)]
        for i in range(self.num_agents):
            measurements_new[i] = MeasurementGen.get_measurements(i)
        return measurements_new

    def chose_action(self):
        all_actions = [0]*self.num_agents
        for i in range(self.num_agents):
            all_actions[i] = DecisionLogic.chose_action(i)
        return all_actions

    def create_agents(self,measurements_now,decisions):
        agents=self.init_agents(measurements_now,decisions)
        for a in agents:
            self.schedule.add(a)

    def init_agents(self,measurements_now,decisions):
        ret=[]
        for i in range(self.num_agents):
            ret.append(NegoAgent(i, self, measurements_now[i],decisions[i]))
        return ret

    def step(self,decisions,timestep):
        self.schedule.step(self,decisions,timestep)

    def evaluate(self,decisions,timestep):
        all_reward = dict(gini=utils.gini(decisions),
                          efficiency=utils.efficiency(self.num_agents,utils.tot_contributions(decisions)),
                          success=utils.success(self.num_agents,utils.tot_contributions(decisions)),
                          tot_contrib = utils.tot_contributions(decisions))
        return all_reward

    def feedback(self):
        return random.choice([3, 4])  # make this dependent on the evaluate rewards


