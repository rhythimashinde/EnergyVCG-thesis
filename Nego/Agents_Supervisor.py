from mesa import Agent, Model
from mesa.time import RandomActivation
from Mesurements import MeasurementGen
from Decisions import DecisionLogic
from src.utils import *
import random
import numpy as np

class NegoAgent(Agent):
    def __init__(self,unique_id,model,measurements,decisions):
        super().__init__(unique_id,model)
        self.unique_id = unique_id
        self.partner = None
        self.action = decisions  # no action
        self.reward = 0  # no reward
        self.production = measurements[0]
        self.consumption = measurements[1]
        self.tariff = measurements[2]

    def step(self, model):
        self.decision_fct(NegoModel.chose_action(model))
        self.partner_selection()  # this would not be called again separately here
        self.feedback(NegoModel.evaluate(model))  # this would not be called again separately here

    def partner_selection(self):
        self.partner = random.choice(self.model.schedule.agents)
        # check that it's not selecting itself
        return self.partner

    def decision_fct(self, decision):
        self.action = DecisionLogic.chose_action(decision)

    def feedback(self, rewards):
        self.reward = NegoModel.feedback(rewards)

class NegoModel(Model):
    def __init__(self, N):
        super().__init__(N)
        self.num_agents = N
        self.schedule = RandomActivation(self)
        m=self.perception()
        decisions = self.chose_action()
        self.create_agents(m,decisions)

    def perception(self):
        # measurements_new = [0 for x in range(self.num_agents)]
        # for i in range(self.num_agents):
        #     measurements_new[i] = MeasurementGen.get_measurements(i)
        measurements_new=[MeasurementGen.get_measurements(i) for i in range(self.num_agents)]
        return measurements_new

    def chose_action(self):
        # all_actions = [0]*self.num_agents
        # for i in range(self.num_agents):
        #     all_actions[i] = DecisionLogic.chose_action(i)
        all_actions=[DecisionLogic.chose_action(i) for i in range(self.num_agents)]
        return all_actions

    def create_agents(self,measurements_now,decisions):
        agents=self.init_agents(measurements_now,decisions)
        for a in agents:
            self.schedule.add(a)

    def init_agents(self,measurements_now,decisions):
        # ret=[]
        # for i in range(self.num_agents):
        #     ret.append(NegoAgent(i, self, measurements_now[i],decisions[i]))
        ret=[NegoAgent(i, self, m,d) for i,m,d in zip(range(self.num_agents),measurements_now,decisions)]
        return ret

    def step(self):
        self.schedule.step(self)
        log=self.evaluate()
        # now you can log the results to file

    def evaluate(self):
        decisions=[a.action for a in self.schedule.agents]
        timestep=self.schedule.time
        all_reward = dict(gini=gini(decisions),
                          efficiency=efficiency(self.num_agents,tot_contributions(decisions)),
                          success=success(self.num_agents,tot_contributions(decisions)),
                          tot_contrib = tot_contributions(decisions))
        return all_reward

    def feedback(self):
        return random.choice([3, 4])  # make this dependent on the evaluate rewards

    def run_model(self, steps):
        for _ in range(steps):
            self.step()
