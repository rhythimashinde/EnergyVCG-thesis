from mesa import Agent, Model
from mesa.time import RandomActivation
from Mesurements import MeasurementGen
from Decisions import DecisionLogic
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

    def step(self):
        self.schedule.step(self)

    def evaluate(self,decisions,timestep):
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

def gini(array):
    """Calculate the Gini coefficient of a numpy array.
    https://github.com/oliviaguest/gini/blob/master/gini.py
    based on bottom eq:
    http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from:
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
#    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def contributions(decisions):
    """
    Computes the ratio of volunteering and free riding
    Args:
    decisions: the actions of agents, 1 for volunteers, 0 for free riders
    Returns:
    The proportion of volunteers
    """
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.mean(decisions)

def tot_contributions(decisions):
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.sum(decisions)

def efficiency(thresh,tot_contrib):
    """
    Returns the value of efficiency for one round.
    Similar values of needs and total contributions correspond to high efficiency
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either the ratio between needs and contributions if successful or 0
    """
    return (thresh/tot_contrib) if tot_contrib>=thresh else 0

def success(thresh,tot_contrib):
    """
    Returns the value of success for one round
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either 1 if successful or a fraction corresponding to the needs covered
    """
    assert(thresh>0)
    return (tot_contrib/thresh) if thresh>tot_contrib else 1
