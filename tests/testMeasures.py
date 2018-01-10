# provide custom measurements:
# - all costs equal and all values equal
# - all costs equal and different values
# - ...
# and check that measures are computed and saved correctly

import unittest
from Supervisor import BaseSupervisor
from Agent import BaseAgent
from DecisionLogic import *
import numpy as np
from utils import efficiency

class TestMeasures(unittest.TestCase):
    """
    This class implements test scenarios that confirm that the simulation works correctly
    """

    def __init__(self, *args, **kwargs):
        super(TestMeasures, self).__init__(*args, **kwargs)
        N=np.random.randint(1,100)
        self.s=BaseSupervisor(N,decision_fct=DecisionLogicSupervisorTesting,agent_decision_fct=DecisionLogicTesting)
        self.s.decision_fct=DecisionLogicSupervisorTesting(model=self.s) # update supervisor's decision fct

    def test_decentralized_always_contrib(self):
        percs=[{"value":i,"cost":1} for i in range(self.s.N)]
        decs=self.s.decisions(percs)
        self.assertTrue(all([d["contributed"] for d in decs]))
        measures=self.s.evaluate(decs)[0]
        self.assertEqual(measures["tot_contrib"],self.s.N) # everyone contributes

    def test_decentralized_always_contrib_cost(self):
        percs=[{"value":1,"cost":i} for i in range(self.s.N)]
        decs=self.s.decisions(percs)
        measures=self.s.evaluate(decs)[0]
        self.assertEqual(round(measures["cost"],2),round(np.mean(range(self.s.N)),2)) # the same contribution for all

    def test_decentralized_always_contrib_welfare(self):
        n1=np.random.randint(1,100)
        n2=np.random.randint(1,100)
        percs=[{"value":1,"cost":n1} for i in range(self.s.N)] # same cost for each agent
        decs=self.s.decisions(percs)
        rewards=[n2 for i in range(self.s.N)] # same reward for all
        measures=self.s.evaluate(decs,rewards=rewards)[0]
        self.assertEqual(round(measures["social_welfare"],2),n2-n1) # social welfare is the average difference between rewards and costs

    def test_decentralized_always_contrib_gini(self):
        percs=[{"value":1,"cost":i} for i in range(self.s.N)]
        decs=self.s.decisions(percs)
        measures=self.s.evaluate(decs)[0]
        self.assertEqual(round(measures["gini"],2),0) # the same contribution for all

    def test_decentralized_always_contrib_efficiency(self):
        percs=[{"value":1,"cost":i} for i in range(self.s.N)] # everyone contributes 1
        ## all contributors are required, efficiency 1
        decs=self.s.decisions(percs)
        measures=self.s.evaluate(decs,threshold=self.s.N)[0] # the threshold is the number of contributors
        self.assertEqual(round(measures["efficiency"],2),1) # efficiency is 0.5
        ## only half contributors are required, efficiency 0.5
        if self.s.N%2 !=0:
            decs[0].update({"contributed":False}) # make the number of contributions even
        measures=self.s.evaluate(decs,threshold=self.s.N//2)[0] # the threshold is half the number of contributors
        self.assertEqual(round(measures["efficiency"],2),0.5)

    def test_decentralized_always_contrib_success(self):
        percs=[{"value":1,"cost":i} for i in range(self.s.N)] # everyone contributes 1
        ## all contributors are required, efficiency 1
        decs=self.s.decisions(percs)
        n=np.random.randint(1,self.s.N) # pick a random number of contributors
        for i in range(n):
            decs[i].update({"contributed":False})       # does not contribute
        thresh=self.s.N-n
        # just enough contribute
        measures=self.s.evaluate(decs,threshold=thresh)[0] # the threshold is the number of contributors
        self.assertEqual(measures["success"],1)
        # more than enough contribute
        measures=self.s.evaluate(decs,threshold=thresh-1)[0] # the threshold is lower than the number of contributors
        self.assertEqual(measures["success"],1)
        # less than enough contribute
        measures=self.s.evaluate(decs,threshold=thresh+1)[0] # the threshold is hicher than the number of contributors
        self.assertLess(measures["success"],1)              # success is not 1

class DecisionLogicTesting(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions={"contribution":perceptions["value"],"cost":perceptions["cost"],"contributed":True}
        return self.last_actions

class DecisionLogicSupervisorTesting(BaseDecisionLogic):
    """
    Returns a constant decision
    """

    def get_decision(self,perceptions):
        decs=[a.decisions(perceptions[n]) for a,n in zip(self.model.schedule.agents,range(self.model.N))] # call each agent's decision fct with the appropriate perception
        return decs

# class MeasurementGenTesting():
#     def __init__(self):
#         self.counter=0

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         ret=[{"value":self.counter,"timestep":timestep}]*len(population)
#         self.counter+=1
#         return ret
