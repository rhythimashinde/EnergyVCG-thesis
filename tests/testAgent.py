import unittest
from src.Supervisor import BaseSupervisor
from src.Agent import BaseAgent
from src.DecisionLogic import *
import numpy as np

class TestBaseAgent(unittest.TestCase):
    """
    This class testes the base supervisor
    """

    def __init__(self, *args, **kwargs):
        super(TestBaseAgent, self).__init__(*args, **kwargs)
        self.N=np.random.randint(1,100)
        self.s=BaseSupervisor(self.N)
        self.a=BaseAgent(0,self.s)

    def test_log(self):
        """
        Tests that the log is updated correctly
        """
        self.a.decision_fct=DecisionLogicTesting(self.a)
        self.a.perception({"1":1,"2":2})
        self.a.decisions()      # init the action
        self.a.feedback("rew",0) # give some feedback
        reference={"perception":{"1":1,"2":2},"action":"act","reward":"rew","timestep":0}
        self.assertEqual(reference,self.a.log[0])
        # next step
        self.a.perception({"1":4,"2":5})
        self.a.feedback("rew3",3)
        reference3={"perception":{"1":4,"2":5},"action":"act","reward":"rew3","timestep":3}
        self.assertEqual(reference,self.a.log[0]) # same as before
        self.assertEqual(reference3,self.a.log[1]) # new step

    def test_decision(self):
        """
        Tests that the decision is updated correctly
        """
        self.a.decision_fct=DecisionLogicTestingDynamic(self.a)
        n=np.random.randint(1,20)
        perc={i:"a" for i in range(n)}
        dec=self.a.decisions(perc)
        self.assertEqual(len(perc),dec)

class DecisionLogicTesting(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions="act"
        return self.last_actions

class DecisionLogicTestingDynamic(BaseDecisionLogic):
    """
    Returns a decision that depends on the perception length
    """
    def get_decision(self,perceptions):
        self.last_actions=len(perceptions)
        return self.last_actions
