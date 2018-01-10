import unittest
from Supervisor import BaseSupervisor
from Agent import BaseAgent
from mesa.time import RandomActivation
import numpy as np
from MeasurementGen import *
from DecisionLogic import *
from RewardLogic import *
from EvaluationLogic import *

class TestBaseSupervisor(unittest.TestCase):
    """
    This class testes the base supervisor
    """

    def __init__(self, *args, **kwargs):
        super(TestBaseSupervisor, self).__init__(*args, **kwargs)
        self.N=np.random.randint(1,100)
        #self.N=5
        self.s=BaseSupervisor(self.N,evaluation_fct=EmptyEvaluationLogic)

    def test_init_population(self):
        """
        Test that the population is initialized correctly
        """
        self.assertIsInstance(self.s.schedule,RandomActivation) # test that the scheduler is initialized
        self.assertEqual(len(self.s.schedule.agents),self.N) # test that the population has the right length
        ids=[]
        for a in self.s.schedule.agents:
            self.assertIsInstance(a,BaseAgent) # test that agents have the right type
            self.assertNotEqual(a.decision_fct,None)
            ids.append(a.unique_id)
        self.assertTrue(len(np.unique(ids)),self.N) # test that agents are different instances

    def test_init_emptypopulation_fail(self):
        """
        Test that the program fails if the population is initialized empty
        """
        N=0
        self.assertRaises(AssertionError,lambda: BaseSupervisor(N))

    def test_init_measurements(self):
        class TestMeasurementGen(BaseMeasurementGen):
            def get_measurements(self,pop,i):
                return [{1:2}]*len(pop)
        self.assertNotEqual(self.s.measurement_fct,None)
        self.s.measurement_fct=TestMeasurementGen()
        measurements=self.s.measurements(0)
        self.assertEqual(len(measurements),self.N)
        for i in measurements:
            self.assertEqual(i,{1:2})

    def test_perceptionDict(self):
        """
        Test that perception dictionaries are generated correctly
        """
        T=np.random.randint(1,100)
        pdict=self.s.perception_dict(T)
        self.assertIsInstance(pdict,list)
        self.assertEqual(len(pdict),self.N)
        for i,d in zip(range(len(pdict)),pdict):
            self.assertIsInstance(d,dict)
            self.assertEqual(d["agentID"],i) # update is correct

    def test_perception(self):
        """
        Test how perceptions are given to the agents
        """
        T=np.random.randint(1,100)
        perceptions=self.s.perception_dict(T) # generate measurements
        self.s.perception(perceptions)
        for i,a in zip(range(self.N),self.s.schedule.agents):
            val=a.current_state["perception"]
            self.assertEqual(val["agentID"],i)
            self.assertEqual(val["timestep"],T)

    def test_update_measurement(self):
        """
        Tests that the measurements are updated at each step
        """
        self.s.measurement_fct=MeasurementGenTesting()
        self.s.step()
        ms=self.s.current_state["perception"]
        self.assertTrue(all([m["value"]==0 for m in ms]))
        self.s.step()
        self.s.step()
        ms=self.s.current_state["perception"]
        self.assertTrue(all([m["value"]==2 for m in ms]))

    def test_update_measurements_agents(self):
        """
        Tests that the perceptions of all agents are updated at each step
        """
        self.s.measurement_fct=MeasurementGenTesting()
        self.s.step()
        ms=[a.current_state["perception"] for a in self.s.schedule.agents]
        self.assertTrue(all([m["value"]==0 for m in ms]))
        self.s.step()
        self.s.step()
        ms=[a.current_state["perception"] for a in self.s.schedule.agents]
        self.assertTrue(all([m["value"]==2 for m in ms]))

    def test_update_feedback(self):
        """
        Tests that the decision function obtains the feedback after each step
        """
        self.s.decision_fct=DecisionLogicTesting() # update the decision logic
        self.s.measurement_fct=MeasurementGenTesting()
        self.s.step()
        self.assertNotEqual(self.s.decision_fct.last_actions,None) # actions have been updated
        self.assertEqual(self.s.current_state["perception"],self.s.decision_fct.values["p"]) # perception has been updated
        self.assertEqual(self.s.current_state["reward"],self.s.decision_fct.values["r"]) # rewards have been updated

    def test_decisions(self):
        """
        Test that the supervisor collects decisions correctly from the agents
        """
        for a in self.s.schedule.agents:
            a.decision_fct=DecisionLogicTestingDynamic(a) # update decision fcts of agents
        self.s.decision_fct=DecisionLogicSupervisorTesting(self.s) # update supervisor's decision fct
        percs=[{j:j for j in range(i)} for i in range(len(self.s.schedule.agents))]        # create perceptions
        decs=self.s.decisions(percs)
        self.assertEqual(len(decs["agents"]),self.s.N)
        self.assertEqual(decs["agents"],list(range(self.s.N)))
        # check that the decision is correctly updated after every timestep
        self.s.step()
        self.s.step()
        decs=self.s.decisions(percs)
        self.assertEqual(decs["agents"],list(range(self.s.N))) # actions do not change
        self.assertEqual(decs["timestep"],2)                      # timestamp is updated

class DecisionLogicTesting(BaseDecisionLogic):

    def __init__(self):
        self.values={}

    def feedback(self,perceptions,reward):
        self.values={"p":perceptions,"r":reward}

class DecisionLogicTestingDynamic(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        self.last_actions=len(perceptions)
        return self.last_actions

class DecisionLogicSupervisorTesting(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        decs={"timestep":self.model.schedule.steps,"agents":[a.decisions(perceptions[n]) for a,n in zip(self.model.schedule.agents,range(self.model.N))]} # call each agent's decision fct with the appropriate perception
        return decs


class MeasurementGenTesting():
    def __init__(self):
        self.counter=0

    def get_measurements(self,population,timestep):
        """
        Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
        """
        ret=[{"value":self.counter,"timestep":timestep,"agentID":i} for i in range(len(population))]
        self.counter+=1
        return ret

class EmptyEvaluationLogic(BaseEvaluationLogic):
    def get_evaluation(self,decisions,rewards,threshold):
        return []

class TestBaseMeasurementGen(unittest.TestCase):
    """
    This class testes the base measurement generation
    """
    def __init__(self, *args, **kwargs):
        super(TestBaseMeasurementGen, self).__init__(*args, **kwargs)
        self.N=np.random.randint(1,100)
        self.s=BaseSupervisor(self.N)

    def test_dict(self):
        """
        Tests that the measurement dictionary is properly formatted
        """
        a=BaseMeasurementGen()
        measurements=a.get_measurements(self.s.population,0)
        self.assertIsInstance(measurements,list)
        self.assertEqual(len(measurements),self.N)
        for m in measurements:
            self.assertIsInstance(m,dict)
            self.assertEqual(len(m),len(measurements[0])) # all of the same lenght
