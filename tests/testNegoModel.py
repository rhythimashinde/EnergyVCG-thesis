import unittest
from nego.bilateral.Agents_Supervisor import NegoModel
from nego.bilateral.MeasurementGen import MeasurementGen
#from nego.mediated.Agents_Supervisor import NegoModel
#from nego.mediated.MeasurementGen import MeasurementGen
import numpy as np

class TestMeasurement(MeasurementGen):
    def get_measurements(self,i):
        # productions, consumptions, tariffs ,social value
        return [i,i+0.1,i+0.2,i+0.3]

class TestNegoModel(unittest.TestCase):
    """
    This class testes the negotiation model
    """

    def __init__(self, *args, **kwargs):
        super(TestNegoModel, self).__init__(*args, **kwargs)
        self.n=np.random.randint(1,40)
        self.m=NegoModel(self.n)

    def test_createagents_addtoscheduler(self):
        """
        Tests that create_agents() correctly creates the agents and adds them to the scheduler
        """
        me=MeasurementGen()
        measurements=[me.get_measurements(i) for i in range(self.n)]
        print(self.n)
        print(measurements)
        decisions=range(self.n)
        print(decisions)
        rewards = self.m.feedback()
        print(rewards)
        self.m.create_agents(measurements,decisions,rewards)
        print (self.m.schedule.agents)
        print (len(self.m.schedule.agents))
        self.assertEqual(len(self.m.schedule.agents),self.n) # all agent are added correctly

    def test_createagents_zeroagents(self):
        """
        Tests that create_agents() correctly creates the agents and adds them to the scheduler
        """
        m=NegoModel(0)          # zero agents
        m.create_agents([],[],[])
        self.assertEqual(len(m.schedule.agents),0) # all agent are added correctly

    def test_initagents_correctvalues(self):
        """
        Tests that init_agent initializes agents correctly
        """
        mfct=TestMeasurement()
        measurements=[mfct.get_measurements(i) for i in range(self.n)] # each measurement contains [i,i+0.1,i+0.2]
        decisions=range(self.n) # each decision is the agent id
        rewards = self.m.feedback()
        agents=self.m.init_agents(measurements,decisions,rewards)
        ms=[[round(a.production,2),round(a.consumption,2),round(a.tariff,2),round(a.social_value,2)] for a in agents] # the measurement
        ds=[a.action for a in agents] # the decisions
        self.assertEqual(ms,measurements) # check that initialization was correct
        self.assertEqual(ds,list(decisions))

    def test_initagents_wrongsize(self):
        """
        Tests that create_agents() correctly creates the agents and adds them to the scheduler
        """
        mfct=TestMeasurement()
        # more measurements than agents
        measurements=[mfct.get_measurements(i) for i in range(2*self.n)] # each measurement contains [i,i+0.1,i+0.2]
        decisions=range(2*self.n)
        rewards=self.m.feedback()
        agents=self.m.init_agents(measurements,decisions,rewards)
        self.assertEqual(len(agents),self.n) # agent have the correct length
        ms=[[a.production,a.consumption,a.tariff,a.social_value] for a in agents] # the measurement
        ds=[a.action for a in agents] # the decisions
        ms1=measurements[:self.n]     # take only the first n
        ds1=list(decisions)[:self.n]     # take only the first n
        self.assertEqual(ms,ms1) # check that initialization was correct
        self.assertEqual(ds,ds1)
        # less measurements than agents
        measurements=[mfct.get_measurements(i) for i in range(self.n-1)] # each measurement contains [i,i+0.1,i+0.2]
        decisions=range(self.n-1)
        agents=self.m.init_agents(measurements,decisions,rewards)
        self.assertEqual(len(agents),self.n) # agent have the correct length
        ms=[[a.production,a.consumption,a.tariff,a.social_value] for a in agents] # the measurement
        ds=[a.action for a in agents] # the decisions
        ms1=ms[:-1]     # take only the first n-1
        ds1=ds[:-1]     # take only the first n-1
        self.assertEqual(ms1,measurements) # check that initialization was correct
        self.assertEqual(ds1,list(decisions))
        self.assertEqual(ms[-1],[0,0,0,0]) # default value of 0 for the last element
        self.assertEqual(ds[-1],0) # default value of 0 for the last element
