import unittest
from src import Supervisor
from src.utils import *
import numpy as np

class TestMeasures(unittest.TestCase):
    """
    This class tests the behavior of the model in a simple scenario
    """

    def __init__(self, *args, **kwargs):
        super(TestMeasures, self).__init__(*args, **kwargs)
        self.N=5
        print("fff")
        self.s=Supervisor.BaseSupervisor(self.N)
        self.s.threshold=3

    # def test_mandatory_contrib(self):
    #     # costs and values are all 1
    #     pdict=[{"value":1,"cost":1}]*self.N
    #     s.perception(pdict,0)
    #     # test that all agents contribute
    #     actions=s.decisionFct(pdict)
    #     self.assertTrue(all(np.array(actions)==1)) # everybody contributes

    def test_gini_all_equal(self):
        n=np.random.randint(1,40)
        vals=np.array([n]*10)
        self.assertEqual(round(gini(vals),2),0.0)

    def test_gini_not_all_equal(self):
        n=np.random.randint(1,40)
        vals=np.array([n]*10+[n+1])
        self.assertNotEqual(round(gini(vals),5),0.0)

    def test_gini_all_negative(self):
        n=np.random.randint(1,40)
        vals=np.array([-n]*10)
        self.assertEqual(round(gini(vals),5),0.0)

    def test_gini_all_zero(self):
        vals=np.array([0]*10)
        self.assertEqual(round(gini(vals),2),0.0)

    def test_gini_non_flat(self):
        vals=np.array([[0]*5,[0]*5])
        self.assertEqual(round(gini(vals),2),0.0)

    # def test_all_equal_full_contrib(self):
    #     # costs and values are all 1
    #     pdict=[{"value":1,"cost":1}]*self.N
    #     print(self.s.num_agents)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1]*self.N,0)
    #     self.assertEqual(measures["tot_contrib"],5)
    #     self.assertEqual(measures["tot_cost"],5)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],0.6)
    #     self.assertEqual(measures["welfare"],0)
    #     self.assertEqual(round(measures["gini"],2),0)

    # def test_all_equal_partial_contrib(self):
    #     # costs and values are all 1
    #     pdict=[{"value":1,"cost":1}]*self.N
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1,1,1,0,0],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],3)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],2)
    #     self.assertEqual(round(measures["gini"],2),0.4)

    # def test_all_equal_no_contrib(self):
    #     # costs and values are all 1
    #     pdict=[{"value":1,"cost":1}]*self.N
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([0]*self.N,0)
    #     self.assertEqual(measures["tot_contrib"],0)
    #     self.assertEqual(measures["tot_cost"],0)
    #     self.assertEqual(measures["success"],0)
    #     self.assertEqual(measures["efficiency"],0)
    #     self.assertEqual(measures["welfare"],-5)
    #     self.assertEqual(round(measures["gini"],2),0.0)

    # def test_different_values_full_contrib(self):
    #     # costs are the same, values differ
    #     pdict=[{"value":2,"cost":1}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1]*self.N,0)
    #     self.assertEqual(measures["tot_contrib"],6)
    #     self.assertEqual(measures["tot_cost"],5)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],0.5)
    #     self.assertEqual(measures["welfare"],0)
    #     self.assertEqual(round(measures["gini"],2),0)

    # def test_different_values_lowcontribs(self):
    #     # costs are the same, values differ
    #     pdict=[{"value":2,"cost":1}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([0,0,1,1,1],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],3)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],2)
    #     self.assertEqual(round(measures["gini"],2),0.4)

    # def test_different_values_highcontrib(self):
    #     # costs are the same, values differ
    #     pdict=[{"value":2,"cost":1}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1,1,0,0,0],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],2)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],3)
    #     self.assertEqual(round(measures["gini"],2),0.6)

    # def test_different_costs_full_contrib(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":1,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1]*self.N,0)
    #     self.assertEqual(measures["tot_contrib"],5)
    #     self.assertEqual(measures["tot_cost"],7)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],0.6)
    #     self.assertEqual(measures["welfare"],-2)
    #     self.assertEqual(round(measures["gini"],2),0.23)

    # def test_different_costs_highcost(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":1,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1,1,1,0,0],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],5)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],0)
    #     self.assertEqual(round(measures["gini"],2),0.56)

    # def test_different_costs_lowcosts(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":1,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([0,0,1,1,1],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],3)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],2)
    #     self.assertEqual(round(measures["gini"],2),0.4)

    # def test_all_different_full_contrib(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":2,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1]*self.N,0)
    #     self.assertEqual(measures["tot_contrib"],6)
    #     self.assertEqual(measures["tot_cost"],7)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],0.5)
    #     self.assertEqual(measures["welfare"],-2)
    #     self.assertEqual(round(measures["gini"],2),0.23)

    # def test_all_different_highcost(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":2,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([1,1,1,0,0],0)
    #     self.assertEqual(measures["tot_contrib"],4)
    #     self.assertEqual(measures["tot_cost"],5)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],0.75)
    #     self.assertEqual(measures["welfare"],0)
    #     self.assertEqual(round(measures["gini"],2),0.56)

    # def test_all_different_lowcosts(self):
    #     # values are the same, costs differ
    #     pdict=[{"value":2,"cost":3}]+[{"value":1,"cost":1}]*(self.N-1)
    #     self.s.perception(pdict,0)
    #     measures=self.s.evaluate([0,0,1,1,1],0)
    #     self.assertEqual(measures["tot_contrib"],3)
    #     self.assertEqual(measures["tot_cost"],3)
    #     self.assertEqual(measures["success"],1)
    #     self.assertEqual(measures["efficiency"],1)
    #     self.assertEqual(measures["welfare"],2)
    #     self.assertEqual(round(measures["gini"],2),0.4)
