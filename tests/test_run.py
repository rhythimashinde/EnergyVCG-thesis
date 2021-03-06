from nego.mediated.Agents_Supervisor import NegoModel
from nego.bilateral.Agents_Supervisor import NegoModel
import unittest

class TestMeasures(unittest.TestCase):
    """
    This class tests the behavior of the model in a simple scenario
    """

    def __init__(self, *args, **kwargs):
        super(TestMeasures, self).__init__(*args, **kwargs)
        self.N=5
        self.s=NegoModel(self.N)
        self.s.threshold=3

    def test_uniform(self):
        measures = self.s.evaluate([1,1,1,1,1],0)
        self.assertEqual(round(measures["gini"],2),0)
        self.assertEqual(measures["efficiency"],1)
        self.assertEqual(round(measures["success"],2),1)
        self.assertEqual(measures["tot_contrib"],5)

    def test_bipolar(self):
        measures = self.s.evaluate([1,1,1,0,0],0)
        self.assertEqual(round(measures["gini"],2),0.4)
        self.assertEqual(measures["efficiency"],0)
        self.assertEqual(round(measures["success"],2),0.6)
        self.assertEqual(measures["tot_contrib"],3)