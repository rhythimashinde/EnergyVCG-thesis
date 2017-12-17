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

    def test_uniform(self):
        m=self.s.perception()
        decisions = self.s.decision_fct()
        rewards = self.s.feedback()
        perceptions = self.s.perception()
        social_measurements = self.s.social_measurements(perceptions)
        costs = self.s.transactions_all()
        self.s.create_agents(m,decisions,rewards)
        self.s.step(decisions,rewards,perceptions,0)
        full_log = self.s.log_all()
        agents_total = full_log.shape[0]
        ratio = self.s.log(full_log)['ratio_seller'].sum()
        total = self.s.log(full_log).shape[0]
        if ratio != 0:
            measures = self.s.evaluate(decisions,social_measurements,agents_total,ratio,total,rewards,costs,0)
            self.assertEqual(round(measures["gini"],2),0.4)
            self.assertEqual(round(measures["efficiency"],2),0.47)
            self.assertEqual(round(measures["success"],2),0)
            self.assertEqual(measures["fairness"],0.25)
