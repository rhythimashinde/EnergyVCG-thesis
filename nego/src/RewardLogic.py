import numpy as np
from src.RewardLogic import BaseRewardLogic
class NegoRewardLogic(BaseRewardLogic):

    def __init__(self,model):
        np.random.seed()
        self.model=model

    def get_rewards(self,decisions):
        """
        Returns a list of dictionaries containing the reward (float) for each agent
        """
        return [{"reward":0}]*len(decisions)
