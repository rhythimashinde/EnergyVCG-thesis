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
        actions = [a.current_state["action"] for a in self.model.schedule.agents]
        rewards = [0]*len(actions)
        for i in range(len(actions)):
            rewards[i] = actions[i]*2 # seller has double rewards than buyer
        #print(actions,rewards)
        return [{"agentID":d["agentID"],"reward":r} for d,r in zip(decisions, rewards)]
