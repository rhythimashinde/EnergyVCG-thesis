import random

class DecisionLogic():
    def __init__(self):
        super().__init__()
        self.choice = 0
        self.partner_selected = None

    def chose_action(self):
        # here add some relation to make choice based on partner selected
            # for i in range(model.num_agents):
            #    self.partner_selected = NegoAgent.partner_selection(i)
            # if other.production > self.consumption:
            #     self.choice = 1  # "buy"
            # if self.production > other.consumption:
            #     self.choice = 2  # "sell"
        return random.choice([1, 0])  # buy or sell