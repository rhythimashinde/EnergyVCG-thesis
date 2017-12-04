import random

class DecisionLogic():
    def __init__(self):
        super().__init__()
        self.choice = 0
        self.partner_selected = None

    def chose_action(self):
        # agents = NegoAgent.model.schedule.agents
        # for i in agents:
        #     j = i.partner
        #     if j.energy < i.energy:
        #         i.action = 1
        #     else:
        #         i.action = 0
        return random.choice([1, 0])  # buy or sell