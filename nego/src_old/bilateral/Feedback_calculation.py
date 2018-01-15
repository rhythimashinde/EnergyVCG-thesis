import random

class feedback():
    def __init__(self):
        self.rewards_all = 0

    def feedbackGen(self):
        return random.choice([3, 4])  # make this dependent on the evaluate rewards