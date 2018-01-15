from src.Agent import BaseAgent
from src.Supervisor import BaseSupervisor
from nego.src.Agent import NegoAgent
from mesa import Model
import operator

class NegoModel(Model):

    def __init_population(self,population=[]):
        super().__init_population(agent_type=NegoAgent)



