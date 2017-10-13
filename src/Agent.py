class BaseAgent():
    """
    Base agent
    """
    def __init__(self,id_):
        """
        create agent
        """
        # parameters
        self.ID=id_

    def __log(self,timestep):
        """
        Save params to file

        Args:
        timestep: the index of the current round
        """
        raise NotImplementedError("Please Implement this method")

    def __learn(self,feedback):
        """
        Update decision process.
        Private, called from within feedback

        Args:
        feedback: the reward obtained in this round
        """
        raise NotImplementedError("Please Implement this method")

    def feedback(self,decisions,partner=None):
        """
        Updates own state according to feedback obtained this round

        Args:
        decisions: the actions of all other agents
        partner: the agent partner in the transaction
        """
        raise NotImplementedError("Please Implement this method")

    def decision_fct(self,*args):
        """
        Implements the decision logic
        """
        raise NotImplementedError("Please Implement this method")

    def perception(self,perceptions,population=[]):
        """
        Updates the state of the agent based on the current state

        Args:
        perceptions: the perception dictionaries for each agent
        population: the other agents, in case information about them is known
        """
        raise NotImplementedError("Please Implement this method")
