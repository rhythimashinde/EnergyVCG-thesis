class BaseSupervisor():

    def __init__(self,N):
        self.population=self.init_population(self.numagents)
        # parameters
        self.numagents=N

    def __init_population(N):
        """
        Creates the objects

        Args:
        N: the size of the population

        Returns:
        A list of pre-initialized agents
        """
        measurements=self.__measurement_gen(0)
        pop=[]
        for i in range(N):
            a=BaseAgent()       # give it the measurement
            pop.append(a)
        raise NotImplementedError("Please Implement this method")
        return pop

    def __measurement_gen(self,timestep):
        """
        Produces the new measurement for the current round

        Args:
        timestep: the index of the current round
        """
        raise NotImplementedError("Please Implement this method")

    def decisionFct(self,perceptions):
        """
        Decide the action for each agent

        Args:
        perceptions: the state of all agents in the population

        Returns:
        A list of actions of the same length as the population
        """
        raise NotImplementedError("Please Implement this method")

    def __learn(self,decisions,rewards):
        """
        Updates the decision function of the supervisor based on the current reward

        Args:
        decisions: the list of actions for all agents
        rewards: the list of rewards for all agents
        """
        raise NotImplementedError("Please Implement this method")

    def perception_dict(self,timestep):
        """
        Computes a new perception vector for every agent in the population

        Args:
        timestep: the index of the current round

        Returns:
        A list of dictionaries containing the perception for the current timestep
        """
        raise NotImplementedError("Please Implement this method")

    def perception(self,perceptions):
        """
        Assigns the new perception to every agent in the population

        Args:
        perceptions: the list of perception dictionaries for each agent
        """
        raise NotImplementedError("Please Implement this method")

    def feedback(self,decisions):
        """
        Computes the feedback for every user

        Args:
        decisions: the list of actions for all agents

        Returns:
        A list of rewards, one for each user
        """
        raise NotImplementedError("Please Implement this method")

    def run(self,T):
        """
        Executes the simulation

        Args:
        T: the simulation length
        """
        t=0
        while(t<T):
            t++
        raise NotImplementedError("Please Implement this method")

    def evaluate(self,decisions,timestep):
        """
        Computes the measures for this round

        Args:
        decisions: the list of actions for all agents
        timestep: the index of the current round

        Returns:
        A list of measures on the population behavior
        """
        raise NotImplementedError("Please Implement this method")
