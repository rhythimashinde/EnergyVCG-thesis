import numpy as np
from scipy.stats import linregress

def efficiency_nego(transaction_gap_ratio,tot_transactions):
    """
    calculates whether the demands are met exactly for the agents during transaction
    Args:
        transaction_gap_ratio: record of every transaction gap/maximum of the consumption and production in transaction
        tot_transactions: total number of transactions in the round

    Returns:
        either 1 if every agent gets their demand satisfied as needed or a fraction as per the demands satisfied
    """
    print("efficiency=",transaction_gap_ratio/tot_transactions)
    return transaction_gap_ratio/tot_transactions

def success_nego(tot_agents,tot_transactions):
    """
    calculates the ratio of number of agents who got allocated to partners
    Args:
        tot_agents: total agents who want to meet demands (are either seller or buyer)
        tot_transactions: total number of transactions in the round

    Returns:
        either 1 if all agents who need to buy/sell gets a partner or a fraction as per the number of agents getting partners
    """
    print("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return (tot_transactions*2-tot_agents)/tot_agents

def fairness(measurements,decisions,N):
    """
    Args:
        decisions: list of decisions associated to every agent
        measurements: list of measurements associated to every agent
    Returns:
        relation coefficient based on the measurements of agents being related to the decisions they take
    """
    if len(measurements)>N:
        measurements = measurements[(len(measurements)-N):]
    if len(decisions)>N:
        decisions = decisions[(len(decisions)-N):]
    assert(len(measurements)==len(decisions))
    slope,intercept,rvalue,pvaue,stderr = linregress(measurements,decisions)
    print("fairness =", slope)
    return linregress(measurements,decisions)

def social_welfare(costs,rewards,N):
    """
    Computes the social welfare for the current round
    Args:
        costs: a list of costs, one for each agent
        rewards: a list of rewards, one for each agent
    Returns:
        the social welfare value
    """

    if len(costs)>N:
        costs = costs[(len(costs)-N):]
    if len(rewards)>N:
        rewards = rewards[(len(rewards)-N):]
    assert(len(costs)==len(rewards))
    print ("social_welfare=",np.mean(np.array(costs)-np.array(rewards)))
    return np.mean(np.array(costs)-np.array(rewards))

def gini(array):
    """Calculate the Gini coefficient of a numpy array.
    https://github.com/oliviaguest/gini/blob/master/gini.py
    based on bottom eq:
    http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from:
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    Args:
        array: decisions' list as an array
    Returns:
        either 1 if everyone is taking decision in the model equally.
    """
    # All values are treated equally, arrays must be 1d:
    array=np.array(array,dtype=np.float64)
    array = array.flatten()
    if len(array)==0:
        return -1
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
