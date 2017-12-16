def efficiency_nego(transaction_gap_ratio,tot_transactions):
    """
    calculates whether the demands are met exactly for the agents during transaction
    Args:
        transaction_gap_ratio: record of every transaction gap/maximum of the consumption and production in transaction
        tot_transactions: total number of transactions in the round

    Returns:
    either 1 if every agent gets their demand satisfied as needed or a fraction as per the demands satisfied
    """
    print ("efficiency=",transaction_gap_ratio/tot_transactions)
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
    print ("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return (tot_transactions*2-tot_agents)/tot_agents

# TODO gini, costs, social_welfare, fairness
