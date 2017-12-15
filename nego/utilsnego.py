import numpy as np

def efficiency_nego(transaction_gap_ratio,tot_transactions):
    # calculates whether the demands are met for the agents during transaction
    # Arguments:
    # transaction_gap_ratio: record of every transaction gap/maximum of the consumption and production in transaction
    # tot_transactions: total number of transactions in the round
    print ("efficiency=",transaction_gap_ratio/tot_transactions)
    return transaction_gap_ratio/tot_transactions

def success_nego(tot_agents,tot_transactions):
    # calculates the number of agents who got partners
    # Arguments:
    # tot_agents: total agents who want to meet demands (are either seller or buyer)
    # tot_transactions: total number of transactions in the round
    # Returns: either 1 if successful or a fraction corresponding to the needs covered
    print ("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return ((tot_transactions*2-tot_agents)/tot_agents)

# TODO gini, costs, social_welfare, discrimination
