import numpy as np
from scipy.stats import linregress
import pandas as pd
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

def efficiency_nego(transaction_gap_ratio,tot_transactions):
    """
    calculates whether the demands are met exactly for the agents during transaction
    Args:
        transaction_gap_ratio: total energy traded
        tot_transactions: total energy available for trading with the sellers

    Returns:
        either 1 if every agent gets their demand satisfied as needed or a fraction as per the demands satisfied
    """
    # print("total",tot_transactions)
    # print("traded",transaction_gap_ratio)
    # print("difference",tot_transactions-transaction_gap_ratio)
    if tot_transactions != 0:
        return (tot_transactions - transaction_gap_ratio)/tot_transactions
    else:
        return 0

def success_nego(tot_agents,tot_transactions):
    """
    calculates the ratio of number of agents who got allocated to partners
    Args:
        tot_agents: total agents who want to meet demands (are either seller or buyer)
        tot_transactions: total number of transactions in the round

    Returns:
        either 1 if all agents who need to buy/sell gets a partner or a fraction
        as per the number of agents getting partners
    """
    #print("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return (tot_transactions*2-tot_agents)/tot_agents

def market_access(tot_agents_poor,tot_transactions_poor):
    """
    calculates the ratio of number of agents who got allocated to partners
    Args:
        tot_agents: total poor agents who want to meet demands (are either seller or buyer)
        tot_transactions: total number of transactions in the round for poor agents

    Returns:
        either 1 if all agents who need to buy/sell gets a partner or a fraction
        as per the number of agents getting partners
    """
    #print("success=",(tot_transactions*2-tot_agents)/tot_agents)
    return (tot_transactions_poor*2-tot_agents_poor)/tot_agents_poor

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
    #print ("social_welfare=",np.mean(np.array(costs)-np.array(rewards)))
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

def compute_stats(data,idx=False,columns=False,drop_count=True):
    """
    Computes statistics (mean,std,confidence interval) for the given columns

    Args:
    data_: a data frame

    Kwargs:
    idx: a list of indexes on which to group, must be a list of valid column names.
        By default the index of the dataframe is used.
    columns: the columns to aggregate, must be a list of valide column names.
        By default all columns are considered

    Returns:
    A data frame with columns 'X_mean', 'X_std' and 'X_ci' containing the statistics for each column name X in 'columns'
    """
    data_=data.copy()
    assert(not idx or isinstance(idx,list))
    assert(not columns or isinstance(columns,list))
    if isinstance(data_,list):
        data_=pd.concat(data_,copy=False) # join all files
    if not idx:
        idx=data_.index
        idx_c=[]
    else:
        idx_c=idx
    if not columns:
        columns=list(data_.columns[np.invert(data_.columns.isin(idx_c))])
    data_["count"]=1
    aggregations={c:[np.mean,np.std] for c in columns if c in data_._get_numeric_data().columns} # compute mean and std for each column
    aggregations.update({"count":np.sum})                # count samples in every bin
    data_=data_[columns+["count"]+idx_c].groupby(idx,as_index=False).agg(aggregations)
    # flatten hierarchy of col names
    data_.columns=["_".join(col).strip().strip("_") for col in data_.columns.values] # rename
    # compute confidence interval
    for c in columns:
        data_[c+"_ci"]=data_[c+"_std"]*1.96/np.sqrt(data_["count_sum"])
    if drop_count:
        data_.drop("count_sum",1,inplace=True)
    return data_

def get_stats(log,varname,idx=["timestep"],cols=None):
    """
    Log: a list of dictionaries or data frames
    """
    df=[pd.DataFrame(i[varname]) for i in log]
    return compute_stats(df,idx=idx,columns=cols)

def plot_trend(df,xname,filename,trends=None):
    if trends is None:
        trends=[d[:-5] for d in df.columns if ("_mean" in d)]
    fig,ax=plt.subplots()
    x=df[xname]
    ax.set_xlabel(xname)
    #fig.suptitle(title)
    #ax.set_ylabel(ylab or str(y))
    # if ylim:
    #     ax.set_ylim(ylim)
    for y in trends:
        ax.plot(x,df[y+"_mean"],label=y)
        ax.fill_between(x,np.asarray(df[y+"_mean"])-np.asarray(df[y+"_ci"]),
                        np.asarray(df[y+"_mean"])+np.asarray(df[y+"_ci"]),alpha=0.2)
    fig.legend()
    fig.savefig(filename,format='png')
    plt.close(fig)

def plot_measures(df,xname,filename,trends=None):
    fig=plt.figure()
    for measures,ylim,i in [[["gini","success","efficiency"],[0,1],0],
                            [["social_welfare"],None,1]]:
        ax = fig.add_subplot(121+i)
        x=df[xname]
        ax.set_xlabel(xname)
    #fig.suptitle(title)
    #ax.set_ylabel(ylab or str(y))
    # if ylim:
    #     ax.set_ylim(ylim)
        for y in measures:
            ax.plot(x,df[y+"_mean"],label=y)
            ax.fill_between(x,np.asarray(df[y+"_mean"])-np.asarray(df[y+"_ci"]),
                            np.asarray(df[y+"_mean"])+np.asarray(df[y+"_ci"]),alpha=0.2)
        ax.legend()
    fig.savefig(filename,format='png')
    plt.close(fig)

def expandgrid(dct):
    """
    dct: A dictionary where the keys are variables and the values are lists of values for these variables
    Returns: A dataframe where the columns are the variables and the rows contain combinations of values
    """
    return pd.DataFrame(list(itertools.product(*dct.values())),columns=list(dct.keys()))
