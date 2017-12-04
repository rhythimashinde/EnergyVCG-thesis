import numpy as np

def gini(array):
    """Calculate the Gini coefficient of a numpy array.
    https://github.com/oliviaguest/gini/blob/master/gini.py
    based on bottom eq:
    http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from:
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    # array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def contributions(decisions):
    """
    Computes the ratio of volunteering and free riding
    Args:
    decisions: the actions of agents, 1 for volunteers, 0 for free riders
    Returns:
    The proportion of volunteers
    """
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.mean(decisions)

def tot_contributions(decisions):
    assert(all(np.logical_or(np.array(decisions)==1,np.array(decisions)==0))) # either 0 or 1
    return np.sum(decisions)

def efficiency(thresh,tot_contrib):
    """
    Returns the value of efficiency for one round.
    Similar values of needs and total contributions correspond to high efficiency
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either the ratio between needs and contributions if successful or 0
    """
    return (thresh/tot_contrib) if tot_contrib>=thresh else 0

def success(thresh,tot_contrib):
    """
    Returns the value of success for one round
    Args:
    thresh: the needs
    tot_contrib: the sum of contributions
    Returns: either 1 if successful or a fraction corresponding to the needs covered
    """
    assert(thresh>0)
    return (tot_contrib/thresh) if thresh>tot_contrib else 1