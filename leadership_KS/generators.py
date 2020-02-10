
def generate_random_times(lam = 1.0, a = 2.0, tmax = 1000.0):
    """
    Generate two Poisson processes with rates lam and a*lam
    Parameters
    ----------
    lam : float (optional)
        Rate of Poisson process 1.
    a : float (optional)
        Rate of second Poisson process will be a*lam.
    tmax : float (optional)
        Maximum time of the simulated processes.
    Returns
    -------
    times : dictionary of 2 lists of times
        Dictionary with the lists of times of events for each process.
    """
    from random import expovariate
    times = {1:[], 2:[]}
    ids = times.keys()
    lam = 1.0
    l = [lam, a*lam]
    for idi in ids:
        lm = l[idi - 1]
        t = expovariate(lm)
        while t < tmax:
            times[idi].append(t)
            t += expovariate(lm)
    return times

def generate_correlated_times(delta = 2.0, dt = 0.5, tmax = 1000.0):
    """
    Generate one Poisson process of rate 1.0 and for the second one
    the rate increases from 1.0 to 1.0+delta during dt after the
    first one did an event.
    Parameters
    ----------
    delta : float (optional)
        Excess rate for the second process.
    dt : float (optional)
        Time during which the excess rate has impact.
    tmax : float (optional)
        Maximum time of the time series.
    Returns
    -------
    times : dictionary of 2 lists of times
        Dictionary with the lists of times of events for each process.
    """
    from random import expovariate
    times = {1:[], 2:[]}
    ids = times.keys()
    #do first time series as a Poisson process of rate 1
    lam = 1.0
    t = expovariate(lam)
    while t < tmax:
        times[1].append(t)
        t += expovariate(lam)
    #to integrate the second process I use a Poisson process of rate one
    #and do a change of timescales
    #first I have to compute lambda(t) in order to do it only once
    t = 0.0
    tau = expovariate(lam)
    lam_l = lam_det(times[1], dt)
    t = inverted_time_step(tau, t, lam_l, delta, dt)
    while t < tmax:
        times[2].append(t)
        tau = expovariate(lam)
        t += inverted_time_step(tau, t, lam_l, delta, dt)
    return times

def lam_det(times, dt):
    """
    Auxiliary function to know the instantaneous firing rate
    Parameters
    ----------
    times: dictionary of 2 lists of times
        Dictionary with the lists of times of events for each process.
    dt: float
        time during which there is an extra firing rate
    Returns
    -------
    lam_det: list of tuples
    """
    N = len(times)
    lam = [(0, 0)]
    lam.append((times[0], 1))
    for i in range(1, N):
        delta_t = times[i] - times[i - 1]
        if delta_t > dt:
            lam.append((times[i - 1] + dt, 0))
            lam.append((times[i], 1))
    return lam

def inverted_time_step(tau_targ, t, lam, delta, dt):
    '''
    Auxiliary function for the generation of a non-homogeneous Poisson process.
    Parameters
    ----------
    tau_targ : float
        time to invert.
    t : float
        time of previous event.
    lam : float (optional)
        Rate of Poisson process 1.
    delta : float (optional)
        Excess rate for the second process.
    dt : float
        Time during which the excess rate has impact.
    Returns
    -------
    t_good - t: time until next event.
    '''
    ind = 0
    N = len(lam)
    while lam[ind][0] < t and ind < N-1:
        ind += 1
    if ind == N - 1:
        if lam[N - 1][0] < t:
            if lam[N - 1][1] == 0:
                lam_g = [(t, 0)]
            else:
                lam_g = [(t, 1), (t + dt, 0)]
        else:
            lam_g = [(t, lam[ind - 1][1])] + lam[ind:]
    else:
        lam_g = [(t, lam[ind - 1][1])] + lam[ind:]
    N_lam_g = len(lam_g)
    tau = 0.0
    for i in range(N_lam_g - 1):
        dtau = (lam_g[i + 1][0] - lam_g[i][0]) * (1.0 + delta * lam_g[i][1])
        tau += dtau
        if tau > tau_targ:
            t_good = lam_g[i][0] + (tau_targ - tau + dtau) / (1.0 + delta * lam_g[i][1])
            return t_good - t
    t_good = lam_g[N_lam_g - 1][0] + (tau_targ - tau) / (1.0 + delta * lam_g[N_lam_g - 1][1])
    return t_good - t


def generate_hawkes(edges):
    """
    UNDER CONSTRUCTION. Will need the package pyhawkes. Generates firing times
    of N individuals in a given network defined by edges.
    """
    from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
    np.random.seed(1122334455)
    # Create a simple random network with K nodes a sparsity level of p
    # Each event induces impulse responses of length dt_max on connected nodes
    a = set()
    for edge in edges:
        a.add(edge[0])
        a.add(edge[1])
    K = len(a)
    p = 0.25
    dt_max = 20
    network_hypers = {"p": p, "allow_self_connections": False}
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K = K, dt_max = dt_max, network_hypers = network_hypers)
    A2 = np.zeros((K, K))
    for edge in edges:
        A2[edge[0]][edge[1]] = 1.0

    true_model.weight_model.A = A2
    A3 = 0.5*np.ones((K, K))
    true_model.weight_model.W = A3

    Tmax=50000
    S,R = true_model.generate(T = Tmax)


    times = dict()

    for idi in range(K):
        for it in range(Tmax):
            n = S[it][idi]
            if n > 0:
                if idi not in times.keys():
                    times[idi] = []
                else:
                    times[idi].append(it)

    ids = list(times.keys())

    for idn in ids:
        times[idn].sort()

    return ids,times
