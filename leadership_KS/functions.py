import numpy as np
import datetime

def ks_2samp(data1, data2):
    """
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution. It is an asymetric version.
    Parameters
    ----------
    a, b : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different
    Returns
    -------
    d, D : float
        KS statistic
    prob, p-value : float
        two-tailed p-value
    tau : same type as data
        value of data at which the two cumulative distributions have larger difference
    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample K-S test, the distribution is
    assumed to be continuous.
    This is the two-sided test, one-sided tests are not implemented.
    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.
    If the K-S statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.
    Examples
    --------
    >>> from scipy import stats
    >>> np.random.seed(12345678)  #fix random seed to get the same result
    >>> n1 = 200  # size of first sample
    >>> n2 = 300  # size of second sample
    For a different distribution, we can reject the null hypothesis since the
    pvalue is below 1%:
    >>> rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
    >>> rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
    >>> stats.ks_2samp(rvs1, rvs2)
    (0.20833333333333337, 4.6674975515806989e-005)
    For a slightly different distribution, we cannot reject the null hypothesis
    at a 10% or lower alpha since the p-value at 0.144 is higher than 10%
    >>> rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
    >>> stats.ks_2samp(rvs1, rvs3)
    (0.10333333333333333, 0.14498781825751686)
    For an identical distribution, we cannot reject the null hypothesis since
    the p-value is high, 41%:
    >>> rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
    >>> stats.ks_2samp(rvs1, rvs4)
    (0.07999999999999996, 0.41126949729859719)
    """
    from scipy import asarray
    from scipy.stats import kstwobign
    data1, data2 = map(asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side = 'right') / (1.0*n1)
    cdf2 = np.searchsorted(data2, data_all, side = 'right') / (1.0*n2)
    tau=0
    darray = cdf1 - cdf2
    d = np.max(np.absolute(darray))
    # Note: d signed distance
    if d == -np.min(darray):
        d = -d
        jamfri = np.min(np.where(darray == np.min(darray))[0])
    else:
        jamfri = np.min(np.where(darray == np.max(darray))[0])
    tau = data_all[jamfri]
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        prob = kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    except:
        prob = 1.0
    return d, prob, tau

def randomize_times(times, ids = []):
    """
    Randomize the times of the point events of all the ids that are given. This just reshuffles the event times among all the individuals taking into account.
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    ids : list of ids
        If not given, the reshuffling is global, if some ids are given,
        only those will be used for the reshuffling.
    Returns
    -------
    times_random : dictionary of lists
        For each element a list of reshuffled event times

    """
    from random import shuffle
    times_random = dict()
    if len(ids) == 0:
        ids = times.keys()
    Nevents = dict()
    aux = 0
    tlist = []
    for idn in ids:
        aux += len(times[idn])
        Nevents[idn] = aux
        tlist.extend(times[idn])
    shuffle(tlist)
    aux=0
    for idn in ids:
        times_random[idn] = tlist[aux:Nevents[idn]]
        aux += Nevents[idn]
        times_random[idn].sort()
    return times_random

def randomize_ietimes(times, ids = []):
    """
    Randomize the times of the point events of all the ids that are given.
    This randomization keeps the starting time of each individual and reshuffles
    its own interevent times.
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    ids : list of ids
        If not given, the reshuffling is global, if some ids are given,
        only those will be used for the reshuffling.
    Returns
    -------
    times_random : dictionary of lists
        For each element a list of reshuffled event times

    """
    from random import shuffle
    times_random = dict()
    if len(ids) == 0:
        ids = times.keys()
    for idn in ids:
        Nevents = len(times[idn])
        ietlist = [times[idn][i+1]-times[idn][i] for i in range(Nevents-1)]
        shuffle(ietlist)
        t0 = times[idn][0]
        times_random[idn] = [t0]
        for i in range(Nevents-1):
            t0 += ietlist[i]
            times_random[idn] = [t0]
    return times_random

def waiting_times(times, ids, tfloat=True):
    """
    Get the waiting times for two individuals
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    ids : 2 ids for the reshuffling in a list
        If not given, the reshuffling is global, if some ids are given,
        only those will be used for the reshuffling.
    tfloat : boolean variable
        If True the times are taken as floats, if False event times are datetime
        type
    Returns
    -------
    tab, tba : lists of time differences

    """
    flag = 0
    tab = list()
    tba = list()
    idi = ids[0]
    idj = ids[1]
    imin = min(times[idi])
    jmin = min(times[idj])
    if jmin > imin:
        a = idj
        b = idi
        flag = 1
    else:
        a = idi
        b = idj
        flag = 0
    Na = len(times[a])
    Nb = len(times[b])
    bx = 0
    ax = 0
    if tfloat:
        while ax < Na-1 and bx < Nb-1:
            while times[b][bx] <= times[a][ax] and bx < Nb-1:
                bx += 1
            if bx != Nb-1:
                aux = times[a][ax] - times[b][bx-1]
                dtab = aux
                tab.append(dtab)
            while times[a][ax] <= times[b][bx] and ax < Na-1:
                ax+ = 1
            if ax! = Na-1:
                aux = times[b][bx] - times[a][ax-1]
                dtba = aux
                tba.append(dtba)
    else:
        while ax < Na-1 and bx < Nb-1:
            while times[b][bx] <= times[a][ax] and bx < Nb-1:
                bx += 1
            if bx != Nb-1:
                aux = times[a][ax] - times[b][bx-1]
                dtab = aux.days*24.0*60.0 + aux.seconds/60.0
                tab.append(dtab)
            while times[a][ax] <= times[b][bx] and ax < Na-1:
                ax += 1
            if ax != Na-1:
                aux = times[b][bx] - times[a][ax-1]
                dtba = aux.days*24.0*60.0 + aux.seconds/60.0
                tba.append(dtba)
    tba = list(filter(lambda x: x != 0.0, tba))
    tab = list(filter(lambda x: x != 0.0, tab))
    if flag == 0:
        return tab, tba
    else:
        return tba, tab

def D_KS_tau_pvalue_global(times, pmax = 1.0, Nruns=100,
                           min_int = 50, tfloat = True, rand = 't'):
    """
    Gives back the network of follower-followees with a maximum p-value pmax,
    following a global reshuffling scheme.
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    pmax : float (optional)
        maximum p-value allowed for each edge
    Nruns : integer (optional)
        Number of reshufflings used for getting the p-value
    min_int : integer
        minimum number of interactions (waiting times)
    tfloat : boolean variable
        If True the times are taken as floats, if False event times are datetime
        type
    rand : string
        't' reshuffles the event times among all the individuals
        'iet' reshuffles the interevents for each individual
    Returns
    -------
    g : Networkx DiGraph
        Graph containing the information about the follower-followee network.
        The edges have properties such as D_KS, p and tau.
    """
    import networkx as nx
    g=nx.DiGraph()
    tlist = []
    for key in times.keys():
        tlist.extend(times[key])
    ids = list(times.keys())
    N = len(ids)
    for i in range(N-1):
        for j in range(i+1,N):
            tab, tba = waiting_times(times, [ids[i], ids[j]], tfloat=tfloat)
            if len(tab) > min_int and len(tba) > min_int:
                D_KS, p_bad, tau = ks_2samp(tab, tba)
            else:
                D_KS, p_bad, tau=(0.0, 0.0, 0.0)
            if D_KS < 0.0:
                g.add_edge(ids[j], ids[i], D_KS = -D_KS, tau=tau, p=Nruns)
            else:
                g.add_edge(ids[i], ids[j], D_KS = D_KS, tau=tau, p=Nruns)
    for irun in range(Nruns):
        print(Nruns-irun)
        if rand = 't':
            t_rand = randomize_times(times)
        elif rand = 'iet':
            t_rand = randomize_ietimes(times)
        for edge in g.edges():
            i = edge[0]
            j = edge[1]
            D_KS = g[i][j]['D_KS']
            tab, tba = waiting_times(times, [i, j], tfloat = tfloat)
            if len(tab) > min_int and len(tba) > min_int:
                D_KS_rand, p_bad, tau = ks_2samp(tab, tba)
            else:
                D_KS_rand, p_bad, tau=(0.0, 0.0, 0.0)
            if abs(D_KS_rand) < abs(D_KS):
                g[i][j]['p'] -= 1
    for edge in g.edges():
        i = edge[0]
        j = edge[1]
        g[i][j]['p'] = float(g[i][j]['p'])/float(Nruns)
    G=networkx.DiGraph( [ (u,v,d) for u,v,d in g.edges(data=True) if d['p'] < pmax] )
    return G

def D_KS_tau_pvalue_local(times,
                          pmax = 1.0,
                          Nruns=100,
                          min_int=50,
                          tfloat=True
                          ):
    """
    Gives back the network of follower-followees with a maximum p-value pmax,
    following a local reshuffling scheme.
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    pmax : float (optional)
        Maximum p-value allowed for each edge
    Nruns : integer (optional)
        Number of reshufflings used for getting the p-value
    min_int : integer
        Minimum number of interactions (waiting times)
    tfloat : boolean variable
        If True the times are taken as floats, if False event times are datetime type
    Returns
    -------
    g : Networkx DiGraph
        Graph containing the information about the follower-followee network.
        The edges have properties such as D_KS, p and tau.
    """
    g = nx.DiGraph()
    ids = times.keys()
    for i in range(N-1):
        idi = ids[i]
        for j in range(i+1,N):
            idj = ids[j]
            tab, tba = waiting_times(times, [idi, idj], tfloat = tfloat)
            if len(tab) < min_int or len(tba) < min_int:
                D_KS, p_bad, tau = 0.0, 0.0, 0.0
            else:
                D_KS, p_bad, tau = ks_2samp(tab, tba)
            p = Nruns
            for irun in range(Nruns):
                print(Nruns-irun)
                t_rand = randomize_times( times, [idi, idj])
                tab,tba = waiting_times(t_rand, [idi, idj], tfloat = tfloat)
                D_KS_rand, p_bad, tau_rand = ks_2samp(tab, tba)
                if abs(D_KS_rand) < abs(D_KS):
                    p-=1
            p=float(p)/float(Nruns)
            if p < pmax:
                if D_KS < 0.0:
                    g.add_edge(idj, idi, D_KS = -D_KS, tau=tau, p=p)
                else:
                    g.add_edge(ids[i], ids[j], D_KS = D_KS, tau=tau, p=p)
    return g

def excess(times, ids, dt = 5 , tmax = 500, tfloat=True):
    """
    Function to compute the excess rate of events of an element
    just after events of another one
    Parameters
    ----------
    times : dictionary of lists
        The dictionary contains for each element their times of events in a list
    ids : list of ids
        The two (in order), which will be compared
    dt : float (optional)
        Size of temporal bins
    tmax : float (optional)
        Maximum time after which we compute
    tfloat : boolean variable
        If True the times are taken as floats, if False event times are datetime
        type
    Returns
    -------
    x : list of floats
        Time values for each bin
    y_f : list of floats
        Excess rate for each bin
    """
    x = np.asarray([dt*(i + 0.5) for i in range(int(tmax/dt))])
    y = np.zeros(int(tmax/dt))
    y_norm = np.zeros(int(tmax/dt))
    ids = [1 for i in range(len(t1))]+[2 for i in range(len(t2))]
    t = times[ids[0]] + times[ids[1]]
    temp = [i for i in sorted(zip(t,ids))]
    t_s = [temp[i][0] for i in range(len(temp)) ]
    ids_s = [temp[i][1] for i in range(len(temp)) ]
    j = 1
    N = len(t_s)
    i = 0
    while ids_s[i] != 1:
        i+=1
    while i < N-2:
        prod = ids_s[i]*ids_s[i+1]
        while i < N-2 and prod != 2:
            dtemp = t_s[i+1] - t_s[i]
            if not tfloat:
                minutes = dtemp.days*24.0*60.0 + dtemp.seconds/60.0
                i_dt = int((minutes)/dt)
            else:
                i_dt = int((dtemp)/dt)
            if i_dt+1 < len(y):
                for j in range(i_dt+1):
                    y_norm[j] += 1.0
            else:
                for j in range(len(y)):
                    y_norm[j] += 1.0
            i+=1
            prod = ids_s[i] * ids_s[i+1]
        it1 = i
        dtemp = t_s[i+1] - t_s[it1]
        if not tfloat:
            minutes = dtemp.days*24.0*60.0+dtemp.seconds/60.0
            i_dt = int((minutes)/dt)
        else:
            i_dt = int((dtemp)/dt)
        if i_dt < int(tmax/dt):
            y[i_dt] += 1.0
        i+=1
        while i < N-2 and ids_s[i+1] == 2:
            dtemp = t_s[i+1] - t_s[it1]
            if not tfloat:
                minutes = dtemp.days*24.0*60.0 + dtemp.seconds/60.0
                i_dt = int((minutes)/dt)
            else:
                i_dt = int((dtemp)/dt)
            if i_dt < int(tmax/dt):
                y[i_dt] += 1.0
            i += 1
        i+=1
        if i < len(t_s):
            dtemp = t_s[i]-t_s[it1]
            if not tfloat:
                minutes = dtemp.days*24.0*60.0 + dtemp.seconds/60.0
                i_dt = int((minutes)/dt)
            else:
                i_dt = int((dtemp)/dt)
        if i_dt+1 < len(y):
            for j in range(i_dt+1):
                y_norm[j] += 1.0
        else:
            for j in range(len(y)):
                y_norm[j] += 1.0
    y_f = [y[i] / (dt*y_norm[i]) for i in range(len(y))]
    return x,y_f

# TO DO: functions to plot basic quantities
