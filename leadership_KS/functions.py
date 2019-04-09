#!/usr/bin/env python
import numpy as np
import datetime
import sys
from scipy import asarray
from scipy.stats import kstwobign
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import networkx as nx
import powerlaw
from random import expovariate
from random import seed


import os
import pickle
import gzip

from sklearn.metrics import roc_auc_score

from pybasicbayes.util.text import progprint_xrange

from pyhawkes.models import \
    DiscreteTimeNetworkHawkesModelGammaMixture, \
    DiscreteTimeStandardHawkesModel, \
    DiscreteTimeNetworkHawkesModelSpikeAndSlab

def ks_2samp(data1, data2):
    """
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.
    Parameters
    ----------
    a, b : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different
    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value
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
    data1, data2 = map(asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1,data2])
    #print(len(data_all))
    cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = np.searchsorted(data2,data_all,side='right')/(1.0*n2)
    tau=0
    darray=cdf1-cdf2
    d = np.max(np.absolute(darray))
    if d==-np.min(darray):
        d=-d
        jamfri=np.min(np.where(darray == np.min(darray))[0])
    else:
        jamfri=np.min(np.where(darray == darray.max())[0])
    #print(jamfri)
    tau=data_all[jamfri]
    #tau=data_all[list(darray).index(d)]
    # Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        prob = kstwobign.sf((en + 0.12 + 0.11 / en) * d)
        #prob = distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    except:
        prob = 1.0
    return d, prob, tau

#randomize the data:
def randomize_times(tlist,times,ids):
    times_random=dict()
    for idn in ids:
        times_random[idn]=list()
        for i in range(len(times[idn])):
            t=random.choice(tlist)
            times_random[idn].append(t)
            tlist.remove(t)
        times_random[idn].sort()
    return times_random

def waiting_times(times,tfloat=True):
    ids=times.keys()
    flag=0
    tab=list()
    tba=list()
    idi=1
    idj=2
    imin=min(times[idi])
    jmin=min(times[idj])
    if jmin>imin:
        a=idj
        b=idi
        flag=1
    else:
        a=idi
        b=idj
        flag=0
    Na=len(times[a])
    Nb=len(times[b])
    bx=0
    ax=0
    if tfloat:
        while ax < Na-1 and bx < Nb-1:
            #print ('hi',ax,bx,Na,Nb)
            while times[b][bx]<=times[a][ax] and bx<Nb-1:
                bx+=1
            if bx!=Nb-1:
                aux=times[a][ax]-times[b][bx-1]
                dtab=aux
                #if dtab > 0.0:
                tab.append(dtab)
            while times[a][ax]<=times[b][bx] and ax<Na-1:
                ax+=1
            if ax!=Na-1:
                aux=times[b][bx]-times[a][ax-1]
                dtba=aux
                #if dtba > 0.0:
                tba.append(dtba)
    else:
        while ax < Na-1 and bx < Nb-1:
            #print ('hi',ax,bx,Na,Nb)
            while times[b][bx]<=times[a][ax] and bx<Nb-1:
                bx+=1
            if bx!=Nb-1:
                aux=times[a][ax]-times[b][bx-1]
                dtab=aux.days*24.0*60.0+aux.seconds/60.0
                #if dtab > 0.0:
                tab.append(dtab)
            while times[a][ax]<=times[b][bx] and ax<Na-1:
                ax+=1
            if ax!=Na-1:
                aux=times[b][bx]-times[a][ax-1]
                dtba=aux.days*24.0*60.0+aux.seconds/60.0
                #if dtba > 0.0:
                tba.append(dtba)
    tba = list(filter(lambda x: x!= 0.0, tba))
    tab = list(filter(lambda x: x!= 0.0, tab))
    if flag==0:
        return tab,tba
    else:
        return tba,tab
    
def D_KS_tau_pvalue_local(t1,t2,Nruns=100,tfloat=True):
    times={1:t1,2:t2}
    tlist=t1+t2
    ids=times.keys()
    tab,tba=waiting_times(times,tfloat=tfloat)
    if len(tab) < 50 or len(tba) < 50:
        return 1.0,0.0,1.0
    D_KS,p_bad,tau=ks_2samp(tab,tba)
    p=Nruns
    for irun in range(Nruns):
        print(Nruns-irun)
        tlist=t1+t2
        t_rand=randomize_times(tlist,times,ids)
        tab,tba=waiting_times(t_rand,tfloat=tfloat)
        D_KS_rand,p_bad,tau_rand=ks_2samp(tab,tba)
        print(Nruns-i,D_KS_rand,D_KS)
        if abs(D_KS_rand) < abs(D_KS):
            p-=1
    p=float(p)/float(Nruns)
    return D_KS,tau,p

def D_KS_tau_pvalue_global(times,Nruns=100,tfloat=True):
    g=nx.DiGraph()
    tlist=[]
    for key in times.keys():
        for j in range(len(times[key])):
            tlist.append(times[key][j])
    ids=list(times.keys())
    N=len(ids)
    for i in range(N-1):
        for j in range(i+1,N):
            tab,tba=waiting_times({1:times[ids[i]],2:times[ids[j]]},tfloat=tfloat)
            if len(tab) > 50 and len(tba) > 50:
                D_KS,p_bad,tau=ks_2samp(tab,tba)
            else:
                D_KS,p_bad,tau=(0.0,0.0,0.0)
            if D_KS < 0.0:
                g.add_edge(ids[j],ids[i],D_KS=-D_KS,tau=tau,p=Nruns)
            else:
                g.add_edge(ids[i],ids[j],D_KS=D_KS,tau=tau,p=Nruns)
    for irun in range(Nruns):
        print(Nruns-irun)
        tlist=[]
        for key in times.keys():
            for j in range(len(times[key])):
                tlist.append(times[key][j])
        #print(len(tlist))
        t_rand=randomize_times(tlist,times,ids)
        for edge in g.edges():
            i=edge[0]
            j=edge[1]
            D_KS=g[i][j]['D_KS']
            tab,tba=waiting_times({1:t_rand[i],2:t_rand[j]},tfloat=tfloat)
            if len(tab) > 50 and len(tba) > 50:
                D_KS_rand,p_bad,tau=ks_2samp(tab,tba)
            else:
                D_KS_rand,p_bad,tau=(0.0,0.0,0.0)
            if abs(D_KS_rand) < abs(D_KS):
                g[i][j]['p']-=1
    for edge in g.edges():
        i=edge[0]
        j=edge[1]
        g[i][j]['p']=float(g[i][j]['p'])/float(Nruns)
    return g


#function to compute the excess probability
def excess(t1,t2,dt=5,tmax=500,tfloat=True):
    x=np.asarray([dt*(i+0.5) for i in range(int(tmax/dt))])
    y=np.zeros(int(tmax/dt))
    y_norm=np.zeros(int(tmax/dt))
    ids=[1 for i in range(len(t1))]+[2 for i in range(len(t2))]
    t=t1+t2
    temp=[i for i in sorted(zip(t,ids))]
    t_s=[temp[i][0] for i in range(len(temp)) ]
    ids_s=[temp[i][1] for i in range(len(temp)) ]
    i=0
    j=1
    N=len(t_s)
    i=0
    while ids_s[i] != 1:
        i+=1
    while i < N-2:
        prod=ids_s[i]*ids_s[i+1]
        while i < N-2 and prod != 2:
            dtemp=t_s[i+1]-t_s[i]
            if not tfloat:
                minutes=dtemp.days*24.0*60.0+dtemp.seconds/60.0
                i_dt=int((minutes)/dt)
            else:
                i_dt=int((dtemp)/dt)
            if i_dt+1 < len(y):
                for j in range(i_dt+1):
                    y_norm[j]+=1.0
            else:
                for j in range(len(y)):
                    y_norm[j]+=1.0
            i+=1
            prod=ids_s[i]*ids_s[i+1]
        it1=i
        dtemp=t_s[i+1]-t_s[it1]
        if not tfloat:
            minutes=dtemp.days*24.0*60.0+dtemp.seconds/60.0
            i_dt=int((minutes)/dt)
        else:
            i_dt=int((dtemp)/dt)
        if i_dt < int(tmax/dt):
            y[i_dt]+=1.0
        i+=1
        while i < N-2 and ids_s[i+1] == 2:
            dtemp=t_s[i+1]-t_s[it1]
            if not tfloat:
                minutes=dtemp.days*24.0*60.0+dtemp.seconds/60.0
                i_dt=int((minutes)/dt)
            else:
                i_dt=int((dtemp)/dt)
            if i_dt < int(tmax/dt):
                y[i_dt]+=1.0
            i+=1
        i+=1
        if i < len(t_s):
            dtemp=t_s[i]-t_s[it1]
            if not tfloat:
                minutes=dtemp.days*24.0*60.0+dtemp.seconds/60.0
                i_dt=int((minutes)/dt)
            else:
                i_dt=int((dtemp)/dt)
        if i_dt+1 < len(y):
            for j in range(i_dt+1):
                y_norm[j]+=1.0
        else:
            for j in range(len(y)):
                y_norm[j]+=1.0
    #print(y,y_norm,y/(dt*y_norm))
    y_f=[y[i]/(dt*y_norm[i]) for i in range(len(y))]
    return x,y_f


#generate two random sequences

def generate_random_times(a=2.0,tmax=1000):
    times={1:[],2:[]}
    ids=times.keys()
    lam=1.0
    l=[lam,a*lam]
    for idi in ids:
        lm=l[idi-1]
        t=expovariate(lm)
        while t<tmax:
            times[idi].append(t)
            t+=expovariate(lm)
    return times

def generate_correlated_times(delta=2.0,dt=0.5,tmax=1000):
    times={1:[],2:[]}
    ids=times.keys()
    #do first time series as a Poisson process of rate 1
    lam=1.0
    t=expovariate(lam)
    while t<tmax:
        times[1].append(t)
        t+=expovariate(lam)
    #to integrate the second process I use a Poisson process of rate one and do a change of timescales
    #first I have to compute lambda(t) in order to do it only once
    t=0.0
    tau=expovariate(lam)
    #print('Hi!',tau,t,times[1][0])
    lam_l=lam_det(times[1],dt)
    t=inverted_time_step(tau,t,lam_l,delta,dt)
    #print('Hi2!',tau,t)
    while t<tmax:
    #for i in range(10):
        times[2].append(t)
        tau=expovariate(lam)
        t+=inverted_time_step(tau,t,lam_l,delta,dt)
        #print(t)
    return times

def lam_det(times,dt):
    N=len(times)
    lam=[(0,0)]
    lam.append((times[0],1))
    for i in range(1,N):
        delta_t = times[i] - times[i-1] 
        if delta_t > dt:
            lam.append((times[i-1]+dt,0))
            lam.append((times[i],1))
    return lam

def inverted_time_step(tau_targ,t,lam,delta,dt):
    ind=0
    N=len(lam)
    while lam[ind][0] < t and ind < N-1:
        ind += 1
    if ind == N-1:
        if lam[N-1][0] < t:
            if lam[N-1][1] == 0:
                lam_g=[(t,0)]
            else:
                lam_g=[(t,1),(t+dt,0)]
        else:
            lam_g=[(t,lam[ind-1][1])]+lam[ind:]
    else:
        lam_g=[(t,lam[ind-1][1])]+lam[ind:]
    N_lam_g=len(lam_g)
    tau=0.0
    for i in range(N_lam_g-1):
        dtau=(lam_g[i+1][0]-lam_g[i][0])*(1.0+delta*lam_g[i][1])
        #print('dd',dtau,N_lam)
        tau+=dtau
        if tau > tau_targ:
            t_good=lam_g[i][0]+(tau_targ-tau+dtau)/(1.0+delta*lam_g[i][1])
            return t_good-t
    t_good=lam_g[N_lam_g-1][0]+(tau_targ-tau)/(1.0+delta*lam_g[N_lam_g-1][1])
    return t_good-t


#------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------

def generate_hawkes(edges):

    np.random.seed(1122334455)
    # Create a simple random network with K nodes a sparsity level of p
    # Each event induces impulse responses of length dt_max on connected nodes
    a=set()
    for edge in edges:
        a.add(edge[0])
        a.add(edge[1])
    K = len(a)
    p = 0.25
    dt_max = 20
    network_hypers = {"p": p, "allow_self_connections": False}


    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dt_max, network_hypers=network_hypers)

    A2=np.zeros((K,K))

    for edge in edges:
        A2[edge[0]][edge[1]]=1.0

    true_model.weight_model.A = A2
    A3=0.5*np.ones((K,K))
    true_model.weight_model.W = A3

    Tmax=50000
    S,R = true_model.generate(T=Tmax)
    
    
    times=dict()
    
    for idi in range(K):
        for it in range(Tmax):
            n=S[it][idi]
            if n > 0:
                if idi not in times.keys():
                    times[idi]=[]
                else:
                    times[idi].append(it)

    ids=list(times.keys())

    for idn in ids:
        times[idn].sort()
    
    #print(true_model.impulse_model)
    
    #sys.exit()
    
    return ids,times
