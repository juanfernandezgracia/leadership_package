import numpy as np
import datetime
import sys
from scipy import asarray
from scipy.stats import kstwobign
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

#import powerlaw

from random import expovariate

mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

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

#generate two random sequences and look at the distribution of D_KS:

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
    #first I have to compute lambda(t) in order to do it only once TO DO
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
        
def waiting_times(times):
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
    while ax<Na-1 and bx<Nb-1:
        #print ('hi',ax,bx,Na,Nb)
        while times[b][bx]<=times[a][ax] and bx<Nb-1:
            bx+=1
        if bx!=Nb-1:
            dtab=times[a][ax]-times[b][bx-1]
            tab.append(dtab)
        while times[a][ax]<=times[b][bx] and ax<Na-1:
            ax+=1
        if ax!=Na-1:
            dtba=times[b][bx]-times[a][ax-1]
            tba.append(dtba)
    if flag==0:
        return tab,tba
    else:
        return tba,tab
    
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#end of definitions of functions 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#example of uncorrelated sequence and distribution of distances for reshufflings of the sequences

ids=[1,2]

Nruns=10000
Nbins=1000
dp=Nbins/(2.0*Nruns)


#times=generate_correlated_times(delta=4.0,dt=0.2,tmax=100)
times=generate_random_times(a=1,tmax=100)

w, h = mpl.figure.figaspect(0.17)
fig=plt.figure(figsize=(w,h))



plt.eventplot(np.array([times[1],times[2]]),colors=['b','r'])

plt.xlabel('t',fontsize=30)
#plt.ylabel('Id',fontsize=30)

plt.tick_params(axis='y', which='both',right='off', left='off', labelleft='off')

#plt.axis('off')

fig.savefig('./figures/raw_data_example_Poisson.png',bbox_inches='tight')
fig.savefig('./figures/raw_data_example_Poisson.eps',bbox_inches='tight')
plt.show()
plt.close()


#-------------------------------------------------------------------------
##sys.exit()
#-------------------------------------------------------------------------

    #plot waiting times and D_KS


fig=plt.figure()
tab,tba=waiting_times(times)
D_KS,p,tau=ks_2samp(tab,tba)
#print(tau)

tab.sort()
tba.sort()

Nab=len(tab)
Nba=len(tba)
tab.append(tab[-1])
tba.append(tba[-1])
plt.step(tab, np.arange(Nab+1)/float(Nab), label='$1 \\rightarrow 2$',color='b',lw=2)
plt.step(tba, np.arange(Nba+1)/float(Nba), label='$2 \\rightarrow 1$',color='r',lw=2)

if tau in tab:
    j=0
    while tba[j] < tau:
        j+=1
    y1=(j)/float(Nba)
else:
    y1=(tba.index(tau)+1)/float(Nba)

if D_KS < 0.0:
    l=D_KS+0.1
else:
    l=D_KS-0.1

ax = plt.axes()
ax.arrow(tau, y1, 0, l, lw=3, head_width=0.05, head_length=0.1, fc='k', ec='k',zorder=10)
ax.text(tau+0.2, y1+l/3.0, '$A_{KS}=%.2f$' % D_KS,fontsize=25)

plt.ylim(0,1)
plt.xlabel('$t$',fontsize=30)
plt.ylabel('$P(t^* < t )$',fontsize=30)
plt.legend(fontsize=20)
##plt.show()
fig.savefig('./figures/cumulative_KS_example_Poisson.png',bbox_inches='tight')
fig.savefig('./figures/cumulative_KS_example_Poisson.eps',bbox_inches='tight')
plt.close()


sys.exit()





fig = plt.figure()

D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    #times=generate_correlated_times(delta=4.0,dt=0.2,tmax=1000)
    times=generate_random_times(a=1.0,tmax=1000)
    tab,tba=waiting_times(times)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,ls='--',lw=2,color='k',label='Real')

ax = plt.axes()
ax.arrow(D_KS, 0, 0, np.max(y)/2, lw=3, head_width=0.05, head_length=1, fc='r', ec='r',zorder=10)

tlist_good=[]
for idi in ids:
    for i in range(len(times[idi])):
        tlist_good.append(times[idi][i])
D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    tlist=tlist_good[:]
    times_rand=randomize_times(tlist,times,ids)
    #times=generate_random_times(a=1.0,tmax=100)
    tab,tba=waiting_times(times_rand)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,lw=2,color='g',label='Reshuffled')

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/example_reshuffling_Poisson.png',bbox_inches='tight')
fig.savefig('./figures/example_reshuffling_Poisson.eps',bbox_inches='tight')
#plt.show()
plt.close()
#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------


#heatmap varying parameters of the correlated case

Nruns=100

step=0.2

delta_min=-0.8
delta_max=4.0+step

dt_min=step
dt_max=5.0


delta_range=np.arange(delta_min,delta_max,step=step)
dt_range=np.arange(dt_min,dt_max,step=step)

N_delta=len(delta_range)
N_dt=len(dt_range)

av_D_KS=np.zeros((N_delta,N_dt))

i=0
for i_delta in delta_range:
    j=0
    for i_dt in dt_range:
        for irun in range(Nruns):
            #print(i_delta,i_dt,Nruns-irun)
            times=generate_correlated_times(delta=i_delta,dt=i_dt,tmax=1000)
            tab,tba=waiting_times(times)
            D_KS,p,tau=ks_2samp(tab,tba)
            av_D_KS[i][j]+=D_KS/Nruns
            print(i_delta,i_dt,Nruns-irun,av_D_KS[i][j])
        j+=1
    i+=1

fig=plt.figure()

lim=np.max([-np.min(av_D_KS), np.max(av_D_KS)])

plt.imshow(av_D_KS, cmap=plt.cm.RdBu, interpolation='none', origin='lower', vmin=-lim, vmax=lim, extent=[dt_min-step/2,dt_max-step/2,delta_min-step/2,delta_max-step/2])

plt.colorbar()

plt.xlabel('$\Delta t$',fontsize=30)
plt.ylabel('$\delta $',fontsize=30)
fig.savefig('./figures/av_D_KS_heatmap_corr.png',bbox_inches='tight')
fig.savefig('./figures/av_D_KS_heatmap_corr.eps',bbox_inches='tight')
#plt.show()
plt.close()


#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------

ids=[1,2]

Nruns=10000
Nbins=1000
dp=Nbins/(2.0*Nruns)

#example of correlated sequence and distribution of distances for reshufflings of the sequences

fig = plt.figure()

D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times=generate_correlated_times(delta=4.0,dt=0.2,tmax=1000)
    #times=generate_random_times(a=1.0,tmax=1000)
    tab,tba=waiting_times(times)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,ls='--',lw=2,color='k',label='Real')

ax = plt.axes()
ax.arrow(D_KS, 0, 0, np.max(y)/2, lw=3, head_width=0.05, head_length=1, fc='r', ec='r',zorder=10)

tlist_good=[]
for idi in ids:
    for i in range(len(times[idi])):
        tlist_good.append(times[idi][i])
D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    tlist=tlist_good[:]
    times_rand=randomize_times(tlist,times,ids)
    #times=generate_random_times(a=1.0,tmax=100)
    tab,tba=waiting_times(times_rand)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,lw=2,color='g',label='Reshuffled')

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/example_reshuffling_corr.png',bbox_inches='tight')
fig.savefig('./figures/example_reshuffling_corr.eps',bbox_inches='tight')
#plt.show()
plt.close()


ids=[1,2]

Nruns=1000
Nbins=1000
dp=Nbins/(2.0*Nruns)

#correlated case: 

    #distribution KS varying delta and deltat
    
    #heatmap delta deltat
    
    #distri + one case + distri from reshufling


fig=plt.figure()
for ia in [0.0,0.5,1.0,2.0,5.0,10.0]:
    D_KS_distri=np.zeros(Nbins)
    for irun in range(Nruns):
        print(ia,Nruns-irun)
        times=generate_correlated_times(delta=10.0,dt=ia,tmax=1000)
        #times=generate_random_times(a=ia,tmax=1000)
        tab,tba=waiting_times(times)
        D_KS,p,tau=ks_2samp(tab,tba)
        ibin=int((D_KS+1.0)*Nbins/2.0)
        D_KS_distri[ibin]+=dp
    x=[]
    y=[]
    for ibin in range(Nbins):
        #if D_KS_distri[ibin] != 0.0:
        x.append((0.5+2*ibin)/Nbins-1.0)
        y.append(D_KS_distri[ibin])
    plt.plot(x,y,lw=1,label='$\Delta t=%.1f$' % (ia))

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_corr_var_dt.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_corr_var_dt.eps',bbox_inches='tight')
#plt.show()
plt.close()

#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------

fig=plt.figure()
for ia in [-0.5,0.0,1.0,10.0]:
    D_KS_distri=np.zeros(Nbins)
    for irun in range(Nruns):
        print(ia,Nruns-irun)
        times=generate_correlated_times(delta=ia,dt=0.2,tmax=1000)
        #times=generate_random_times(a=ia,tmax=1000)
        tab,tba=waiting_times(times)
        D_KS,p,tau=ks_2samp(tab,tba)
        ibin=int((D_KS+1.0)*Nbins/2.0)
        D_KS_distri[ibin]+=dp
    x=[]
    y=[]
    for ibin in range(Nbins):
        #if D_KS_distri[ibin] != 0.0:
        x.append((0.5+2*ibin)/Nbins-1.0)
        y.append(D_KS_distri[ibin])
    plt.plot(x,y,lw=1,label='$\delta=%.1f$' % (ia))

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_corr_var_delta.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_corr_var_delta.eps',bbox_inches='tight')
#plt.show()
plt.close()




#first: two Poisson processes of different rates


fig=plt.figure()
for ia in [10.0,2.0,1.0]:
    D_KS_distri=np.zeros(Nbins)
    for irun in range(Nruns):
        print(ia,Nruns-irun)
        #times=generate_correlated_times(delta=2.0,dt=0.5,tmax=1000)
        times=generate_random_times(a=ia,tmax=1000)
        tab,tba=waiting_times(times)
        D_KS,p,tau=ks_2samp(tab,tba)
        ibin=int((D_KS+1.0)*Nbins/2.0)
        D_KS_distri[ibin]+=dp
    x=[]
    y=[]
    for ibin in range(Nbins):
        #if D_KS_distri[ibin] != 0.0:
        x.append((0.5+2*ibin)/Nbins-1.0)
        y.append(D_KS_distri[ibin])
    plt.plot(x,y,lw=1,label='$\lambda_2=%i$' % (int(ia)))

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_Poisson_var_a.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_Poisson_var_a.eps',bbox_inches='tight')
##plt.show()
plt.close()

#-------------------------------------------------------------------------
##sys.exit()
#-------------------------------------------------------------------------

#second: poisson process same lambda varying tmax

fig=plt.figure()
for tm in [100,1000,10000]:
    D_KS_distri=np.zeros(Nbins)
    for irun in range(Nruns):
        print(ia,Nruns-irun)
        #times=generate_correlated_times(delta=2.0,dt=0.5,tmax=1000)
        times=generate_random_times(a=ia,tmax=tm)
        tab,tba=waiting_times(times)
        D_KS,p,tau=ks_2samp(tab,tba)
        ibin=int((D_KS+1.0)*Nbins/2.0)
        D_KS_distri[ibin]+=dp
    x=[]
    y=[]
    for ibin in range(Nbins):
        #if D_KS_distri[ibin] != 0.0:
        x.append((0.5+2*ibin)/Nbins-1.0)
        y.append(D_KS_distri[ibin])
    plt.plot(x,y,lw=1,label='$t_{max}=%i$' % (int(tm)))

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_Poisson_var_tmax.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_Poisson_var_tmax.eps',bbox_inches='tight')
##plt.show()
plt.close()
#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------


#exemplifying the method (both with uncorrelated and correlated case)

    #plot raw data

times=generate_correlated_times(delta=4.0,dt=0.2,tmax=100)
#times=generate_random_times(a=1,tmax=100)

w, h = mpl.figure.figaspect(0.17)
fig=plt.figure(figsize=(w,h))

plt.eventplot(np.array([times[1],times[2]]),colors=['b','r'])

plt.xlabel('t',fontsize=30)
#plt.ylabel('Id',fontsize=30)

plt.tick_params(axis='y', which='both',right='off', left='off', labelleft='off')

#plt.axis('off')

fig.savefig('./figures/raw_data_example_corr.png',bbox_inches='tight')
fig.savefig('./figures/raw_data_example_corr.eps',bbox_inches='tight')
##plt.show()
plt.close()


#-------------------------------------------------------------------------
##sys.exit()
#-------------------------------------------------------------------------

    #plot waiting times and D_KS


fig=plt.figure()
tab,tba=waiting_times(times)
D_KS,p,tau=ks_2samp(tab,tba)
#print(tau)

tab.sort()
tba.sort()

Nab=len(tab)
Nba=len(tba)
tab.append(tab[-1])
tba.append(tba[-1])
plt.step(tab, np.arange(Nab+1)/float(Nab), label='$1 \\rightarrow 2$',color='b',lw=2)
plt.step(tba, np.arange(Nba+1)/float(Nba), label='$2 \\rightarrow 1$',color='r',lw=2)

if tau in tab:
    j=0
    while tba[j] < tau:
        j+=1
    y1=(j)/float(Nba)
else:
    y1=(tba.index(tau)+1)/float(Nba)

if D_KS < 0.0:
    l=D_KS+0.1
else:
    l=D_KS-0.1

ax = plt.axes()
ax.arrow(tau, y1, 0, l, lw=3, head_width=0.05, head_length=0.1, fc='k', ec='k',zorder=10)
ax.text(tau+0.2, y1+l/3.0, '$A_{KS}=%.2f$' % D_KS,fontsize=25)

plt.ylim(0,1)
plt.xlabel('$t$',fontsize=30)
plt.ylabel('$P(t^* < t )$',fontsize=30)
plt.legend(fontsize=20)
##plt.show()
fig.savefig('./figures/cumulative_KS_example_corr.png',bbox_inches='tight')
fig.savefig('./figures/cumulative_KS_example_corr.eps',bbox_inches='tight')
plt.close()

#-------------------------------------------------------------------------
##sys.exit()
#-------------------------------------------------------------------------

#distribution of D_KS


ids=[1,2]

Nruns=100000
Nbins=200
#dp=2.0/(Nbins)
dp=Nbins/(2.0*Nruns)

fig=plt.figure()
D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    #times=generate_correlated_times(delta=2.0,dt=0.5,tmax=1000)
    times=generate_random_times(a=1.0,tmax=100)
    tab,tba=waiting_times(times)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,lw=2,color='g')

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_Poisson_example.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_Poisson_example.eps',bbox_inches='tight')
##plt.show()
plt.close()

fig=plt.figure()
D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times=generate_correlated_times(delta=4.0,dt=0.2,tmax=100)
    #times=generate_random_times(a=1.0,tmax=100)
    tab,tba=waiting_times(times)
    D_KS,p,tau=ks_2samp(tab,tba)
    ibin=int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin]+=dp
x=[]
y=[]
for ibin in range(Nbins):
    #if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x,y,lw=2,color='g')

plt.xlabel('$A_{KS}$',fontsize=30)
plt.ylabel('$P(A_{KS})$',fontsize=30)
plt.legend(fontsize=20)
fig.savefig('./figures/D_KS_distri_corr_example.png',bbox_inches='tight')
fig.savefig('./figures/D_KS_distri_corr_example.eps',bbox_inches='tight')
##plt.show()
plt.close()



#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------



times=generate_correlated_times(delta=10.0,dt=0.2,tmax=10)

w, h = mpl.figure.figaspect(0.17)
fig=plt.figure(figsize=(w,h))

for idn in ids:
    y=[idn]*len(times[idn])
    plt.scatter(times[idn],y,marker='|',color='r',linewidth=0.5,s=500)

plt.xlabel('t',fontsize=30)
plt.ylabel('Id',fontsize=30)

fig.savefig('./figures/raw_data_correlated.png',bbox_inches='tight')
#plt.show()
plt.close()

#sys.exit()



fout=open('kk_rand_from_corr_10_02.dat','w')
times=generate_correlated_times(delta=10.0,dt=0.2,tmax=1000)
tab,tba=waiting_times(times)
D_KS,p,tau=ks_2samp(tab,tba)
print(D_KS)
ids=times.keys()
tlist_good=[]
for idi in ids:
    for i in range(len(times[idi])):
        tlist_good.append(times[idi][i])
for irun in range(Nruns):
    print(Nruns-irun)
    tlist=tlist_good[:]
    ts_rand=randomize_times(tlist,times,ids)
    tab,tba=waiting_times(ts_rand)
    D_KS,p,tau=ks_2samp(tab,tba)
    fout.write('%f\n' % D_KS)
fout.close()


#-------------------------------------------------------------------------
#sys.exit()
#-------------------------------------------------------------------------

fout=open('kk_corr_2_05.dat','w')
for irun in range(Nruns):
    print(Nruns-irun)
    times=generate_correlated_times(delta=2.0,dt=0.5,tmax=1000)
    #times=generate_random_times(a=100.0,tmax=1000)
    tab,tba=waiting_times(times)
    D_KS,p,tau=ks_2samp(tab,tba)
    fout.write('%f\n' % D_KS)
fout.close()


