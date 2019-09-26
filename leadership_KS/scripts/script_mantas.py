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
    while ax < Na-1 and bx < Nb-1:
        #print ('hi',ax,bx,Na,Nb)
        while times[b][bx]<=times[a][ax] and bx<Nb-1:
            bx+=1
        if bx!=Nb-1:
            aux=times[a][ax]-times[b][bx-1]
            if tfloat:
                dtab=aux
            else:
                dtab=aux.days*24.0*60.0+aux.seconds/60.0
            tab.append(dtab)
        while times[a][ax]<=times[b][bx] and ax<Na-1:
            ax+=1
        if ax!=Na-1:
            aux=times[b][bx]-times[a][ax-1]
            if tfloat:
                dtba=aux
            else:
                dtba=aux.days*24.0*60.0+aux.seconds/60.0
            tba.append(dtba)
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
    ids=times.keys()
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

#read metadata

sex=dict()
size=dict()
idnumber=dict()
ids=list()

fin=open('manta_cleaning_station_metadata.csv','r')
i=0
for line in fin:
    line=line.split(',')
    if line[0]!='Tag ID':
        idn=int(line[0])
        idnumber[idn]=i
        ids.append(idn)
        i+=1
        sex[idn]=line[2][0]
        size[idn]=int(line[2][1])
fin.close()

N=len(ids)

#read event data

times=dict()
for idn in sex.keys():
    times[idn]=list()

tlist=[]

fin=open('manta_cleaning_station_data.csv','r')

for line in fin:
    line=line.split(',')
    if line[0]!='timestamp':
        idn=int(line[2])
        t=datetime.datetime.strptime(line[0], '%m/%d/%Y %H:%M')
        times[idn].append(t)
        tlist.append(t)
fin.close()

for idn in ids:
    times[idn].sort()

#event number rank plot

event_num=[]
for idi in ids:
    event_num.append(len(times[idi]))

ids_sorted = [x for _,x in sorted(zip(event_num,ids),reverse=True)]

event_num.sort(reverse=True)

fig=plt.figure()

plt.yscale('log')

x_f=[]
event_num_f=[]
x_m=[]
event_num_m=[]

for i in range(len(event_num)):
    if sex[ids_sorted[i]] == 'f':
        x_f.append(i+1)
        event_num_f.append(event_num[i])
    else:
        x_m.append(i+1)
        event_num_m.append(event_num[i])


plt.plot(x_f,event_num_f,ls='',marker='o',color='r',ms=7,label='F')
plt.plot(x_m,event_num_m,ls='',marker='o',color='b',ms=7,label='M')

plt.xlabel('Rank',fontsize=30)
plt.ylabel('\# of events',fontsize=30)

plt.legend(fontsize=20)

fig.savefig('./figures/rank_plot_events_mantas.png',bbox_inches='tight')
fig.savefig('./figures/rank_plot_events_mantas.eps',bbox_inches='tight')

plt.close()


#first i am going to remake the data so that it only contains the mantas we want

times_good=dict()
ids_good=[]
tlist_good=[]


for i in range(N):
    if event_num[i] > 100:
        idi=ids_sorted[i]
        ids_good.append(idi)
        times_good[idi]=times[idi]
        for j in range(len(times[idi])):
            tlist_good.append(times[idi][j])

N_good=len(ids_good)

#raw data

#plot raw data only event num over 100

w, h = mpl.figure.figaspect(0.17)
fig=plt.figure(figsize=(w,h))

j=1
for idn in ids_good:
    y=[j+1]*len(times[idn])
    j+=1
    if sex[idn] == 'f':
        plt.scatter(times_good[idn],y,marker='|',color='r',linewidth=0.1,s=50,label='F')
    else:
        plt.scatter(times_good[idn],y,marker='|',color='b',linewidth=0.1,s=50,label='M')

plt.xticks(rotation=70)
plt.xlabel('Date',fontsize=30)
plt.ylabel('Id',fontsize=30)

fig.savefig('./figures/raw_data_mantas_long.png',bbox_inches='tight')
fig.savefig('./figures/raw_data_mantas_long.eps',bbox_inches='tight')
plt.close()

#Interevent time distributions

#print distribution of interevent times
    #fit tail of interevent times distributions

fig=plt.figure()

base=1.1
r=1.0/np.log(base)
norm=0
norm_m=0
norm_f=0

nbins=301
nhalf=nbins/2
t_distri=np.zeros(nbins)
t_distri_m=np.zeros(nbins)
t_distri_f=np.zeros(nbins)
interevent_times=dict()
interevents_f=[]
interevents_m=[]
interevents_all=[]
#fout=open('exponents_interevents.dat','w')
#fout.write('# i id sex alpha sigma xmin xmax n_times\n')
fout2=open('exponents_interevents_table.dat','w')
fout2.write(' i & id & sex & $\alpha$ & $\sigma$ & $x_{min}$ & $x_{max}$ & $n$ \\\\ \hline\n')
for idi in ids_good:
    t_distri_single=np.zeros(nbins)
    norm_single=0
    interevent_times[idi]=[]
    for ix in range(len(times[idi])-1):
        dt=times[idi][ix+1]-times[idi][ix]
        minutes=dt.days*24.0*60.0+dt.seconds/60.0
        if minutes!=0.0:
            minlog=int(round(r*np.log(minutes)))
            #print(minlog,nhalf,minlog+nhalf)
            t_distri[minlog+nhalf]+=1.0
            t_distri_single[minlog+nhalf]+=1.0
            #print(minlog+nhalf,t_distri_single[minlog+nhalf])
            norm+=1
            norm_single+=1
            interevent_times[idi].append(minutes)
            interevents_all.append(minutes)
            if sex[idi]=='m':
                interevents_m.append(minutes)
                t_distri_m[minlog+nhalf]+=1.0
                norm_m+=1
            else:
                interevents_f.append(minutes)
                t_distri_f[minlog+nhalf]+=1.0
                norm_f+=1
    results = powerlaw.Fit(interevent_times[idi],xmin=10.0)
    #print(idi,sex[idi],results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevent_times[idi]))
    fout2.write('%i & %i & %s & %f & %f & %f & %f & %i \\\\ \n' % (i+1,idi,sex[idi],results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevent_times[idi]),len(interevent_times[idi])))
    x_single=list()
    y_single=list()
    for ibin in range(nbins):
        if t_distri_single[ibin]!=0.0:
            exp=ibin-nhalf
            x_single.append(np.power(base,exp))
            w_bin=(base-1.0)*np.power(base,exp-0.5)
            y_single.append(t_distri_single[ibin]/(norm_single*w_bin))
    plt.plot(x_single,y_single,ls='--',marker='o',lw=1,ms=4,color='grey',alpha=0.5)


results = powerlaw.Fit(interevents_f,xmin=10.0)
fout2.write('all & all & f & %f & %f & %f & %f & %f \\\\ \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_f),len(interevents_f)))
results = powerlaw.Fit(interevents_m,xmin=10.0)
fout2.write('all & all & m & %f & %f & %f & %f & %f \\\\ \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_m),len(interevents_m)))
results = powerlaw.Fit(interevents_all,xmin=10.0)
fout2.write('all & all & all & %f & %f & %f & %f & %f \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_all),len(interevents_all)))

#fout.close()
fout2.close()
#sys.exit()

plt.xscale('log')
plt.yscale('log')
x=list()
y=list()
for ibin in range(nbins):
    if t_distri_m[ibin]!=0.0:
        exp=ibin-nhalf
        x.append(np.power(base,exp))
        w_bin=(base-1.0)*np.power(base,exp-0.5)
        y.append(t_distri_m[ibin]/(norm_m*w_bin))
plt.plot(x,y,ls='--',marker='o',color='b',lw=3,ms=7,label='M')

x=list()
y=list()
for ibin in range(nbins):
    if t_distri_f[ibin]!=0.0:
        exp=ibin-nhalf
        x.append(np.power(base,exp))
        w_bin=(base-1.0)*np.power(base,exp-0.5)
        y.append(t_distri_f[ibin]/(norm_f*w_bin))
plt.plot(x,y,ls='--',marker='o',color='r',lw=3,ms=7,label='F')
x=list()
y=list()
for ibin in range(nbins):
    if t_distri[ibin]!=0.0:
        exp=ibin-nhalf
        x.append(np.power(base,exp))
        w_bin=(base-1.0)*np.power(base,exp-0.5)
        y.append(t_distri[ibin]/(norm*w_bin))
plt.plot(x,y,ls='--',marker='o',color='k',lw=3,ms=7,label='Total')


A=0.01

x=[10,346721]
y=A*np.power(x,-1.384)
plt.plot(x,y,ls='--',color='k',lw=5,ms=7,label='$t^{-1.38}$')



plt.xlabel('$t$ (min)',fontsize=30)
plt.ylabel('$P(t)$',fontsize=30)

plt.legend(fontsize=20)

fig.savefig('./figures/interevent_times_mantas.png',bbox_inches='tight')
fig.savefig('./figures/interevent_times_mantas.eps',bbox_inches='tight')
plt.close()

#cyrcadian rythm plot

fig=plt.figure()

cyrc=np.zeros(24)

cyrc_m=dict()
cyrc_f=dict()

cyrc_m_av=np.zeros(24)
cyrc_f_av=np.zeros(24)

for idi in ids_good:
    N_events=len(times[idi])
    if sex[idi] == 'm':
        cyrc_m[idi]=np.zeros(24)
        for i in range(len(times[idi])):
            cyrc_m[idi][times[idi][i].time().hour-1]+=1.0/N_events
            cyrc_m_av[times[idi][i].time().hour-1]+=1.0/N_events
            cyrc[times[idi][i].time().hour-1]+=1.0/N_events
    else:
        cyrc_f[idi]=np.zeros(24)
        for i in range(len(times[idi])):
            cyrc_f[idi][times[idi][i].time().hour-1]+=1.0/N_events
            cyrc_f_av[times[idi][i].time().hour-1]+=1.0/N_events
            cyrc[times[idi][i].time().hour-1]+=1.0/N_events

x=np.arange(1,25)

for idi in cyrc_m.keys():
    plt.plot(x,cyrc_m[idi],c='cyan')
for idi in cyrc_f.keys():
    plt.plot(x,cyrc_f[idi],c='pink')

plt.plot(x,cyrc_f_av/len(cyrc_f.keys()),c='r', lw=2, label='F')
plt.plot(x,cyrc_m_av/len(cyrc_m.keys()),c='b', lw=2, label='M')

plt.plot(x,cyrc/len(ids_good),c='k', lw=4)

plt.xlabel('$t$ (hour)',fontsize=30)
plt.ylabel('Appearance probability',fontsize=30)

plt.legend(fontsize=20)

fig.savefig('./figures/cyrcadian_rythm_mantas.png',bbox_inches='tight')
fig.savefig('./figures/cyrcadian_rythm_mantas.eps',bbox_inches='tight')
#plt.show()
plt.close()
#sys.exit()

#follower-followee network and p values
    #get real value of KS and p value directly and leave only those for which p<0.01

Nruns=20

net=D_KS_tau_pvalue_global(times_good,Nruns=Nruns,tfloat=False)

fout=open('leadership_net_mantas_global.csv','w')
fout.write('Source,Target,Weight,Tau,P\n')
for edge in net.edges():
    fr=edge[0]
    to=edge[1]
    pval=net[fr][to]['p']
    if pval < 1.0:
        fout.write('%i,%i,%f,%f,%f\n' % (fr,to,net[fr][to]['D_KS'],net[fr][to]['tau'],pval) )
fout.close()

#sys.exit()

net=nx.DiGraph()
for i in range(N_good-1):
    idi=ids_good[i]
    for j in range(i+1,N_good):
        print(N_good-i,N_good-j)
        idj=ids_good[j]
        D_KS,tau,p=D_KS_tau_pvalue_local(times_good[idi],times_good[idj],Nruns=Nruns,tfloat=False)
        print(N_good-i,N_good-j,D_KS,tau,p)
        if D_KS < 0.0:
            net.add_edge(idj,idi,D_KS=-D_KS,tau=tau,p=p)
        else:
            net.add_edge(idi,idj,D_KS=D_KS,tau=tau,p=p)

    #guardar la red

    #need to write this for gephi (done)
fout=open('leadership_net_mantas_local.csv','w')
fout.write('Source,Target,Weight,Tau,P\n')
for edge in net.edges():
    fr=edge[0]
    to=edge[1]
    pval=net[fr][to]['p']
    if pval < 1.0:
        fout.write('%i,%i,%f,%f,%f\n' % (fr,to,net[fr][to]['D_KS'],net[fr][to]['tau'],pval) )
fout.close()
