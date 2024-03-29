import leadership_KS.functions
import leadership_KS.generators
import sys
import datetime
import matplotlib.pyplot as plt
import numpy as np

#read metadata

sex=dict()
size=dict()
fin=open('/home/juanf/Work/acoustic/mantas/manta_cleaning_station_metadata.csv','r')
for line in fin:
    line=line.split(',')
    if line[0]!='Tag ID':
        idn=int(line[0])
        sex[idn]=line[2][0]
        size[idn]=int(line[2][1])
fin.close()
ids = list(sex.keys())
N=len(ids)

#read event data
times = dict()
ids = []

fin=open('/home/juanf/Work/acoustic/mantas/manta_cleaning_station_data.csv','r')
for line in fin:
    line=line.split(',')
    if line[0]!='timestamp':
        idn=int(line[2])
        t=datetime.datetime.strptime(line[0], '%m/%d/%Y %H:%M')
        if idn in ids:
            times[idn].append(t)
        else:
            ids.append(idn)
            times[idn] = [t]
fin.close()

for idn in ids:
    times[idn].sort()

# get distribution of waiting times for all the pairs

fig=plt.figure()

base = 1.1
r = 1.0/np.log(base)
nbins = 131

wt_glob_distri = np.zeros(nbins)
norm_glob = 0

for idi in range(len(ids)-1):
    for idj in range(idi+1, len(ids)):
        ids_ = [ids[idi], ids[idj]]
        tba, tab = leadership_KS.functions.waiting_times(times, ids_, tfloat=False)
        if len(tba) > 50:
            norm = 0
            wt_distri = np.zeros(nbins)
            for t in tba:
                ibin = int(round(r*np.log(t)))
                wt_distri[ibin] += 1
                norm += 1
                wt_glob_distri[ibin] += 1
                norm_glob += 1
            x = []
            y = []
            for ibin in range(nbins):
                if wt_distri[ibin]!=0.0:
                    exp=ibin
                    x.append(np.power(base,exp))
                    w_bin=(base-1.0)*np.power(base,exp-0.5)
                    y.append(wt_distri[ibin]/(norm*w_bin))
            plt.plot(x,y,ls='--',marker='o',lw=1,ms=4,color='grey',alpha=0.5)
        if len(tab) > 50:
            norm = 0
            wt_distri = np.zeros(nbins)
            for t in tab:
                ibin = int(round(r*np.log(t)))
                wt_distri[ibin] += 1
                norm += 1
                wt_glob_distri[ibin] += 1
                norm_glob += 1
            x = []
            y = []
            for ibin in range(nbins):
                if wt_distri[ibin]!=0.0:
                    exp=ibin
                    x.append(np.power(base,exp))
                    w_bin=(base-1.0)*np.power(base,exp-0.5)
                    y.append(wt_distri[ibin]/(norm*w_bin))
            plt.plot(x,y,ls='--',marker='o',lw=1,ms=4,color='grey',alpha=0.5)
x = []
y = []
for ibin in range(nbins):
    if wt_glob_distri[ibin]!=0.0:
        exp=ibin
        #print(ibin, np.power(base,exp))
        x.append(np.power(base,exp))
        w_bin=(base-1.0)*np.power(base,exp-0.5)
        y.append(wt_glob_distri[ibin]/(norm_glob*w_bin))
plt.plot(x,y,ls='--',marker='o',lw=1,ms=4,color='k',alpha=1.0)
#print(x, y)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('time (minutes)', fontsize=20)
plt.ylabel('pdf', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fig.savefig('../figures/wt_distributions.png', bbox_inches='tight')
plt.show()
sys.exit()


#     interevent_times[idi]=[]
#     for ix in range(len(times[idi])-1):
#         dt=times[idi][ix+1]-times[idi][ix]
#         minutes=dt.days*24.0*60.0+dt.seconds/60.0
#         if minutes!=0.0:
#             minlog=int(round(r*np.log(minutes)))
#             #print(minlog,nhalf,minlog+nhalf)
#             t_distri[minlog+nhalf]+=1.0
#             t_distri_single[minlog+nhalf]+=1.0
#             #print(minlog+nhalf,t_distri_single[minlog+nhalf])
#             norm+=1
#             norm_single+=1
#             interevent_times[idi].append(minutes)
#             interevents_all.append(minutes)
#             if sex[idi]=='m':
#                 interevents_m.append(minutes)
#                 t_distri_m[minlog+nhalf]+=1.0
#                 norm_m+=1
#             else:
#                 interevents_f.append(minutes)
#                 t_distri_f[minlog+nhalf]+=1.0
#                 norm_f+=1
#     results = powerlaw.Fit(interevent_times[idi],xmin=10.0)
#     #print(idi,sex[idi],results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevent_times[idi]))
#     fout2.write('%i & %i & %s & %f & %f & %f & %f & %i \\\\ \n' % (i+1,idi,sex[idi],results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevent_times[idi]),len(interevent_times[idi])))
#     x_single=list()
#     y_single=list()
#     for ibin in range(nbins):
#         if t_distri_single[ibin]!=0.0:
#             exp=ibin-nhalf
#             x_single.append(np.power(base,exp))
#             w_bin=(base-1.0)*np.power(base,exp-0.5)
#             y_single.append(t_distri_single[ibin]/(norm_single*w_bin))
#     plt.plot(x_single,y_single,ls='--',marker='o',lw=1,ms=4,color='grey',alpha=0.5)


# results = powerlaw.Fit(interevents_f,xmin=10.0)
# fout2.write('all & all & f & %f & %f & %f & %f & %f \\\\ \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_f),len(interevents_f)))
# results = powerlaw.Fit(interevents_m,xmin=10.0)
# fout2.write('all & all & m & %f & %f & %f & %f & %f \\\\ \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_m),len(interevents_m)))
# results = powerlaw.Fit(interevents_all,xmin=10.0)
# fout2.write('all & all & all & %f & %f & %f & %f & %f \n' % (results.power_law.alpha,results.power_law.sigma,results.power_law.xmin,max(interevents_all),len(interevents_all)))


sys.exit()



# g = leadership_KS.functions.leadership_network(times,
                        #    scheme = 'global',
                        #    pmax = 0.002,
                        #    Nruns = 500,
                        #    min_int = 50,
                        #    tfloat = True,
                        #    rand = 'iet'
                        #    ):
#
# print(len(g.edges()))
#
# sys.exit()

ids=[1,2]

Nruns=5000
Nbins=1000
dp=Nbins/(2.0*Nruns)

#example of correlated sequence and distribution of distances for reshufflings of the sequences

fig = plt.figure()

D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times=leadership_KS.generators.generate_correlated_times(delta=4.0,dt=0.2,tmax=1000)
    #times=generate_random_times(a=1.0,tmax=1000)
    tab,tba=leadership_KS.functions.waiting_times(times, ids)
    D_KS,p,tau=leadership_KS.functions.ks_2samp(tab,tba)
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

D_KS_distri=np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times_rand=leadership_KS.functions.randomize_ietimes(times)
    #times=generate_random_times(a=1.0,tmax=100)
    tab,tba=leadership_KS.functions.waiting_times(times_rand, ids)
    D_KS,p,tau=leadership_KS.functions.ks_2samp(tab, tba)
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
fig.savefig('example_reshuffling_corr.png',bbox_inches='tight')
plt.show()
plt.close()

sys.exit()


times = leadership_KS.generators.generate_correlated_times()

g = leadership_KS.functions.D_KS_tau_pvalue_global(times,
                                                   pmax = 0.6,
                                                   Nruns = 500,
                                                   min_int = 50,
                                                   tfloat = True,
                                                   rand = 'iet')

print(len(g.edges()))

sys.exit()







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

event_num = [len(times[idi]) for idi in ids]

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
