# %%
import numpy as np
import leadership_KS.functions as ksfunc
import leadership_KS.generators as ksgen
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
# %%
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
# %%
# example of uncorrelated sequence and distribution of distances for reshufflings of the sequences

ids = [1, 2]
Nruns = 10000
Nbins = 1000
dp = Nbins/(2.0*Nruns)

# times=generate_correlated_times(delta=4.0,dt=0.2,tmax=100)
times = ksgen.generate_random_times(a=1, tmax=100)

w, h = mpl.figure.figaspect(0.17)
fig = plt.figure(figsize=(w, h))

plt.eventplot(np.array([times[1], times[2]]), colors=['b', 'r'])

plt.xlabel('t', fontsize=30)
# plt.ylabel('Id',fontsize=30)

plt.tick_params(axis='y', which='both', right='off',
                left='off', labelleft='off')

# plt.axis('off')

# fig.savefig('./figures/raw_data_example_Poisson.png',bbox_inches='tight')
# fig.savefig('./figures/raw_data_example_Poisson.eps',bbox_inches='tight')
# plt.show()
# plt.close()
# %%
# plot waiting times and D_KS


fig = plt.figure(figsize=(8, 6))
tab, tba = ksfunc.waiting_times(times, ids)
D_KS, p, tau = ksfunc.ks_2samp(tab, tba)
# print(tau)

tab.sort()
tba.sort()

Nab = len(tab)
Nba = len(tba)
tab.append(tab[-1])
tba.append(tba[-1])
plt.step(tab, np.arange(Nab+1)/float(Nab),
         label='$1 \\rightarrow 2$', color='b', lw=2)
plt.step(tba, np.arange(Nba+1)/float(Nba),
         label='$2 \\rightarrow 1$', color='r', lw=2)

if tau in tab:
    j = 0
    while tba[j] < tau:
        j += 1
    y1 = (j)/float(Nba)
else:
    y1 = (tba.index(tau)+1)/float(Nba)

if D_KS < 0.0:
    l = D_KS+0.1
else:
    l = D_KS-0.1

ax = plt.axes()
ax.arrow(tau, y1, 0, l, lw=3, head_width=0.05,
         head_length=0.1, fc='k', ec='k', zorder=10)
ax.text(tau+0.2, y1+l/3.0, '$A_{KS}=%.2f$' % D_KS, fontsize=25)

plt.ylim(0, 1)
plt.xlabel('$t$', fontsize=30)
plt.ylabel('$P(t^* < t )$', fontsize=30)
plt.legend(fontsize=20)
# plt.show()
# fig.savefig('./figures/cumulative_KS_example_Poisson.png',bbox_inches='tight')
# fig.savefig('./figures/cumulative_KS_example_Poisson.eps',bbox_inches='tight')
# plt.close()
# %%
fig = plt.figure(figsize=(8, 6))

D_KS_distri = np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    # times=generate_correlated_times(delta=4.0,dt=0.2,tmax=1000)
    times = ksgen.generate_random_times(a=1.0, tmax=1000)
    tab, tba = ksfunc.waiting_times(times, ids)
    D_KS, p, tau = ksfunc.ks_2samp(tab, tba)
    ibin = int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin] += dp
x = []
y = []
for ibin in range(Nbins):
    # if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x, y, ls='--', lw=2, color='k', label='Real')

ax = plt.axes()
ax.arrow(D_KS, 0, 0, np.max(y)/2, lw=3, head_width=0.05,
         head_length=1, fc='r', ec='r', zorder=10)

D_KS_distri = np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times_rand = ksfunc.randomize_ietimes(times, ids)
    tab, tba = ksfunc.waiting_times(times_rand, ids)
    D_KS, p, tau = ksfunc.ks_2samp(tab, tba)
    ibin = int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin] += dp
x = []
y = []
for ibin in range(Nbins):
    # if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x, y, lw=2, color='g', label='Reshuffled')

plt.xlabel('$A_{KS}$', fontsize=30)
plt.ylabel('$P(A_{KS})$', fontsize=30)
plt.legend(loc=2, fontsize=20)
# fig.savefig('./figures/example_reshuffling_Poisson.png',bbox_inches='tight')
# fig.savefig('./figures/example_reshuffling_Poisson.eps',bbox_inches='tight')
# plt.show()
# plt.close()
# %%


ids = [1, 2]

Nruns = 10000
Nbins = 1000
dp = Nbins/(2.0*Nruns)

# example of correlated sequence and distribution of distances for reshufflings of the sequences

fig = plt.figure()

D_KS_distri = np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    times = ksgen.generate_correlated_times(delta=4.0, dt=0.2, tmax=1000)
    # times=generate_random_times(a=1.0,tmax=1000)
    tab, tba = ksfunc.waiting_times(times, ids)
    D_KS, p, tau = ksfunc.ks_2samp(tab, tba)
    ibin = int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin] += dp
x = []
y = []
for ibin in range(Nbins):
    # if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x, y, ls='--', lw=2, color='k', label='Real')

ax = plt.axes()
ax.arrow(D_KS, 0, 0, np.max(y)/2, lw=3, head_width=0.05,
         head_length=1, fc='r', ec='r', zorder=10)

tlist_good = []
for idi in ids:
    for i in range(len(times[idi])):
        tlist_good.append(times[idi][i])
D_KS_distri = np.zeros(Nbins)
for irun in range(Nruns):
    print(Nruns-irun)
    tlist = tlist_good[:]
    times_rand = ksfunc.randomize_ietimes(times, ids)
    # times=generate_random_times(a=1.0,tmax=100)
    tab, tba = ksfunc.waiting_times(times_rand, ids)
    D_KS, p, tau = ksfunc.ks_2samp(tab, tba)
    ibin = int((D_KS+1.0)*Nbins/2.0)
    D_KS_distri[ibin] += dp
x = []
y = []
for ibin in range(Nbins):
    # if D_KS_distri[ibin] != 0.0:
    x.append((0.5+2*ibin)/Nbins-1.0)
    y.append(D_KS_distri[ibin])
plt.plot(x, y, lw=2, color='g', label='Reshuffled')

plt.xlabel('$A_{KS}$', fontsize=30)
plt.ylabel('$P(A_{KS})$', fontsize=30)
plt.legend(fontsize=20)
# fig.savefig('./figures/example_reshuffling_corr.png',bbox_inches='tight')
# fig.savefig('./figures/example_reshuffling_corr.eps',bbox_inches='tight')
# plt.show()


# %%
