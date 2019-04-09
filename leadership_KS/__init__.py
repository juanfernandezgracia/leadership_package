import datetime
from scipy import asarray
from scipy.stats import kstwobign
from random import shuffle




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


import matplotlib.pyplot as plt
import matplotlib as mpl

name = 'leadership_KS'
