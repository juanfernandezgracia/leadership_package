

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
