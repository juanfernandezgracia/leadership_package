[![DOI](https://zenodo.org/badge/180329553.svg)](https://doi.org/10.5281/zenodo.14281447)

# Leadership_KS

This package encapsulates the functions needed to apply the method described in [Ref to paper]. The package is structured into two parts: one which encodes the function for getting the follower-followee networks from presence point data of multiple individuals at the same location, and one which contains generators for different types of models of point processes to test the measure.

## Installation

### Linux

You can compile the code yourself by typing on a terminal:

`python setup.py sdist`

Or download the compressed package `dist/leadership_KS-0.0.1.tar.gz` and run in a terminal (in the folder where you have downloaded the file):

`pip install leadership_KS-0.0.1.tar.gz`

### Other OSs

:man_shrugging:

## Using the package

This package is meant to be applied to point data for several entities and to extract leader-follower relations for those entities.

### Loading the data

#### Basic data

The data has to be loaded into a dictionary called `times` which has as keys the id's of the individuals and as values an ordered list with the times of appearance/activity of each individual. The times can be floats, integers or `datetime` instances (see [datetime module](https://docs.python.org/3/library/datetime.html)). In case they are floats or integers, when creating the network the argument `tfloat` has to be set to `True`. If they are datetime instances the `tfloat` argument has to be set to `False`.

Examples of a valid `times` dictionaries are

```python3
times = {'1' : [1, 1.8, 2.3, 10], 
         '2': [89], 
         'Pepe': [2, 3.1415, 8.7]}
```
or **change this example to put one with datetime instances for the times**

```
times = {'1' : [1, 1.8, 2.3, 10], 
         '2': [89], 
         'Pepe': [2, 3.1415, 8.7]}
```

#### Adding metadata

Right now the only supported metadata are `sex` and `size`. They are stored initially in a dictionary whose keys are the ids and values are `'f'` or `'m'` for sex (femenine and masculine) and a number between 1 and 4 for size.

### Ploting basic quantities

To start one would like to have a look at how the data looks like. For that there are a couple of ploting functions included in the package.

**WORK IN PROGRESS**

### The leadership network

The leadership network will be stored as a `networkx.DiGraph`. 

#### Getting the network

The network will be calculated using the call

```python3
import leadership_KS.functions
import datetime
import numpy as np

# load the data in the dictionary 'times'
times = {'1' : [1, 1.8, 2.3, 10], 
         '2': [89], 
         'Pepe': [2, 3.1415, 8.7]}
         
# calculate the leadership network and put it in 'g'
g = leadership_KS.functions.leadership_network(times,
                       scheme = 'global',
                       pmax = 1.0,
                       Nruns = 100,
                       min_int = 50,
                       tfloat = True,
                       rand = 't')
```
which will store the leadership network in the variable `g`. 

Let's go through the arguments!

* `times` : dictionary of lists. The dictionary contains for each element their times of events in a list (see [Basic data](#basic-data)). 

* `scheme` : string. Tells which reshuffling scheme to use.

    *  `'global'` for a global reshuffling scheme. All individuals are taken together for doing the reshuffling of times.
    
    * `'local'` for a local reshuffling scheme. The individuals are taken by pairs to do the reshuffling.
    
* `pmax` : float (optional). Initializes to 1.0. It is the maximum p-value allowed for each edge.

* `Nruns` : integer (optional). Number of reshufflings used for getting the p-value.

* `min_int` : integer (optional). Minimum number of interactions (waiting times).

* `tfloat` : boolean variable. If True the times are taken as floats, if False event times are datetime type.

* `rand` : string. Type of time reshuffling to be done.

    * `'t'` reshuffles the event times among all the individuals.
    
    * `'iet'` reshuffles the interevents for each individual.

**Although it is coded, it makes no sense to make a local reshuffling of interevent times. Have it in mind!**

#### Properties of the network

The network object has several variables embedded in it. These variables are divided into two classes

* Node variables (if we have entered the metadata for it):

    * `sex`: sex of the individual.
    
    * `size`: size of the individual.
 
 * Edge variables:
 
     * `D_KS`: Value of the Kolmogorov-Smirnof distance.
     
     * `p`: p-value associated to the D_KS of the edge.
     
     * `tau`: Value of time at which we find the maximum distance between cumulative waiting time distributions.
 
 These variables can be accessed in the usual way when there are additional variables in a `networkx.DiGraph` object
 
  ```python3
  node_sex = g[node_id]['sex']
  edge_sex = g[node_1][node_2]['D_KS']
 ```

### Generating fake data

It is useful to generate data where we already know the result in terms of the leadershiop network. For this purpose the package implements through the *generators* some models where who is leading who is obvious by construction.

#### Two uncorrelated Poisson processes

One of the properties of the measure is that when two series of events are uncorrelated among each other, it does not predict a leadership-follower relation. To test this we can generate two independent Poisson processes and feed them to the method.

How to generate two Poisson processes? Here is how

```python3
import leadership_KS.functions
import leadership_KS.generators
import numpy as np

# generate two independent Poisson processes and load the data in the dictionary 'times'
times = leadership_KS.generators.generate_random_times(lam = 1.0, a = 2.0, tmax = 1000.0)
         
# calculate the leadership network and put it in 'g'
g = leadership_KS.functions.leadership_network(times,
                       scheme = 'global',
                       pmax = 1.0,
                       Nruns = 100,
                       min_int = 50,
                       tfloat = True,
                       rand = 't')
```
**put a figure of it working.**


#### One Poisson process and a non-homogeneous Poisson process correlated with the first one


#### A multivariate Hawkes process on an arbitrary network


