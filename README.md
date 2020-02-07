# Leadership_KS

This package encapsulates the functions needed to apply the method described in [Ref to paper]. The package is structured into two parts: one which encodes the function for getting the follower-followee networks from presence point data of multiple individuals at the same location, and one which contains generators for different types of models of point processes to test the measure.

## Installation

### Linux

You can compile the code yourself by typing on a terminal:

`python setup.py sdist`

Or download the compressed package `dist/leadership_KS-0.0.1.tar.gz` and run in a terminal (in the folder where you have downloaded the file):

`pip install leadership_KS-0.0.1.tar.gz`

#### Windows

Install linux and go to the previous section.

#### Mac Os

Same as with windows.

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



### Ploting basic quantities


### The leadership network

The leadership network will be stored as a `networkx.DiGraph`. 

#### Getting the network

The network will be calculated by the call

```python3
g = leadership_network(times,
                       scheme = 'global',
                       pmax = 1.0,
                       Nruns = 100,
                       min_int = 50,
                       tfloat = True,
                       rand = 't')
```
which will store the leadership network in the variable `g`. 

#### Properties of the network

### Generating fake data


#### Two uncorrelated Poisson processes


#### One Poisson process and a non-homogeneous Poisson process correlated with the first one


#### A multivariate Hawkes process on an arbitrary network



