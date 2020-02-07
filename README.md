# Leadership_KS

This package encapsulates the functions needed to apply the method described in [Ref to paper]. The package is structured into two parts: one which encodes the function for getting the follower-followee networks from presence point data of multiple individuals at the same location, and one which contains generators for different types of models of point processes to test the measure.

## Installation

### Linux

You can compile the code yourself by typing on a terminal:

`python setup.py sdist`

Or download the compressed package `leadership_KS-0.0.1.tar.gz` and run in a terminal (in the folder where you have downloaded the file):

`pip install leadership_KS-0.0.1.tar.gz`

#### Windows

Install linux and go to the previous section.

#### Mac Os

Same as with windows.

## Using the package

This package is meant to be applied to point data for several entities and to extract leader-follower relations for those entities.

### Loading the data


### Ploting basic quantities


### Getting the leadership network


### Generating fake data


#### Two uncorrelated Poisson processes


#### One Poisson process and a non-homogeneous Poisson process correlated with the first one


#### A multivariate Hawkes process on an arbitrary network



