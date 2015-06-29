## Overview

**Idea:** Trying to get the maximum from datasets where we have very few
training examples, but each example has a very large number of features.
ExampleTanh of such datasets include medical databases where we have gene
activation measurements for very few patients but many different genes.

**Method:** We design a neural network architecture whose number of parameters
is constant with respect to the number of features (which is not the case with a
typical linear classifier). The basic idea is that we use a linear classifier
whose coefficients for each features are generated by a single MLP that takes as
an input a representation for this feature, which is basically a transformation
of the set of values taken by this feature through all the examples. More
complex (deep) architectures are also experimented.

## Datasets

- ICML 2003 feature selection challenge datasets: Arcene, Dorothea
- AML/ALL Leukemia classification dataset

## Details

See `doc/README.pdf` (soon) for detailed explanations.
