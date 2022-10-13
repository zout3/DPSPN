# Dirichlet Process Mixture Model of Semi-Projected Normal Distribution (DPSPN)

This project aims to model the joint distribution of a directional variable and linear variables via a Dirichelt process mixture model of semi-projected normal distribution. The semi-projected normal (SPN) can be obtaind by projecting a few dimensions of a multivariate normal into a hysphere. The SPN is then used as the mixture distribution in a Dirichlet process model to obtain more flexibility. In terms of the prior, we propose a normal conditional inverse-Wishart distribution to form conjugacy to the SPN, and also resolve the issue of identifiability inherited from the projected normal. A Gibbs sampling algorithm is provided for the posterior inference. Here we provide a simple example of how the model can be applied to circular-linear data using the DPSPN.

# Prerequisites

Armadillo library to run the cpp code for MCMC

# Instructions

1. Run generateData.py to generate and visualize circular-linear data (dat/data)

![GitHub data](/img/Data scatter.png)
*Data*

2. Change macro DAT_DIR in src/parameters_config.h to set up the correct directory
3. Run main.cpp to obtain the log-likelihood (dat/loglik) and posterior clustering (dat/z) of the data
4. Run visualizeResult.py to visualize the posterior clustering

![GitHub loglik](/img/Log likelihood.png)
*Log likelihood*

![GitHub clstr](/img/Clustering Result.png)
*Clustering Result*
