# Dirichlet Process Mixture Model of Semi-Projected Normal Distribution (DPSPN)

This project is based on the paper [A Dirichlet Process Mixture Model for Directional-Linear Data](https://arxiv.org/pdf/2212.10704.pdf) and aims to model the joint distribution of a directional variable and linear variables via a Dirichelt process mixture model of semi-projected normal distribution. The semi-projected normal (SPN) can be obtaind by projecting a few dimensions of a multivariate normal into a hysphere. The SPN is then used as the mixture distribution in a Dirichlet process model to obtain more flexibility. In terms of the prior, we propose a normal conditional inverse-Wishart distribution to form conjugacy to the SPN, and also resolve the issue of identifiability inherited from the projected normal. A Gibbs sampling algorithm is provided for the posterior inference. Here we provide a simple example of how the model can be applied to circular-linear data using the DPSPN.

# Prerequisites (R packages)

Rcpp, RcppArmadillo, mvtnorm, plotly, salso

# Instructions

1. Install all prerequisite packages
2. Install DPSPN
3. Run test/demo.R

# Note

The implementation is based on the kasparmartens/mixtureModels repository (https://github.com/kasparmartens/mixtureModels)
