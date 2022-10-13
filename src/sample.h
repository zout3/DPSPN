#pragma once
#include <armadillo>

//Modified sample_main from RcppArmadilloExtensions/sample.h

arma::uvec sample(const arma::uvec& x, const int size, const arma::vec& prob);

arma::uvec sample(const arma::uvec& x, const int size);
