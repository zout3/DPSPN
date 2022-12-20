#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <memory>

int sample(int K, const arma::vec& prob);

void chol_update(arma::mat& U, arma::vec& x);

void chol_downdate(arma::mat& U, arma::vec& x);

double loglik_marginal_CNIW(int N, int D1,
	double kappa, double nu, const arma::mat& S_chol,
	double kappa0, double nu0, const arma::mat& S0_chol);

double logdet_chol(const arma::mat& U);

void softmax_this(arma::vec& logx);

arma::vec rdirichlet(const arma::vec& alpha);

int rAntoniak(int n, double alpha);

void arrange_z(arma::uvec& z); // make z's elements 0, 1, 2, ...

double log_sum(double log_a, double log_b);

double log_subtract(double log_a, double log_b);

double nonzero_randu();

double log_mvnpdf(const arma::vec& x, const arma::vec& m, const arma::mat& Sinv);

double log_kappa(double p, double x);

#endif