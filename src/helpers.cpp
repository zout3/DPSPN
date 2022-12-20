#include "helpers.h"
#include <RcppArmadilloExtensions/sample.h>

#define SubMat(X, k) X.submat(0, 0, k - 1, k - 1)
#define TraceChol(A) arma::accu(arma::square(A))

const double log_pi = log(3.141592653589793238462643383280);
const double log_2 = log(2.0);


int sample(int K, const arma::vec& prob) {
  return Rcpp::RcppArmadillo::sample(arma::regspace<arma::uvec>(0, K), 1, 0, prob)[0];
}


void chol_update(arma::mat& U, arma::vec& x) {
    int n = x.size();
    for (int k = 0; k < n; k++) {
        double r = sqrt(U(k, k) * U(k, k) + x[k] * x[k]);
        double c = r / U(k, k);
        double s = x[k] / U(k, k);
        U(k, k) = r;
        for (int j = k + 1; j < n; j++) {
            U(k, j) = (U(k, j) + s * x[j]) / c;
            x[j] = c * x[j] - s * U(k, j);
        }
    }
}


void chol_downdate(arma::mat& U, arma::vec& x) {
    int n = x.size();
    for (int k = 0; k < n; k++) {
        double r = sqrt(U(k, k) * U(k, k) - x[k] * x[k]);
        if (std::isnan(r)) {
            printf("Error: chol downdate problem!\n");
        }
        double c = r / U(k, k);
        double s = x[k] / U(k, k);
        U(k, k) = r;
        for (int j = k + 1; j < n; j++) {
            U(k, j) = (U(k, j) - s * x[j]) / c;
            x[j] = c * x[j] - s * U(k, j);
        }
    }
}


double loglik_marginal_CNIW(int N, int D1,
    double kappa, double nu, const arma::mat& S_chol,
    double kappa0, double nu0, const arma::mat& S0_chol)
{
    int D = S_chol.n_rows;
    int D2 = D - D1;
    double res = - 0.5 * N * D * log_pi
        - 0.5 * N * D1 * log_2
        - 0.5 * D * (log(kappa) - log(kappa0))
        - 0.5 * (nu * logdet_chol(S_chol) - nu0 * logdet_chol(S0_chol))
        - 0.5 * (D2 - nu) * logdet_chol(SubMat(S_chol, D1))
        + 0.5 * (D2 - nu0) * logdet_chol(SubMat(S0_chol, D1))
        - 0.5 * (TraceChol(SubMat(S_chol, D1)) - TraceChol(SubMat(S0_chol, D1)));
    for (int d = 1; d <= D2; d++) {
        res += lgamma(0.5 * (nu - d + 1)) - lgamma(0.5 * (nu0 - d + 1));
    }
    return res;
}


double logdet_chol(const arma::mat& U) {
    return 2 * sum(log(U.diag()));
}

void softmax_this(arma::vec& logx) {
    logx -= logx.max();
    for (double& x : logx)
        x = exp(x);
    logx /= sum(logx);
}

arma::vec rdirichlet(const arma::vec& alpha)
{
    size_t n = alpha.n_elem;
    arma::vec res(n);
    for (int i = 0; i < n; i++)
    {
        res[i] = arma::randg(arma::distr_param(alpha[i], 1.0));
    }
    return res / sum(res);
}

int rAntoniak(int n, double alpha)
{
    int res = n > 0 ? 1 : 0;
    for (int i = 1; i < n; i++)
    {
        if (arma::randu() < alpha / (i + alpha))
            res++;
    }
    return res;
}


void arrange_z(arma::uvec& z)  // make z's elements 0, 1, 2, ...
{
    std::map<unsigned int, unsigned int> dict;
    size_t count = 0;
    for (auto& x : z) {
        if (dict.find(x) == dict.end()) {
            dict[x] = count;
            x = count;
            count++;
        }
        else
            x = dict[x];
    }
}


double log_sum(double log_a, double log_b) {
    double v;

    if (log_a < log_b)
        v = log_b + log(1 + exp(log_a - log_b));
    else
        v = log_a + log(1 + exp(log_b - log_a));
    return v;
}

double log_subtract(double log_a, double log_b)
{
    if (log_a < log_b)
      Rcpp::stop("log_subtract(a, b): a smaller than b");

    double v;
    v = log_a + log(1 - exp(log_b - log_a));
    return v;
}


double nonzero_randu()
{
    double ru = arma::randu();
    return ru == 0 ? nonzero_randu() : ru;
}


double log_mvnpdf(const arma::vec& x, const arma::vec& m, const arma::mat& Sinv)
{
    double res = 0.0;
    arma::vec x_m = x - m;
    res -= (log_pi + log_2) * x.n_elem;
    res -= arma::as_scalar(x_m.t() * Sinv * x_m);
    res += log(arma::det(Sinv));
    return 0.5 * res;
}


double log_kappa(double p, double x)
{
    double lgk1 = R::pnorm(x,0.0,1.0,1,1) + 0.5*(log_pi+log_2);
    if (p == 1) return lgk1;
    double logx = x > 0 ? log(x) : log(-x);
    double lgk2;
    if(x > 0)
      lgk2 = log_sum(-0.5*x*x, logx+lgk1);
    else
      lgk2 = log_subtract(-0.5*x*x, logx+lgk1);
    if (p == 2) return lgk2;
    
    double d = 2;
    while(d<p){
      d++;
      double lgkTemp;
      if(x > 0)
        lgkTemp = log_sum(lgk1+log(d-2), logx+lgk2);
      else
        lgkTemp = log_subtract(lgk1+log(d-2), logx+lgk2);
      lgk1 = lgk2;
      lgk2 = lgkTemp;
    }
    return lgk2;
}


double log_kappa(int p, double x)
{
  if (p == 1) return R::pnorm(x,0.0,1.0,1,1) + 0.5*(log_pi+log_2);
  if (p == 2){
    if(x > 0)
      return log_sum(-0.5 * x * x, log(x) + log_kappa(p - 1, x));
    else
      return log_subtract(-0.5 * x * x, log(-x) + log_kappa(p - 1, x));
  }
  if(x > 0)
    return log_sum(log_kappa(p - 2, x)+log(p-2), log(x)+log_kappa(p - 1, x));
  else
    return log_subtract(log_kappa(p - 2, x)+log(p-2), log(-x)+log_kappa(p - 1, x));
}