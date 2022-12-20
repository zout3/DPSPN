#include "hierarchical.h"

const double log_2pi = log(2 * 3.141592653589793238462643383280);

void progessbar(int i, int n)
{
  if (n >= 100 && (i + 1) % (n / 100) != 0)
    return;
  double progress = (double)(i + 1) / n;
  Rcpp::Rcout.flush();
  if (progress <= 1.0) {
    static int barWidth = 50;
    Rcpp::Rcout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) Rcpp::Rcout << "=";
      else if (i == pos) Rcpp::Rcout << ">";
      else Rcpp::Rcout << " ";
    }
    Rcpp::Rcout << "] " << int(progress * 100.0) << " %\r";
  }
  if (progress == 1.0)
    Rcpp::Rcout << "\n";
}

// [[Rcpp::export]]
Rcpp::List fitDPSPN(Rcpp::NumericMatrix& data, int D1, int Dp,
                    Rcpp::IntegerVector clstrAssign = Rcpp::IntegerVector::create(-1),
                    int nSamp = 50, int nBurn = 900, int nThin = 2,
                    bool updateAlpha = 0, bool displayProgress = 1)
{
  arma::mat X = Rcpp::as<arma::mat>(Rcpp::wrap(data));
  arma::uvec c;
  if (clstrAssign[0] == -1)
    c = arma::randi<arma::uvec>(X.n_rows, arma::distr_param(1, std::min(5, (int)X.n_rows)));
  else
    c = Rcpp::as<arma::uvec>(Rcpp::wrap(clstrAssign));
  
  arma::umat z(X.n_rows, nSamp);
  arma::mat r(X.n_rows, nSamp);
  std::vector<std::vector<Rcpp::NumericMatrix>> sigma;
  std::vector<std::vector<Rcpp::NumericVector>> mu;
  
  Mixture mod(X, c, D1, Dp);
  int nIter = nBurn + nSamp * nThin;
  arma::vec loglik(nIter);
  arma::vec alpha(nIter);
  arma::uvec K(nIter);
  int count = 0;
  for (int i = 0; i < nIter; i++)
  {
    mod.update_z();
    mod.update_data();
    if (updateAlpha)
      mod.update_alpha();
    if (i >= nBurn && (i-nBurn)%nThin == 0) {
      std::vector<Rcpp::NumericMatrix> s;
      std::vector<Rcpp::NumericVector> m;
      s.resize(mod.K);
      m.resize(mod.K);
      for(int k = 0; k < mod.K; k++){
        s[k] = Rcpp::wrap(mod.components[k]->sigma);
        m[k] = Rcpp::wrap(mod.components[k]->mu);
      }
      sigma.push_back(s);
      mu.push_back(m);
      z.col(count) = mod.z;
      r.col(count) = mod.r;
      count += 1;
    }
    alpha[i] = mod.alpha;
    loglik[i] = mod.get_loglik();
    K[i] = mod.K;
    
    if(displayProgress)
      progessbar(i, nIter);
  }
  return Rcpp::List::create(
    Rcpp::_["z"] = z,
    Rcpp::_["r"] = r,
    Rcpp::_["loglik"] = loglik,
    Rcpp::_["K"] = K,
    Rcpp::_["alpha"] = alpha,
    Rcpp::_["param"] = Rcpp::List::create(
      Rcpp::_["mu"] = mu,
      Rcpp::_["sigma"] = sigma
    )
  );
}



// [[Rcpp::export]]
Rcpp::List fitHDPSPN(Rcpp::NumericMatrix& data, Rcpp::IntegerVector& rid, int D1, int Dp,
                     Rcpp::IntegerVector clstrAssign = Rcpp::IntegerVector::create(-1),
                     int nSamp = 50, int nBurn = 900, int nThin = 2,
                     bool updateAlpha = 0, bool displayProgress = 1)
{
  arma::mat X = Rcpp::as<arma::mat>(Rcpp::wrap(data));
  arma::uvec Xid = Rcpp::as<arma::uvec>(Rcpp::wrap(rid));
  arma::uvec c;
  if (clstrAssign[0] == -1)
    c = arma::randi<arma::uvec>(X.n_rows, arma::distr_param(1, std::min(5, (int)X.n_rows)));
  else
    c = Rcpp::as<arma::uvec>(Rcpp::wrap(clstrAssign));
  
  arma::umat z(X.n_rows, nSamp);
  arma::mat r(X.n_rows, nSamp);
  std::vector<std::vector<Rcpp::NumericMatrix>> sigma;
  std::vector<std::vector<Rcpp::NumericVector>> mu;
  std::vector<Rcpp::NumericVector> beta;
  
  Hierarchical mod(X, Xid, c, D1, Dp);
  int nIter = nBurn + nSamp * nThin;
  arma::vec loglik(nIter);
  arma::vec alpha(nIter);
  arma::uvec K(nIter);
  int count = 0;
  for (int i = 0; i < nIter; i++)
  {
    mod.gibbs_sampling();
    if (updateAlpha)
      mod.update_alpha();
    if (i >= nBurn && (i-nBurn)%nThin == 0) {
      std::vector<Rcpp::NumericMatrix> s;
      std::vector<Rcpp::NumericVector> m;
      s.resize(mod.K);
      m.resize(mod.K);
      for(int k = 0; k < mod.K; k++){
        s[k] = Rcpp::wrap(mod.components[k]->sigma);
        m[k] = Rcpp::wrap(mod.components[k]->mu);
      }
      sigma.push_back(s);
      mu.push_back(m);
      beta.push_back(Rcpp::wrap(mod.beta));
      z.col(count) = mod.z;
      r.col(count) = mod.r;
      count += 1;
    }
    alpha[i] = mod.alpha;
    loglik[i] = mod.get_loglik();
    K[i] = mod.K;
    
    if(displayProgress)
      progessbar(i, nIter);
  }
  return Rcpp::List::create(
    Rcpp::_["z"] = z,
    Rcpp::_["r"] = r,
    Rcpp::_["loglik"] = loglik,
    Rcpp::_["K"] = K,
    Rcpp::_["alpha"] = alpha,
    Rcpp::_["param"] = Rcpp::List::create(
      Rcpp::_["mu"] = mu,
      Rcpp::_["sigma"] = sigma,
      Rcpp::_["beta"] = beta
    )
  );
}


// [[Rcpp::export]]
Rcpp::List computeLoglikhd(Rcpp::NumericMatrix& data,
                           int Dcir, Rcpp::List& param) {
  int Nmcmc = 100;
  Rcpp::List mu = param["mu"];
  Rcpp::List sigma = param["sigma"];
  Rcpp::List beta = param["beta"];
  arma::mat X = Rcpp::as<arma::mat>(Rcpp::wrap(data));
  
  int NIter = mu.size();
  arma::vec loglikhd(NIter);
  loglikhd.fill(-1*arma::datum::inf);
  X.cols(0, Dcir - 1) = normalise(X.cols(0, Dcir - 1), 2, 1);
  arma::mat u(X.cols(0, Dcir - 1));
  
  for(int iter = 0; iter < NIter; iter++){
    Rcpp::List musub = mu[iter];
    Rcpp::List sigmasub = sigma[iter];
    Rcpp::NumericVector betasub = beta[iter];
    int K = musub.size();
    arma::mat loglikhdMat(K, X.n_rows);
    arma::vec b(K);
    for (int k = 0; k < K; k++)
      b[k] = betasub[k];
    b /= arma::sum(b);
    for(int k = 0; k < K; k++){
      Rcpp::NumericVector musubsub = musub[k];
      Rcpp::NumericMatrix sigmasubsub = sigmasub[k];
      int D = musubsub.size();
      
      arma::mat s = Rcpp::as<arma::mat>(sigmasubsub);
      arma::mat sinv = arma::inv_sympd(s);
      arma::vec m = Rcpp::as<arma::vec>(musubsub);
      
      arma::mat sinv11 = sinv.submat(0, 0, Dcir - 1, Dcir - 1);
      arma::mat s12 = s.submat(0, Dcir, Dcir - 1, D - 1);
      arma::mat s22inv = arma::inv_sympd(s.submat(Dcir, Dcir, D - 1, D - 1));
      arma::vec mu1 = m.subvec(0, Dcir - 1);
      arma::vec mu2 = m.subvec(Dcir, D - 1);
      for(int i = 0; i < X.n_rows; i++) {
        arma::vec X2 = X.row(i).subvec(Dcir, D - 1).t();
        arma::vec mu12 = mu1 + s12 * s22inv * (X2 - mu2);
        double C = arma::as_scalar(mu12.t() * sinv11 * mu12);
        double A = arma::as_scalar(u.row(i) * sinv11 * u.row(i).t());
        double B = arma::as_scalar(u.row(i) * sinv11 * mu12);
        double BAsqrt = B / sqrt(A);
        double marginal = log_mvnpdf(X2, mu2, s22inv);
        marginal += log_kappa(Dcir, BAsqrt);
        marginal -= 0.5 * log_2pi * Dcir - 0.5 * log(arma::det(sinv11));
        marginal -= 0.5 * (log(A) * Dcir + C - BAsqrt * BAsqrt);
        loglikhdMat(k, i) = marginal;
      }
    }
    
    for(int i=0; i < Nmcmc; i++){
      arma::mat M = loglikhdMat.each_col() + rdirichlet(b);
      arma::rowvec cMax = arma::max(M);
      double loglikhd_i = arma::sum(cMax+arma::log(arma::sum(arma::exp(M.each_row()-cMax))));
      loglikhd[iter] = log_sum(loglikhd[iter], loglikhd_i);
    }
  }
  
  double loglikhdMean = loglikhd[0];
  for(int i = 1; i < NIter; i++)
    loglikhdMean = log_sum(loglikhdMean, loglikhd[i]);
  return Rcpp::List::create(
    Rcpp::_["loglikhd"] = loglikhd, 
    Rcpp::_["loglikhdMean"] = loglikhdMean - log(NIter)
  );
}


