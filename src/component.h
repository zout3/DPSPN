#ifndef COMPONENT_H
#define COMPONENT_H

#include "helpers.h"

class Component {

//protected:
public:
    int N;
    int D;
    int D1;

    double kappa0;
    double nu0;
    arma::vec m0;
    arma::mat S0;
    arma::mat U0;

    double nu;
    double kappa;
    arma::vec m;
    arma::mat S;
    arma::mat U;

public:
    Component(int dim, int dim1);
    Component(int dim, int dim1, const arma::mat& data);
    
    arma::vec mu;
    arma::mat sigma;
    arma::mat sigmainv;
    
    void add_sample(const arma::vec& x);
    void rm_sample(const arma::vec& x);

    double marginal_loglik() const;
    double posterior_predictive(const arma::vec& x) const;
    
    void update_mu_sigma();

    int get_N() const;
    bool is_empty() const;
};

#endif
