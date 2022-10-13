#pragma once
#include <armadillo>

class Component {

//protected:
public:
    int N;
    int D;

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
    Component() {}
    Component(int dim);
    Component(int dim, const arma::mat& data);

    void add_sample(const arma::vec& x);
    void rm_sample(const arma::vec& x);

    virtual double marginal_loglik() const;
    virtual double posterior_predictive(const arma::vec& x) const;

    int get_N() const;
    bool is_empty() const;

};


class CircularComponent : public Component
{
//private:
public:
    int D1;

public:
    arma::vec mu;
    arma::mat sigma;
    arma::mat sigmainv;

    CircularComponent() {}
    CircularComponent(int dim, int dim1)
        : Component(dim), D1(dim1), sigma(arma::eye(dim, dim)), sigmainv(arma::eye(dim, dim)) {};
    CircularComponent(int dim, const arma::mat& data, int dim1)
        : Component(dim, data), D1(dim1), sigma(arma::eye(dim, dim)), sigmainv(arma::eye(dim, dim)) {};

    double marginal_loglik() const override;
    double posterior_predictive(const arma::vec& x) const override;

    void update_mu_sigma();

};
