#include "mixture.h"
#include "parameters_config.h"

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

Mixture::Mixture(const arma::mat& data, const arma::uvec& clstr, int dim1, int dim_cir) :
    alpha_a(ALPHA_A_PRIOR), alpha_b(ALPHA_B_PRIOR), alpha(ALPHA_PRIOR),
    N(data.n_rows), D(data.n_cols), D1(dim1), Dcir(dim_cir),
    X(data), r(arma::randg(N)), z(clstr), prior_predict(N)
{
    arrange_z(z);
    K = z.max() + 1;
    X.cols(0, Dcir - 1) = normalise(X.cols(0, Dcir - 1), 2, 1);
    u = X.cols(0, Dcir - 1);

    for (int k = 0; k < K; k++)
      components.push_back(make_unique<Component>(D, D1, X.rows(find(z == k))));
      

    Component empty_component(D, D1);
    for (int i = 0; i < N; i++)
      prior_predict(i) = empty_component.posterior_predictive(X.row(i).t());
}

Mixture::Mixture(const arma::mat& data, int dim1, int dim_cir) :
  Mixture(data, arma::uvec(data.n_rows, arma::fill::ones), dim1, dim_cir) {}


void Mixture::add_sample(int i, int k) {
    z[i] = k;
    if (k > K - 1) {
        add_component();
    }
    components[k]->add_sample(X.row(i).t());
}


void Mixture::rm_sample(int i) {
    int k = z[i];
    components[k]->rm_sample(X.row(i).t());
    if (components[k]->is_empty()) {
        rm_component(k);
    }
}


void Mixture::add_component(const arma::uvec& ind) {
    z.elem(ind).fill(K);
    components.push_back(make_unique<Component>(D, D1, X.rows(ind)));
    K = K + 1;
}


void Mixture::add_component() {
    K = K + 1;
    components.push_back(make_unique<Component>(D, D1));
}


void Mixture::rm_component(int k) {
    components.erase(components.begin() + k);
    z.elem(find(z > k)) -= 1;
    K = K - 1;
}


void Mixture::update_z() {
    for (int i = 0; i < N; i++) {
        rm_sample(i);
        arma::vec logprobs(K + 1);
        for (int k = 0; k < K; k++) {
            logprobs[k] = log(components[k]->get_N()) + components[k]->posterior_predictive(X.row(i).t());
        }
        logprobs[K] = log(alpha) + prior_predict(i);
        softmax_this(logprobs);
        add_sample(i, sample(K, logprobs));
    }
}


void Mixture::update_alpha()
{
    double x = rdirichlet(arma::vec({ alpha + 1, (double)N }))[0];
    double p1 = alpha_a + K - 1;
    double p2 = N * (alpha_b - log(x));
    if (arma::randu() < p1 / (p1 + p2))
        alpha = arma::randg(arma::distr_param(alpha_a + K, 1 / (alpha_b - log(x))));
    else
        alpha = arma::randg(arma::distr_param(alpha_a + K - 1, 1 / (alpha_b - log(x))));
}


double Mixture::get_loglik()
{
    double res = 0;
    for (int k = 0; k < K; k++) {
        res += components[k]->marginal_loglik();
    }
    return res;
}


void Mixture::update_data()
{
    for (int k = 0; k < K; k++) {
        arma::uvec index = arma::find(z == k);
        if (!index.empty()) {
            components[k]->update_mu_sigma();
            arma::mat sinv11 = components[k]->sigmainv.submat(0, 0, Dcir - 1, Dcir - 1);
            arma::mat sinv12 = components[k]->sigmainv.submat(0, Dcir, Dcir - 1, D - 1);
            arma::vec mu1 = components[k]->mu.subvec(0, Dcir - 1);
            arma::vec mu2 = components[k]->mu.subvec(Dcir, D - 1);
            for (auto i : index)
            {
                components[k]->rm_sample(X.row(i).t());
                double A = arma::as_scalar(u.row(i) * sinv11 * u.row(i).t());
                double B = arma::as_scalar(u.row(i) * sinv11 * mu1);
                B -= arma::as_scalar(u.row(i) * sinv12 * (X.row(i).subvec(Dcir, D - 1).t() - mu2)); // problem
                double logv = log(nonzero_randu()) - A * pow(r(i) - B / A, 2) / 2;
                double rho1 = B / A - std::min(B / A, sqrt(-2 * logv / A));
                double rho2 = B / A + sqrt(-2 * logv / A);
                r(i) = pow((pow(rho2, Dcir) - pow(rho1, Dcir)) * arma::randu() + pow(rho1, Dcir), 1.0 / Dcir);
                X(i, arma::span(0, Dcir - 1)) = r(i) * u(i, arma::span(0, Dcir - 1));
                components[k]->add_sample(X.row(i).t());
            }
        }
    }
}

