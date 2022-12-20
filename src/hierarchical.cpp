#include "hierarchical.h"
#include "parameters_config.h"

Hierarchical::Hierarchical(const arma::mat& data, const arma::uvec& rstrt_id,
    const arma::uvec& clstr, int dim1, int dim_cir) :
    Mixture(data, clstr, dim1, dim_cir), gamma(GAMMA_PRIOR), rstrt(rstrt_id)
{
    arrange_z(rstrt);
    J = rstrt.max() + 1;
    arma::vec temp(K + 1);
    temp.fill(gamma / (K + 1));
    beta = arma::conv_to<stdvec>::from(rdirichlet(temp));
    for (int k = 0; k < K; k++) {
        n_kj.push_back(arma::uvec(J, arma::fill::zeros));
        m_kj.push_back(arma::uvec(J, arma::fill::zeros));
        for (int j = 0; j < J; j++)
            n_kj[k][j] = sum(rstrt.elem(find(z == k)) == j);
    }
    update_m();
    update_beta();
}


Hierarchical::Hierarchical(const arma::mat& data, const arma::uvec& rtrt_id, int dim1, int dim_cir) :
    Hierarchical(data, rtrt_id, arma::uvec(data.n_rows, arma::fill::ones), dim1, dim_cir) {}


void Hierarchical::add_sample(int i, int k) {
    Mixture::add_sample(i, k);
    n_kj[k][rstrt[i]] += 1;
}


void Hierarchical::rm_sample(int i) {
    n_kj[z[i]][rstrt[i]] -= 1;
    Mixture::rm_sample(i);
}


void Hierarchical::add_component() {
    Mixture::add_component();
    n_kj.push_back(arma::uvec(J, arma::fill::zeros));
    m_kj.push_back(arma::uvec(J, arma::fill::zeros));
    beta.push_back(0);
}


void Hierarchical::rm_component(int k) {
    n_kj.erase(n_kj.begin() + k);
    m_kj.erase(m_kj.begin() + k);
    beta.erase(beta.begin() + k);
    Mixture::rm_component(k);
}


void Hierarchical::update_z()
{
    for (int i = 0; i < N; i++) {
        rm_sample(i);
        arma::vec logprobs(K + 1);
        for (int k = 0; k < K; k++) {
            logprobs[k] = log(n_kj[k][rstrt[i]] + alpha * beta[k]) +
                components[k]->posterior_predictive(X.row(i).t());
        }
        logprobs[K] = log(alpha * beta[K]) + prior_predict(i);
        softmax_this(logprobs);
        add_sample(i, sample(K, logprobs));
    }
}


void Hierarchical::update_m() {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < J; j++) {
            m_kj[k][j] = rAntoniak(n_kj[k][j], alpha * beta[k]);
        }
    }
}


void Hierarchical::update_beta() {
    arma::vec m(K + 1);
    for (int k = 0; k < K; k++) {
        m[k] = sum(m_kj[k]);
    }
    m[K] = gamma;
    beta = arma::conv_to<stdvec>::from(rdirichlet(m));
}


void Hierarchical::gibbs_sampling()
{
    update_z();
    update_data();
    update_m();
    update_beta();
}


void Hierarchical::update_alpha()
{
    arma::uvec n_j(J, arma::fill::zeros);
    double m = 0;
    for (int k = 0; k < K; k++)
    {
        n_j += n_kj[k];
        m += sum(m_kj[k]);
    }

    arma::uvec s(J);
    arma::vec w(J);
    for (int j = 0; j < J; j++)
    {
        s[j] = arma::randu() < (n_j[j] / (alpha + n_j[j])) ? 1 : 0;
        w[j] = rdirichlet(arma::vec({ alpha + 1, (double)n_j[j] }))[0];
    }
    alpha = arma::randg(arma::distr_param(alpha_a + m - sum(s), 1.0 / (alpha_b - sum(log(w)))));
}

