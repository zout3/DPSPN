#include "circular_dp.h"
#include "sample.h"
#include "parameters_config.h"


CircularDP::CircularDP(const arma::mat& data, const arma::uvec& clstr, int dim1, int dim_cir) :
    alpha_a(ALPHA_A_PRIOR), alpha_b(ALPHA_B_PRIOR), alpha(ALPHA_PRIOR),
    N(data.n_rows), D(data.n_cols), D1(dim1), Dcir(dim_cir),
    X(data), r(N, arma::fill::ones), z(clstr), prior_predict(N)
{
    arrange_z(z);
    K = z.max() + 1;
    X.cols(0, Dcir - 1) = normalise(X.cols(0, Dcir - 1), 2, 1);
    u = X.cols(0, Dcir - 1);

    for (int k = 0; k < K; k++)
        components.push_back(std::make_unique<CircularComponent>(D, X.rows(find(z == k)), D1));

    CircularComponent empty_component(D, D1);
    for (int i = 0; i < N; i++)
        prior_predict(i) = empty_component.posterior_predictive(X.row(i).t());
}

CircularDP::CircularDP(const arma::mat& data, int dim1, int dim_cir) :
    CircularDP(data, arma::uvec(data.n_rows, arma::fill::ones), dim1, dim_cir) {}


void CircularDP::add_sample(int i, int k) {
    z[i] = k;
    if (k > K - 1) {
        add_component();
    }
    components[k]->add_sample(X.row(i).t());
}


void CircularDP::rm_sample(int i) {
    int k = z[i];
    components[k]->rm_sample(X.row(i).t());
    if (components[k]->is_empty()) {
        rm_component(k);
    }
}


void CircularDP::add_component(const arma::uvec& ind) {
    z.elem(ind).fill(K);
    components.push_back(std::make_unique<CircularComponent>(D, X.rows(ind), D1));
    K = K + 1;
}


void CircularDP::add_component() {
    K = K + 1;
    components.push_back(std::make_unique<CircularComponent>(D, D1));
}


void CircularDP::rm_component(int k) {
    components.erase(components.begin() + k);
    z.elem(find(z > k)) -= 1;
    K = K - 1;
}


void CircularDP::update_z() {
    for (int i = 0; i < N; i++) {
        rm_sample(i);
        arma::vec logprobs(K + 1);
        //arma::vec logprobs(K);
        for (int k = 0; k < K; k++) {
            logprobs[k] = log(components[k]->get_N()) + components[k]->posterior_predictive(X.row(i).t());
        }
        logprobs[K] = log(alpha) + prior_predict(i);
        softmax_this(logprobs);
        int k0 = sample(arma::regspace<arma::uvec>(0, K), 1, logprobs)[0];
        //int k0 = sample(arma::regspace<arma::uvec>(0, K - 1), 1, logprobs)[0];
        add_sample(i, k0);
    }
}


void CircularDP::split_merge() {
    int Nmm = N - 1;
    arma::uvec sequence = arma::regspace<arma::uvec>(0, Nmm);
    arma::uvec indexes = sample(sequence, 2);
    int i = indexes[0];
    int j = indexes[1];
    if (z[i] == z[j]) {
        propose_split(i, j);
    }
    else {
        propose_merge(i, j);
    }
}


void CircularDP::propose_split(int i, int j) {
    CircularComponent S_i(D, D1);
    CircularComponent S_j(D, D1);
    S_i.add_sample(X.row(i).t());
    S_j.add_sample(X.row(j).t());
    arma::uvec S_ind = find(z == z[i]);
    int n_elements = S_ind.size();
    arma::uvec permutation = sample(S_ind, n_elements);
    arma::uvec temp_z;
    temp_z.zeros(N);
    temp_z[i] = 1;
    temp_z[j] = 2;
    double MH_logratio = 0.0;
    for (int k = 0; k < n_elements; k++) {
        int index = permutation[k];
        if (index == i || index == j) {
            // do nothing
        }
        else {
            arma::vec x = X.row(index).t();
            double p_i = S_i.get_N() * exp(S_i.posterior_predictive(x));
            double p_j = S_j.get_N() * exp(S_j.posterior_predictive(x));
            double prob_i = p_i / (p_i + p_j);
            if (arma::randu() < prob_i) {
                S_i.add_sample(x);
                temp_z[index] = 1;
                MH_logratio += log(prob_i);
            }
            else {
                S_j.add_sample(x);
                temp_z[index] = 2;
                MH_logratio += log(1 - prob_i);
            }
        }
    }
    double logprob_proposed = S_i.marginal_loglik() + S_j.marginal_loglik();
    double logprob_current = components[z[i]]->marginal_loglik();
    MH_logratio = logprob_proposed - logprob_current - MH_logratio;
    MH_logratio += log(alpha) + lgamma(S_i.get_N()) + lgamma(S_j.get_N()) - lgamma(S_i.get_N() + S_j.get_N());

    if (arma::randu() < exp(MH_logratio)) {
        int prev_z_i = z[i];
        add_component(find(temp_z == 1));
        add_component(find(temp_z == 2));
        rm_component(prev_z_i);
        n_split++;
    }
}


void CircularDP::propose_merge(int i, int j) {
    arma::uvec S_ind = find((z == z[i]) + (z == z[j]) == 1);
    arma::mat X_merge = X.rows(S_ind);
    CircularComponent S_merged(D, X_merge, D1);

    CircularComponent S_i(D, D1);
    CircularComponent S_j(D, D1);
    S_i.add_sample(X.row(i).t());
    S_j.add_sample(X.row(j).t());
    // arma::uvec S_ind = find(z == z[i]);
    int n_elements = S_ind.size();
    arma::uvec permutation = sample(S_ind, n_elements);
    // arma::uvec temp_z(N).zeros();
    // temp_z[i] = 1;
    // temp_z[j] = 2;
    double MH_logratio = 0.0;
    for (int k = 0; k < n_elements; k++) {
        int index = permutation[k];
        if (index == i || index == j) {
            // do nothing
        }
        else {
            arma::vec x = X.row(index).t();
            double p_i = S_i.get_N() * exp(S_i.posterior_predictive(x));
            double p_j = S_j.get_N() * exp(S_j.posterior_predictive(x));
            double prob_i = p_i / (p_i + p_j);
            if (z[index] == z[i]) {
                S_i.add_sample(x);
                // temp_z[index] = 1;
                MH_logratio += log(prob_i);
            }
            else if (z[index] == z[j]) {
                S_j.add_sample(x);
                // temp_z[index] = 2;
                MH_logratio += log(1 - prob_i);
            }
            else {
                printf("something went wrong\n");
            }
        }
    }
    double logprob_proposed = S_merged.marginal_loglik();
    double logprob_current = S_i.marginal_loglik() + S_j.marginal_loglik();
    MH_logratio = logprob_proposed - logprob_current + MH_logratio;
    MH_logratio += -log(alpha) - lgamma(S_i.get_N()) - lgamma(S_j.get_N()) + lgamma(S_merged.get_N());
    if (arma::randu() < exp(MH_logratio)) {
        int prev1 = std::min(z[i], z[j]);
        int prev2 = std::max(z[i], z[j]);
        add_component(S_ind);
        rm_component(prev2);
        rm_component(prev1);
        n_merge++;
    }
}


void CircularDP::update_alpha()
{
    double x = rdirichlet(arma::vec({ alpha + 1, (double)N }))[0];
    double p1 = alpha_a + K - 1;
    double p2 = N * (alpha_b - log(x));
    if (arma::randu() < p1 / (p1 + p2))
        alpha = arma::randg(arma::distr_param(alpha_a + K, 1 / (alpha_b - log(x))));
    else
        alpha = arma::randg(arma::distr_param(alpha_a + K - 1, 1 / (alpha_b - log(x))));
}


double CircularDP::get_loglik()
{
    double res = 0;
    for (int k = 0; k < K; k++) {
        res += components[k]->marginal_loglik();
    }
    return res;
}


void CircularDP::update_data()
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
                if (isnan(r(i)))
                {
                    std::cout << "X.row(i): " << X.row(i) << std::endl;
                    std::cout << "mu: " << components[k]->mu << std::endl;
                    std::cout << "sigmainv: " << components[k]->sigmainv << std::endl;
                    std::cout << "A: " << A << std::endl;
                    std::cout << "B: " << B << std::endl;
                    std::cout << "logv: " << logv << std::endl;
                }
            }
        }
    }
}

