#include "component.h"
#include "parameters_config.h"


Component::Component(int dim, int dim1) :
    N(0), D(dim), D1(dim1),
    kappa0(KAPPA_PRIOR), nu0(NU_PRIOR), m0(M_PRIOR),
    S0(S_PRIOR), U0(arma::chol(S0)),
    kappa(kappa0), nu(nu0), m(m0), S(S0), U(U0),
    mu(M_PRIOR), sigma(S_PRIOR), sigmainv(arma::inv(sigma)) {}


Component::Component(int dim, int dim1, const arma::mat& data) :
    N(0), D(dim), D1(dim1),
    kappa0(KAPPA_PRIOR), nu0(NU_PRIOR), m0(M_PRIOR),
    S0(S_PRIOR), U0(arma::chol(S0)),
    kappa(kappa0), nu(nu0), m(m0), S(S0), U(U0),
    mu(M_PRIOR), sigma(S_PRIOR), sigmainv(arma::inv(sigma))
{
    for (auto i = 0; i < data.n_rows; i++) {
        add_sample(data.row(i).t());
    }
}


void Component::add_sample(const arma::vec &x) {
    N++;
    kappa++;
    nu++;
    m = ((kappa - 1) * m + x) / kappa;
    arma::vec m_chol = (x - m) * sqrt(kappa / (kappa - 1));
    S += m_chol * m_chol.t();
    chol_update(U, m_chol);
}


void Component::rm_sample(const arma::vec& x) {
    N--;
    arma::vec m_chol = (x - m) * sqrt(kappa / (kappa - 1));
    kappa--;
    nu--;
    m = ((kappa + 1) * m - x) / kappa;
    S -= m_chol * m_chol.t();
    chol_downdate(U, m_chol);
}


double Component::marginal_loglik() const
{
  return loglik_marginal_CNIW(N, D1, kappa, nu, U, kappa0, nu0, U0);
}


double Component::posterior_predictive(const arma::vec& x) const
{
  arma::vec m_temp = sqrt(kappa / (kappa + 1)) * (x - m);
  arma::mat U_temp(U.memptr(), D, D);
  chol_update(U_temp, m_temp);
  return loglik_marginal_CNIW(1, D1, kappa + 1, nu + 1, U_temp, kappa, nu, U);
}


int Component::get_N () const {
    return N;
}


bool Component::is_empty() const {
    return (N == 0);
}

///////////////////////////////////////////////////////////////////////////////////////

#define Msub11(X) X.submat(0, 0, D1 - 1, D1 - 1)
#define Msub12(X) X.submat(0, D1, D1 - 1, D - 1)
#define Msub21(X) X.submat(D1, 0, D - 1, D1 - 1)
#define Msub22(X) X.submat(D1, D1, D - 1, D - 1)

void Component::update_mu_sigma()
{
    using namespace arma;
    mat U11inv = inv(trimatu(Msub11(U)));
    mat S11inv = U11inv * U11inv.t();
    mat S221 = Msub22(S) - Msub21(S) * S11inv * Msub12(S);
    S221 = (S221 + S221.t()) / 2.0;  // make it symmetric
    mat sigma221 = iwishrnd(S221, nu);
    sigma221 = (sigma221 + sigma221.t()) / 2.0;  // make it symmetric
    
    Msub12(sigma) = reshape(mvnrnd(vectorise(S11inv * Msub12(S)), kron(sigma221, S11inv)), D1, D - D1);
    Msub21(sigma) = Msub12(sigma).t();
    Msub22(sigma) = sigma221 + Msub21(sigma) * Msub12(sigma);
    sigmainv = inv_sympd(sigma);
    mu = mvnrnd(m, sigma / kappa);
}
