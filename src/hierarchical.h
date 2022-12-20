#ifndef HIERARCHICAL_H
#define HIERARCHICAL_H
#include "mixture.h"

typedef std::vector<double> stdvec;

class Hierarchical : public Mixture
{
public:
	double gamma;

	arma::uvec rstrt;   // restaurant id
	int J;              // number of restaurants

	stdvec beta;  // atom weights of G0
	std::vector<arma::uvec> n_kj; // number of customers belonging to component k in restaurant j
	std::vector<arma::uvec> m_kj; // number of tables belonging to component k in restaurant j

	Hierarchical(const arma::mat& data, const arma::uvec& rtrt_id,
		const arma::uvec& clstr, int dim1, int dim_cir);
	Hierarchical(const arma::mat& data, const arma::uvec& rtrt_id, int dim1, int dim_cir);

	void add_sample(int i, int k);
	void rm_sample(int i);
	void add_component() override;
	void rm_component(int k) override;

	void update_z();
	void update_m();
	void update_beta();

	void gibbs_sampling();
	void update_alpha();
};

#endif