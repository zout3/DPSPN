#pragma once
#include "helpers.h"
#include "component.h"
#include <vector>


class CircularDP {
	typedef std::vector<std::unique_ptr<CircularComponent>> container;

public:
	double alpha_a;
	double alpha_b;
	double alpha;

	int N;
	int D;
	int D1;
	int Dcir;
	arma::mat X;
	arma::mat u;
	arma::vec r;

	arma::uvec z;
	int K;

	container components;
	arma::vec prior_predict;

	CircularDP() {}
	CircularDP(const arma::mat& data, const arma::uvec& clstr, int dim1, int dim_cir);
	CircularDP(const arma::mat& data, int dim1, int dim_cir);

	void add_sample(int i, int k);
	void rm_sample(int i);

	void add_component(const arma::uvec& ind);
	virtual void add_component();
	virtual void rm_component(int k);

	void update_z();

	void split_merge();
	void propose_split(int i, int j);
	void propose_merge(int i, int j);

	void update_alpha();
	double get_loglik();

	void update_data();

	int n_split = 0;
	int n_merge = 0;
};

