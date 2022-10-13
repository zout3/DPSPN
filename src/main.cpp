#include <iostream>
#include <armadillo>
#include "helpers.h"
#include "circular_dp.h"
#include "parameters_config.h"

using namespace std;
void progessbar(double progress)
{
	cout.flush();
	if (progress <= 1.0) {
		static int barWidth = 50;
		cout << "[";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) cout << "=";
			else if (i == pos) cout << ">";
			else cout << " ";
		}
		cout << "] " << int(progress * 100.0) << " %\r";
	}
	if (progress == 1.0)
		cout << endl;
}

int main()
{
	arma::arma_rng::set_seed_random();

	std::string filename("data");

	arma::mat X;
	arma::vec loglik(N_ITER);
	arma::vec alpha(N_ITER);

	X.load(DAT_DIR + filename, arma::raw_ascii);
	CircularDP mod(X, arma::uvec(X.n_rows, arma::fill::ones), DIM_d1, DIM_p);
	arma::umat z(X.n_rows, (N_ITER - N_BURN) / N_THIN);

	time_t tstart, tend;
	tstart = time(0);

	for (int i = 0; i < N_ITER; i++)
	{
		mod.update_z();
		mod.update_data();

		if(UPDATE_ALPHA)
			mod.update_alpha();

		alpha[i] = mod.alpha;
		loglik[i] = mod.get_loglik();

		if (i >= N_BURN && ((i - N_BURN) % N_THIN) == 0)
			z.col((i - N_BURN) / N_THIN) = mod.z;

		if ((i + 1) % (N_ITER / 100) == 0)
			progessbar((double)(i + 1) / N_ITER);
	}

	tend = time(0);
	cout << N_ITER << " iterations: " << difftime(tend, tstart) << "s" << endl;

	z.save(DAT_DIR + "z", arma::raw_ascii);
	loglik.save(DAT_DIR + "loglik", arma::raw_ascii);

	if (UPDATE_ALPHA)
		alpha.save(DAT_DIR + "alpha", arma::raw_ascii);

	cin.get();
}
